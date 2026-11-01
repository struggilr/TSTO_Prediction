"""Adapted from FNOTransfomerCDDIM_V2.0: module"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.fft

class SpectralConv2d(nn.Module):
    """Adapted from FNOTransfomerCDDIM_V2.0: SpectralConv2d"""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.float32))
        self.weight_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.float32))

    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum('bixy,ioxy->boxy', input, weights)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        x_ft = torch.fft.fft2(x_fp32)
        out_ft = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=torch.cfloat)
        weight_complex = torch.complex(self.weight_real, self.weight_imag)
        out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(x_ft[:, :, :self.modes, :self.modes], weight_complex)
        out = torch.fft.ifft2(out_ft).real
        return out.to(orig_dtype)

class ConditionedFiLM(nn.Module):
    """Adapted from FNOTransfomerCDDIM_V2.0: ConditionedFiLM"""

    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * feat_dim)

    def forward(self, x: Tensor, cond_emb: Tensor) -> Tensor:
        gamma_beta = self.fc(cond_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x * (1 + gamma) + beta

class FNOEncoder(nn.Module):
    """Adapted from FNOTransfomerCDDIM_V2.0: FNOEncoder"""

    def __init__(self, in_channels: int, cond_dim: int, modes: int, width: int, out_dim: int, num_layers: int):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.num_layers = num_layers
        self.in_conv = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.film = ConditionedFiLM(width, cond_dim)
        self.spectral_blocks = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(num_layers)])
        self.res_blocks = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)])
        self.norm = nn.LayerNorm([width, 1, 1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_linear = nn.Linear(width, out_dim)

    def forward(self, x: Tensor, cond_emb: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.film(x, cond_emb)
        for spec, res in zip(self.spectral_blocks, self.res_blocks):
            y = spec(x)
            x = x + y + res(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.out_linear(x)
        return x
