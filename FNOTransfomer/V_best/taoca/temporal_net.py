"""Adapted from FNOTransfomerCDDIM_V2.0: module"""
import torch
import torch.nn as nn
from torch import Tensor

class TemporalPredictor(nn.Module):
    """Adapted from FNOTransfomerCDDIM_V2.0: TemporalPredictor"""

    def __init__(self, latent_dim: int, nhead: int=4, n_layers: int=2, dropout: float=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.register_buffer('pos_emb', torch.randn(1, 200, latent_dim))

    def forward(self, z_history: Tensor, pred_len: int, teacher_forcing: bool=False, target_z: Tensor=None, ratio: float=1.0) -> Tensor:
        """Adapted from FNOTransfomerCDDIM_V2.0: forward"""
        B, n, dim = z_history.shape
        outputs = []
        inputs = z_history
        for t in range(pred_len):
            seq_len = inputs.shape[1]
            pos = self.pos_emb[:, :seq_len, :]
            x = self.transformer(inputs + pos)
            next_z = x[:, -1, :]
            outputs.append(next_z)
            if teacher_forcing and target_z is not None and (torch.rand(1).item() < ratio):
                next_input = target_z[:, t]
            else:
                next_input = next_z
            inputs = torch.cat([inputs, next_input.unsqueeze(1)], dim=1)[:, 1:, :]
        z_pred = torch.stack(outputs, dim=1)
        return z_pred
