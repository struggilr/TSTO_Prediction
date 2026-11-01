"""Check actual CUDA matrix multiplication and FFT on each requested device."""
import argparse
import json
import sys
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--devices',type=int,default=1)
    args=parser.parse_args()
    available=torch.cuda.is_available();count=torch.cuda.device_count()
    print(json.dumps(dict(python=sys.executable,torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=available,visible_devices=count)),flush=True)
    if not available:
        raise RuntimeError('CUDA initialization failed in the Python environment printed above. Check CUDA build against driver; training will not fall back to CPU.')
    if count<args.devices:
        raise RuntimeError(f'Requested {args.devices} GPUs but only {count} visible. Use run_joint_nohup.sh for a single GPU.')
    for index in range(args.devices):
        with torch.cuda.device(index):
            x=torch.randn(64,64,device=f'cuda:{index}')
            y=x@x
            z=torch.fft.rfft2(x)
            (y.square().mean()+z.abs().mean()).item()
            torch.cuda.synchronize(index)
            print(json.dumps(dict(device=index,name=torch.cuda.get_device_name(index),capability=torch.cuda.get_device_capability(index),vram_gib=torch.cuda.get_device_properties(index).total_memory/2**30,torch=torch.__version__,cuda=torch.version.cuda,matmul_fft='passed')),flush=True)


if __name__=='__main__':main()
