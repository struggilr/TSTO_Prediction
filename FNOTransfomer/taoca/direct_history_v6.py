"""Only the historical chart-field sampling changes from V3; labels stay fixed."""
import json
from pathlib import Path
import numpy as np
import torch
import h5py
from .allocated_data_v3 import AllocatedWindows

class DirectHistoryWindows(AllocatedWindows):
    _direct={}
    def __init__(self,*args,direct_root=None,**kwargs):
        super().__init__(*args,**kwargs);self.direct_root=Path(direct_root or self.root.parent/'taoca_direct_history_v6')
        manifest=json.loads((self.direct_root/'manifest.json').read_text())
        if manifest['heights']!=self.heights or not manifest['verified']:raise RuntimeError('Direct-history cache is not verified')
    def __getitem__(self,index):
        item=super().__getitem__(index)
        if self.route=='cartesian':return item
        case,s=self.samples[index];key=(str(self.direct_root.resolve()),case)
        if key not in self._direct:
            with h5py.File(self.direct_root/(case+'.h5')) as f:
                self._direct[key]=[(torch.from_numpy(f[f'chart{k}/fields'][:]),torch.from_numpy(f[f'chart{k}/valid'][:].astype(bool))) for k in range(4)]
        # Slice exactly [s-history,s). Target frames are never put in history.
        for k,h in enumerate(self.heights):
            field,valid=self._direct[key][k];mask=(item['history'][:,k,5,:h]>.5)&valid[s-self.history:s]
            item['history'][:,k,:5,:h]=field[s-self.history:s]*mask[:,None]
            item['history'][:,k,5,:h]=mask.float()
        return item
