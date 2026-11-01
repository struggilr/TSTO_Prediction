"""Same V2 parameters/operations, accepting genuinely different chart heights.

No padded pixels enter the encoders. Padding is masked bookkeeping for the shared
physical losses. Cartesian and uniform-height inputs use V2 forward exactly.
"""
import torch
import torch.nn.functional as F
from .spatial_physics import SpatialPredictor
from .joint import future_xy,normalize_xy,solid_mask,sample_grid

class ContextProbe(SpatialPredictor):
    def __init__(self,trajectory_path,heights,**kwargs):
        super().__init__(trajectory_path,**kwargs);self.heights=list(heights);self.context_scale=.25
    def forward(self,history,mach,pose_history,dt,last_xy,templates,horizon,route,last_base=None,last_valid=None):
        b,t,k,_,hmax,w=history.shape
        if route=='cartesian' or len(set(self.heights))==1:
            return super().forward(history,mach,pose_history,dt,last_xy,templates,horizon,route,last_base,last_valid)
        if k!=4 or hmax!=max(self.heights):raise ValueError('Allocated chart shape mismatch')
        pose=self.trajectory(pose_history,dt,horizon)
        with torch.no_grad():
            xy=future_xy(last_xy,pose_history[:,-1],pose);nc=normalize_xy(xy)
            valid=(nc.abs()<=1+2e-6).all(-1)&~solid_mask(xy,pose,templates)
            for j,h in enumerate(self.heights):valid[:,:,j,h:]=False
            geo=torch.cat([valid[:,:,:,None].float(),nc.permute(0,1,2,5,3,4)],3)
            relative=(xy[:,:,:,None]-pose.reshape(b,horizon,2,3)[:,:,None,:,None,None,:2])/xy.new_tensor([110.,70.])
            relative=relative.permute(0,1,2,3,6,4,5).reshape(b,horizon,k,4,hmax,w)
            angle=(pose-pose_history[:,-1,None]).reshape(b,horizon,2,3)[...,2]
            angles=angle[:,:,None,:,None,None].expand(-1,-1,k,-1,hmax,w)
            lead=torch.arange(1,horizon+1,device=history.device,dtype=history.dtype)[None,:,None,None,None,None].expand(b,-1,k,1,hmax,w)/horizon
            motion=torch.cat([geo,relative,angles,lead],3)
            persistence=sample_grid(last_base[:,None].expand(-1,horizon,-1,-1,-1),xy)
            observed=(sample_grid(last_valid[:,None,None].expand(-1,horizon,-1,-1,-1),xy)>.999)&valid[:,:,:,None]
            persistence*=observed
        fine=[];maps=[];cond=self.condition((mach/7)[:,None])[:,None].expand(-1,t,-1).reshape(b*t,self.width,1,1)
        for j,h in enumerate(self.heights):
            f=self.fine(history[:,:,j,:,:h].reshape(b*t,8,h,w));z=F.gelu(self.down(f))+cond
            for spec,local in zip(self.spectral,self.local):z=F.gelu(z+spec(z)+local(z))
            fine.append(f);maps.append(z+self.roles.weight[j][None,:,None,None])
        results=[]
        for j,h in enumerate(self.heights):
            hc,wc=maps[j].shape[-2:];hh,ww=fine[j].shape[-2:]
            context=torch.stack([F.interpolate(z,size=(hc,wc),mode='bilinear',align_corners=False) for z in maps]).mean(0)
            z=maps[j]+self.context_scale*context
            tokens=F.adaptive_avg_pool2d(z,(4,4)).reshape(b,t,self.width,16).permute(0,3,1,2).reshape(b*16,t,self.width)
            encoded=self.temporal(tokens+self.position[None,:t])[:,-1]
            temporal=encoded.reshape(b,4,4,self.width).permute(0,3,1,2)
            hidden=z.reshape(b,t,self.width,hc,wc)[:,-1]+F.interpolate(temporal,size=(hc,wc),mode='bilinear',align_corners=False)
            skip=fine[j].reshape(b,t,self.width,hh,ww)[:,-1];outputs=[]
            for step in range(horizon):
                features=motion[:,step,j,:,:h];g=F.interpolate(features,size=(hc,wc),mode='bilinear',align_corners=False)
                hidden=self.advance(self.geometry_encoder(g),hidden)
                half=self.up_half(torch.cat([F.interpolate(hidden,size=(hh,ww),mode='bilinear',align_corners=False),skip],1))
                high=F.interpolate(half,size=(h,w),mode='bilinear',align_corners=False)
                prior=persistence[:,step,j,:,:h];known=observed[:,step,j,:,:h].float()
                outputs.append((prior+self.decode(torch.cat([high,features,prior,known],1)))*valid[:,step,j,None,:h])
            results.append(F.pad(torch.stack(outputs,1),(0,0,0,hmax-h)))
        return dict(fields=torch.stack(results,2),pose=pose,xy=xy,geometry=geo,persistence=persistence,chart_heights=self.heights)
