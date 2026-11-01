"""Physical-coordinate regional losses; labels are used only in supervised loss."""
import torch
import torch.nn.functional as F
from .joint import sample_grid

def physical_gradient(q,xy):
    # q [B,M,K,C,H,W], xy [...,H,W,2]; derivatives on the same logical stencil.
    col=lambda a:(a[...,1:-1,2:]-a[...,1:-1,:-2])*.5
    row=lambda a:(a[...,2:,1:-1]-a[...,:-2,1:-1])*.5
    x=xy[...,0];y=xy[...,1];xc=col(x)[:,:,:,None];xr=row(x)[:,:,:,None];yc=col(y)[:,:,:,None];yr=row(y)[:,:,:,None]
    det=xc*yr-xr*yc;safe=torch.where(det.abs()>1e-8,det,torch.ones_like(det))
    qc=col(q);qr=row(q)
    return (qc*yr-qr*yc)/safe,(-qc*xr+qr*xc)/safe,det.abs()>1e-8

def stencil(mask):
    return mask[...,1:-1,1:-1]&mask[...,2:,1:-1]&mask[...,:-2,1:-1]&mask[...,1:-1,2:]&mask[...,1:-1,:-2]

def regional_maps(target,valid):
    b,m,_,h,w=target.shape
    fluid=valid.reshape(b*m,1,h,w)
    # Solid-only dilation; zero padding must not invent solid beyond the exterior domain.
    near=F.max_pool2d(1-fluid,7,stride=1,padding=3)*fluid
    p=target[:,:,0];gx=F.pad((p[...,2:]-p[...,:-2])*.5,(1,1,0,0));gy=F.pad((p[...,2:,:]-p[...,:-2,:])*.5,(0,0,1,1))
    magnitude=(gx.square()+gy.square()+1e-12).sqrt().reshape(b*m,1,h,w)
    usable=fluid*(1-F.max_pool2d(1-fluid,3,1,1))
    shock=(magnitude>.15).float()*usable
    shock=F.max_pool2d(shock,5,1,2)*fluid
    return near.reshape(b,m,1,h,w),shock.reshape(b,m,1,h,w)

def physical_bin_weights(xy,valid):
    # Partition the quadrature weights among chart samples assigned to one common pixel.
    b,m,k,h,w,_=xy.shape;x=xy[...,0];y=xy[...,1]
    xc=torch.gradient(x,dim=(-2,-1));yc=torch.gradient(y,dim=(-2,-1))
    area=(xc[1]*yc[0]-xc[0]*yc[1]).abs().clamp_min(1e-8)*valid[:,:, :,0]
    ix=((x+10)*439/110).round().long().clamp(0,439);iy=((y+20)*279/70).round().long().clamp(0,279)
    bins=(iy*440+ix).reshape(b*m,-1);a=area.reshape(b*m,-1)
    totals=torch.zeros(b*m,280*440,device=xy.device);totals.scatter_add_(1,bins,a)
    weights=a/totals.gather(1,bins).clamp_min(1e-8)
    return weights.reshape(b,m,k,1,h,w)

class PhysicsLoss:
    def __init__(self,meta,thermo):self.meta=meta;self.thermo=thermo
    def __call__(self,out,batch,epoch=1):
        pred=out['fields'];xy=out['xy'].detach()
        target=sample_grid(batch['target'],xy)
        valid=(sample_grid(batch['target_valid'][:,:,None],xy)>.999)&out['geometry'][:,:,:,:1].bool()
        near,shock=regional_maps(batch['target'],batch['target_valid'])
        wall=sample_grid(near,xy).detach();edge=sample_grid(shock,xy).detach()
        with torch.no_grad():quad=physical_bin_weights(xy,valid.float())
        weight=valid*(1+4*wall+3*edge)*quad
        den=(5*weight.sum()).clamp_min(1)
        mse=((pred-target).square()*weight).sum()/den
        # Gradient matching preserves target jumps; it does not penalize gradients toward zero.
        gx,gy,geomok=physical_gradient(pred,xy);tx,ty,_=physical_gradient(target,xy)
        smask=stencil(valid)&geomok
        gw=smask*(1+2*edge[...,1:-1,1:-1])*quad[...,1:-1,1:-1]
        grad=((F.smooth_l1_loss(gx,tx,reduction='none')+F.smooth_l1_loss(gy,ty,reduction='none'))*gw).sum()/(10*gw.sum()).clamp_min(1)
        mean=pred.new_tensor(self.meta['mean'])[None,None,None,:,None,None];std=pred.new_tensor(self.meta['std'])[None,None,None,:,None,None]
        physical=pred*std+mean
        p,rho,temp,u,v=physical.unbind(3)
        eos=((p-self.thermo['gas_constant']*rho*temp-self.thermo['pressure_offset'])/self.meta['std'][0]).square()
        smoothvalid=valid[:,:,:,0].float()*quad[:,:,:,0]
        eos=(eos*smoothvalid).sum()/smoothvalid.sum().clamp_min(1) if self.thermo['enabled'] else mse*0
        positive=(F.relu(-rho/self.meta['std'][1]).square()+F.relu(-temp/self.meta['std'][2]).square())
        positive=(positive*smoothvalid).sum()/smoothvalid.sum().clamp_min(1)
        continuity=self.continuity(out,batch,physical,valid,edge,wall,quad)
        consistency=self.consistency(pred,xy,valid,quad,batch)
        ramp=min(1.,epoch/10)
        loss=mse+.03*grad+ramp*(.01*eos+.01*positive+.001*continuity+.01*consistency)
        parts=dict(mse=float(mse.detach()),gradient=float(grad.detach()),eos=float(eos.detach()),positivity=float(positive.detach()),continuity=float(continuity.detach()),wall_fraction=float((wall*valid).sum().detach()/valid.sum().clamp_min(1)),shock_fraction=float((edge*valid).sum().detach()/valid.sum().clamp_min(1)))
        parts['overlap_consistency']=float(consistency.detach())
        for k in range(pred.shape[2]):parts[f'chart_{k}_mse']=float((((pred[:,:,k]-target[:,:,k]).square()*weight[:,:,k]).sum()/(5*weight[:,:,k].sum()).clamp_min(1)).detach())
        return loss,parts

    def consistency(self,pred,xy,valid,quad,batch):
        b,m,k,c,h,w=pred.shape
        if k==1:return pred.sum()*0
        # Sparse physical-bin agreement. Smooth regions only: avoid averaging shifted shocks.
        ix=((xy[...,0]+10)*439/110).round().long().clamp(0,439);iy=((xy[...,1]+20)*279/70).round().long().clamp(0,279)
        bins=(iy*440+ix).reshape(b*m*k,1,h*w);q=(quad*valid).reshape(b*m*k,1,h*w)
        den=pred.new_zeros(b*m*k,1,280*440).scatter_add(2,bins,q)
        sums=pred.new_zeros(b*m*k,c,280*440).scatter_add(2,bins.expand(-1,c,-1),pred.reshape(b*m*k,c,h*w)*q)
        values=(sums/den.clamp_min(1e-8)).reshape(b,m,k,c,280*440);covered=(den>0).reshape(b,m,k,1,280*440)
        count=covered.sum(2);average=(values*covered).sum(2)/count.clamp_min(1)
        near,shock=regional_maps(batch['target'],batch['target_valid']);smooth=(near+shock<.01).reshape(b,m,1,280*440)
        use=covered*(count[:, :,None]>1)*smooth[:,:,None]
        return ((values-average[:,:,None]).square()*use).sum()/(c*use.sum()).clamp_min(1)

    def continuity(self,out,batch,current,valid,shock,wall,quad):
        # Smooth-region ALE chain-rule residual. No strong-form enforcement at shocks.
        pred=out['fields'];mean=pred.new_tensor(self.meta['mean'])[None,None,:,None,None];std=pred.new_tensor(self.meta['std'])[None,None,:,None,None]
        last=batch['history'][:,-1,:,:5]*std+mean
        previous=torch.cat([last[:,None],current[:,:-1]],1)
        prevxy=torch.cat([batch['last_xy'][:,None],out['xy'][:,:-1]],1).detach()
        prevvalid=torch.cat([batch['history'][:,-1:,:,5:6]>.5,valid[:,:-1]],1)
        midpoint=.5*(current+previous);midxy=.5*(out['xy'].detach()+prevxy)
        rho=midpoint[:,:,:,1:2];flux=torch.cat([rho*midpoint[:,:,:,3:4],rho*midpoint[:,:,:,4:5]],3)
        rx,ry,geom=physical_gradient(rho,midxy);fx,fy,_=physical_gradient(flux,midxy)
        dt=batch['dt'][:,-1,None,None,None,None]
        drho=((current[:,:,:,1]-previous[:,:,:,1])/dt)[...,1:-1,1:-1]
        velocity=(out['xy'].detach()-prevxy)/dt[...,None]
        divergence=fx[:,:,:,0]+fy[:,:,:,1]
        residual=drho-velocity[...,1:-1,1:-1,0]*rx[:,:,:,0]-velocity[...,1:-1,1:-1,1]*ry[:,:,:,0]+divergence
        residual=residual/(self.meta['std'][1]/dt)
        mask=stencil(valid&prevvalid)[:,:,:,0]&geom[:,:,:,0]
        mask=mask&(shock[:,:,:,0,1:-1,1:-1]<.01)&(wall[:,:,:,0,1:-1,1:-1]<.01)
        weight=mask*quad[:,:,:,0,1:-1,1:-1]
        return (F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none')*weight).sum()/weight.sum().clamp_min(1)
