"""Prepare an isolated metric-conditioned candidate without changing active V7."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    text=(ROOT/'taoca/allocated_model_v3.py').read_text(encoding='utf-8')
    text=text.replace('from .spatial_physics import SpatialPredictor','from .spatial_physics import SpatialPredictor\nfrom .metric_features_v8 import MetricEncoder')
    text=text.replace('class AllocatedPredictor(SpatialPredictor):','class MetricPredictor(SpatialPredictor):')
    # Preserve old parameter names; only the new branch introduces missing keys.
    text=text.replace('self.heights=list(heights)','self.heights=list(heights);self.metric_encoder=MetricEncoder(self.width)')
    text=text.replace("f=self.fine(history[:,:,j,:,:h].reshape(b*t,8,h,w));z=F.gelu(self.down(f))+cond", "observed=history[:,:,j,:,:h].reshape(b*t,8,h,w);f=self.fine(observed);f=f+self.metric_encoder(observed,f.shape[-2:]);z=F.gelu(self.down(f))+cond")
    # Do not shadow the future-persistence validity mask named 'observed'.
    text=text.replace('observed=history[:,:,j,:,:h].reshape(b*t,8,h,w);f=self.fine(observed);f=f+self.metric_encoder(observed,f.shape[-2:])','native_history=history[:,:,j,:,:h].reshape(b*t,8,h,w);f=self.fine(native_history);f=f+self.metric_encoder(native_history,f.shape[-2:])')
    (ROOT/'taoca/metric_model_v8.py').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
