"""Offline official DISTS/DreamSim and existing LPIPS-Alex, native float input."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
import torch
import torch.nn.functional as F
from training_core import verified

def module(spec,name):
    path=verified(spec);sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

class MetricSuite:
    def __init__(self,spec):
        self.spec=spec;self.met=module(spec['loader'],'_rdvq_registered_metric_loader')
        sys.path.insert(0,spec['dependency_overlay'])
        sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/tools')
        import checkpoint_relocation_20261028 as relocation
        four=module(spec['original_four'],'_rdvq_existing_four_metric_definitions');c=spec['four_config']
        assets=relocation.metric_assets(four,c);import lpips
        loader,self.lp_pair,proof=four.lpips_projection(c['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
        with self.met.no_network():
            self.lp=loader({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0')
            self.dists=self.met.load_dists(spec['dists'],torch.device('cuda:0'))
            self.dream=self.met.load_dreamsim(spec['dreamsim'],torch.device('cuda:0'))
        self.feature_cache={}

    @torch.no_grad()
    def feature(self,x):
        key=hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if key not in self.feature_cache:
            self.feature_cache[key]=self.dream.embed(self.met.preprocess_dreamsim(x)).detach()
        return self.feature_cache[key]

    @torch.no_grad()
    def score(self,reference,reconstruction):
        for x in [reference,reconstruction]:
            assert x.dtype==np.float32 and x.shape==(3,256,256) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
        xt=torch.from_numpy(reference[None]).cuda();yt=torch.from_numpy(reconstruction[None]).cuda()
        mse=float(np.mean((reference.astype(np.float64)-reconstruction.astype(np.float64))**2))
        return dict(psnr_db=float(-10*np.log10(max(mse,1e-12))),lpips_alex=float(self.lp_pair(self.lp,yt,xt)[0].item()),
                    dists=float(self.dists(xt,yt,require_grad=False,batch_average=False).reshape(-1)[0]),
                    dreamsim=float((1-F.cosine_similarity(self.feature(xt),self.feature(yt),dim=-1))[0]))
