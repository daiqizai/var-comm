"""Training-only uncentered second-moment projection; no free per-image side data."""
import argparse,time
from pathlib import Path
import torch
from latent_enhancement.training import LatentPopulation
from latent_enhancement.runtime import digest,save_torch,write_json
from latent_enhancement_b.common import CACHE

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.time();pop=LatentPopulation(CACHE,'train');assert len(pop)==20000
    moment=torch.zeros(8192,8192,dtype=torch.float64)
    for offset in range(0,len(pop),256):
        residual=(pop.values['F'][offset:offset+256]-pop.values['Fb_TX'][offset:offset+256]).reshape(-1,8192).double();moment.addmm_(residual.T,residual)
    moment/=len(pop);values,vectors=torch.linalg.eigh(moment);A=vectors[:,-1984:].float();torch.testing.assert_close(A.T@A,torch.eye(1984),atol=2e-5,rtol=1e-5);save_torch(out/'A.pt',A)
    write_json(out/'projection_identity.json',{'status':'TRAINING_ONLY_SECOND_MOMENT_BASIS_READY','sources':20000,'source_role':'train','centering':'none; uncentered E[(F-Fb_TX)(F-Fb_TX)^T]; no RX mean or per-frame norm supplied for free','dimensions':[8192,1984],'A_sha256':digest(out/'A.pt'),'explained_training_second_moment_fraction':float(values[-1984:].sum()/values.sum()),'no_quality_claim_from_explained_fraction':True,'source_sha256':digest(__file__),'cache_receipts':{str(p):digest(p) for p in sorted((CACHE/'train').glob('shard_*.json'))},'elapsed_seconds':time.time()-start})
if __name__=='__main__':main()
