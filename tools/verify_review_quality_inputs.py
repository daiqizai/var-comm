"""Real-weight timing check against the stored normal quality reconstructions."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from latent_followup.timing_clean import configure,load_arm,execute_method,OUT_ROOT
from latent_enhancement.runtime import model_paths,write_json,digest,require_available
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_eval.runner import load_targets
from var_comm.next_scale_prior import load_models

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();configure();require_available();device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device)
    arms={n:load_arm(n,(32,16,16),scale,device) for n in ('enhancement512','enhancement1024','receiver_only_refiner')};targets=load_targets();rows=[]
    for i in (0,11):
        path=OUT_ROOT/f'development_eval_v1/images/{i:03d}/reconstructions.npz'
        with np.load(path,allow_pickle=False) as data:
            names=data['method_order'].tolist();images=data['images']
            for method in ('m8_receiver_only_refiner','m8_plus_latent_512','m8_plus_latent_1024'):
                for si,snr in enumerate((1.,4.,7.,13.,19.)):
                    got=execute_method(targets[i],method,snr,2001,vae,var,decoder,arms,device)['image'];old=images[names.index(method),si*3]
                    error=float(np.abs(got-old).max());rows.append({'source_index':i,'method':method,'snr_db':snr,'stored_quality_max_abs':error})
    write_json(a.output,{'checks':rows,'status':'PASS' if max(r['stored_quality_max_abs'] for r in rows)<=2e-5 else 'MISMATCH_REQUIRES_INVESTIGATION','verifier_sha256':digest(__file__),'input':'existing normal-quality per-source reconstructions; seed2001'})
    if max(r['stored_quality_max_abs'] for r in rows)>2e-5:raise RuntimeError('stored normal-quality mismatch')
if __name__=='__main__':main()
