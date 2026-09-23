"""Real N3060 cache/online/clean-PHY acceptance before bulk training."""
import argparse,torch
from latent_enhancement.runtime import require_available,model_paths,write_json,ResourceBusy
from latent_enhancement_b.common import load_decoder,scale_statistics
from var_comm.next_scale_prior import load_models
from short_prefix.models import matched_models
from short_prefix.common import identity_files
from .common import configure_runtime,OUT
from .C_evaluate import qualify_models
from .evaluation_io import SafeEvaluation

def main():
    p=argparse.ArgumentParser();p.add_argument('--m',type=int,choices=[6,7],required=True);a=p.parse_args()
    configure_runtime();require_available();safe=SafeEvaluation();device=torch.device('cuda:0')
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device)
    models=matched_models(a.m,scale.cpu(),2026092304,N=3060).to(device).eval()
    metadata={n:{'kind':'hybrid','N':3060} for n in models}
    checks=qualify_models(models,metadata,vae,var,decoder,scale,device,safe)
    write_json(OUT/f'C_followups/N3060_m{a.m}_acceptance.json',{'status':'REAL_N3060_PRETRAIN_CACHE_ONLINE_CLEAN_RX_PASS','synthetic':False,'checks':checks,'bindings':identity_files([__file__,__import__('token_efficiency.C_evaluate',fromlist=['x']).__file__])})
if __name__=='__main__':
    try:main()
    except ResourceBusy as exc:print(str(exc),flush=True);raise SystemExit(75)
