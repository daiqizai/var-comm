"""Actual-weight replay of training and online execution before bulk updates."""
import copy
import numpy as np,torch
from latent_enhancement.runtime import model_paths,perceptual_model,write_json,digest,require_available
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.model import render_received
from var_comm.next_scale_prior import load_models,state_sha256
from .common import configure_runtime,OUT,identity_files
from .models import matched_models
from .data import Population
from .train import update
from .execution import execute,transmit,receive
from .protocol import channel,config

@torch.no_grad()
def compare(model,b,index,record,snr,seed,vae,var,decoder,device):
    part={k:v[index:index+1] for k,v in b.items()};z,w=model(part)
    cached=render_received(decoder,z,part['status'])[0].cpu().numpy()
    actual,info=execute(record,model,snr,seed,vae,var,decoder,device)
    max_error=float(np.max(np.abs(actual-cached)))
    if max_error>2e-5:raise RuntimeError(f'cached training vs online quality mismatch {max_error}')
    np.testing.assert_allclose(info['tx']['F'].cpu().numpy(),part['F'].cpu().numpy(),atol=2e-5,rtol=1e-5)
    np.testing.assert_allclose(info['waveform'][68+model.ledger['ND']:],w[0].cpu().numpy(),atol=2e-5,rtol=1e-5)
    # Calling the same frozen path again changes timing, never conditions/noise/results.
    replay,_=execute(record,model,snr,seed,vae,var,decoder,device)
    np.testing.assert_array_equal(actual,replay)
    return max_error

def main():
    configure_runtime();require_available();device=torch.device('cuda:0');cfg=config()
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);lp=perceptual_model(device);scale=scale_statistics(device)
    frozen=state_sha256(decoder);rows=[]
    for m in (6,7,8):
        pop=Population('calibration',m,preflight=True);models=matched_models(m,scale.cpu(),cfg['initialization_seed']).to(device)
        if m==8:del models['H8-P']
        opts={n:torch.optim.AdamW(v.parameters(),lr=2e-4) for n,v in models.items()}
        ids=torch.arange(4);si=torch.tensor([0,0,3,3]);ni=torch.zeros(4,dtype=torch.long)
        b=pop.batch(ids,si,ni,[4101]*4,device)
        update(models,opts,b,decoder,lp,scale);update(models,opts,b,decoder,lp,scale)
        for name,model in models.items():
            for j in range(4):
                record={'pixels':pop.images[j].numpy(),'class_index':int(pop.labels[j]),'image_id':pop.ids[j]}
                error=compare(model.eval(),b,j,record,float(pop.snrs[si[j]]),4101,vae,var,decoder,device)
                rows.append({'arm':name,'source_index':j,'snr_db':float(pop.snrs[si[j]]),'maximum_image_error':error})
        print('execution qualified',m,flush=True)
    assert state_sha256(decoder)==frozen
    write_json(OUT/'qualification_execution.json',{'status':'REAL_CACHED_TRAINING_ONLINE_QUALITY_AND_REPLAY_PASS','comparisons':rows,'probe_updates_discarded':True,'decoder_unchanged':True,'bindings':identity_files([__file__,__import__('short_prefix.execution',fromlist=['x']).__file__,__import__('short_prefix.train',fromlist=['x']).__file__,__import__('short_prefix.data',fromlist=['x']).__file__])})
if __name__=='__main__':main()
