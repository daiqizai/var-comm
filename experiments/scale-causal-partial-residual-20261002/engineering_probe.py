"""One calibration source end-to-end runner check, never a scientific result."""
import time
import torch
import common as c
import m1_runner as runner
from var_comm.quality import dino_features

def main():
    loaded=c.setup();d=c.assets.load_calibration();r=d['records'][0]
    with torch.inference_mode():
        xvae=torch.as_tensor(r['pixels'][None],dtype=torch.float32,device=loaded['device'])/127.5-1
        online_f=loaded['vae'].quant_conv(loaded['vae'].encoder(xvae))
        online_t=torch.cat(loaded['vae'].quantize.f_to_idxBl_or_fhat(online_f,to_fhat=False),1)[0].cpu()
        if not torch.equal(online_t,d['T'][0]):raise RuntimeError('Online TX differs from registered cached tokens')
        f_max_error=float((online_f[0].cpu()-d['F'][0]).abs().max())
        if not torch.allclose(online_f[0].cpu(),d['F'][0],atol=1e-5,rtol=1e-5):raise RuntimeError('Online TX differs from registered F')
        x=torch.as_tensor(r['pixels'][None],dtype=torch.float32,device=loaded['device'])/255
        d['reference']=dino_features(loaded['dino'],x).cpu()
        tm=dict(waves={},logits={});rm={};mm={};rows=[];start=time.time()
        for snr in c.SNRS:
            for a in runner.legal_grid():
                c.check();row,_=runner.score_one(loaded,d,0,a,snr,4101,tm,rm,mm)
                rows.append(row)
            c.status('m1_engineering_probe',snr=snr,frames=len(rows),elapsed_seconds=time.time()-start)
        summary=runner.summarize(rows)
        for N in [512,1024]:
            for fam in ['QPSK','16QAM']:
                for snr in c.SNRS:
                    for method in runner.METHODS:
                        group=[r for r in summary if (r['N'],r['phy_family'],r['snr_db'])==(N,fam,snr) and (r['q']==0 or r['order']==method) and (method!='whole' or r['q']==0)]
                        ranked,ref=runner.rank(group)
                        assert ranked and all(r['psnr_db']>=ref['psnr_db']-.25 for r in ranked)
        c.assert_frozen(loaded)
        c.write(c.OUT/'engineering_probe.json',dict(status='REAL_RUNNER_ENGINEERING_PASS',scientific_result=False,development_read=False,calibration_sources=1,
            frames=len(rows),elapsed_seconds=time.time()-start,unique_received_latents=len(rm),unique_scored_latents=len(mm),
            training_updates=0,online_cached_tokens_equal=True,online_F_max_error=f_max_error,source_bindings=c.source_bindings()))

if __name__=='__main__':main()
