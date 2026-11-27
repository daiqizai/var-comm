"""Measure empty, sparse and full loads with physical image codewords and independent noise checks."""
import json
import math
import numpy as np
import torch
from scipy.stats import f as f_distribution
from study import CFG, setup, source, save, rows, finish
from access import prior, waveform, load_external, reference_stats, build_walsh_16_basis


def predict(score, threshold):
    admitted=score>2.8
    legacy=admitted.clone()
    empty=~legacy.any(dim=-1)
    legacy[empty,score[empty].argmax(-1)]=True
    return torch.stack([legacy,admitted,score>threshold])


@torch.inference_mode()
def main():
    out=setup("activity")
    threshold=float(f_distribution.isf(.01/4,128,256))
    save(out/"threshold.json",{"public_false_alarm_family_target":.01,"threshold":threshold,
            "null_numerator_df":128,"null_denominator_df":256,"fallback":2.8})
    rng=np.random.default_rng(281135)
    quiet=rng.chisquare(256,20000)/256
    score=torch.from_numpy((rng.chisquare(128,(20000,4))/128)/quiet[:,None])
    flags=predict(score,threshold).any(-1).sum(1).tolist()
    save(out/"noise_only.json",dict(frames=20000,false_frames=flags,
                     methods=["force_one","allow_empty","calibrated_F"]))
    model=load_external()
    images,ids=source("common500",out)
    basis=build_walsh_16_basis(12,384,device="cuda",dtype=torch.float32)
    events=[]
    # Masks have distinct public banks. Each bank is shared by all detectors and both power settings.
    masks=torch.tensor([[bool(i&(1<<u)) for u in range(4)] for i in range(16)],device="cuda")
    all_levels={"spread14":[18,13,8,4],"edge":[-6,-9,-12,-15]}
    qual=False
    for seed in CFG["multi"]["seeds"]:
        modem=prior.modem_for(seed)
        for st in range(0,125,4):
            sl=slice(st,min(st+4,125));n=sl.stop-st
            x=images[4*st:4*sl.stop].cuda().reshape(n,4,3,256,256).float()/255.
            tx=model.transmit(x,modem)
            for condition,levels in all_levels.items():
                for mask_id,mask in enumerate(masks):
                    # Use the producer bank's override for channel/noise after explicit per-mask seed offsets.
                    g=torch.Generator().manual_seed(97*seed+157*st+19009*mask_id)
                    from dir4_rn.noma_semsic_blind import sample_multiuser_uplink_channels
                    hr,hi=sample_multiuser_uplink_channels(n,4,2,32,.85,generator=g)
                    eps=torch.randn(n,2,32,384,2,generator=g)
                    power=torch.stack([torch.roll(torch.tensor(levels,dtype=torch.float32),(i+seed)%4) for i in range(st,sl.stop)])
                    y=waveform(tx,(hr,hi,eps,power),slice(0,n),mask.expand(n,4).float())
                    scores=reference_stats(y,basis)
                    preds=predict(scores,threshold)
                    if not qual and mask_id==0:
                        _,_,meta=modem.blind_demix_users(y,4,96,override_mode="linear_1stage")
                        expected=meta["probe_energy"]/meta["sigma_n_hat"][:,None]
                        assert (expected-scores).abs().max()<2.e-4 and torch.equal(preds[0],meta["active_hat"].bool())
                        qual=True
                    ints=(preds.to(torch.int64)*torch.tensor([1,2,4,8],device="cuda")).sum(-1).cpu().tolist()
                    for j in range(n):
                        events.append({"condition":condition,"draw":seed,"group":st+j,"mask":mask_id,
                              "force_one":ints[0][j],"allow_empty":ints[1][j],"calibrated_F":ints[2][j]})
            if st%32==0:print("mask bank",seed,st,flush=True)
    assert len(events)==8000 and qual
    rows(out/"mask_decisions.csv",events)
    finish(out,frames=len(events),detector_decisions=3*len(events),pure_noise_simulations=20000,
           image_transmit_user_vectors=1000,stock_mask_qualification=qual)


if __name__=="__main__":
    main()

