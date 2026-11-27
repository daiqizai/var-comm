"""Use normalized received image statistics as a stronger source-scale control."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from frontend import packet, receive
from single_link import (DEV_SEED, basis, batches, codec, draw_identity, draws, latents, power,
                         render, retrieve, rgb, source_scale)
from study import CFG, HERE, digest, finish, quality, rows, save, scorer, setup, source

ARMS=("inferred","signaled_and_inferred")


def features(state,signal):
    _,ch,snr,h = state
    z=ch.u_norm
    amp=ch.pwr.sqrt() if signal else torch.zeros_like(snr)
    values=torch.cat([z.mean(1),z.square().mean(1).sqrt(),h.abs().mean(1,keepdim=True),
                      h.abs().std(1,keepdim=True),snr[:,None],amp[:,None]],dim=1)
    assert values.shape[1]==196
    return values.cpu().numpy().astype(np.float64)


def fit(mat,target):
    mean=mat.mean(0)
    std=mat.std(0)
    std[std<1e-8]=1.
    z=(mat-mean)/std
    shift=float(target.mean())
    weights=np.linalg.solve(z.T@z+len(z)*.1*np.eye(z.shape[1]),z.T@(target-shift))
    return dict(mean=mean.tolist(),std=std.tolist(),weights=weights.tolist(),intercept=shift)


def predict(matrix,config):
    v=(matrix-config["mean"])/config["std"]
    p=np.clip(v@config["weights"]+config["intercept"],.4,7.)
    return torch.tensor(p,dtype=torch.float32,device="cuda")


@torch.inference_mode()
def main():
    out=setup("source_predictor")
    base=HERE/"frontend"
    frozen=json.loads((base/"freeze.json").read_text())
    mu=frozen["empirical_scale_mean_var"][0]
    model=codec("continuous")
    refs=basis()
    dists=scorer()
    cfgs={}
    preds=[]
    for split in ("val256","common500"):
        im,ids=source(split,out)
        assert (out/(split+".json")).read_bytes()==(base/(split+".json")).read_bytes()
        z=latents(model,im)
        dev=split=="val256"
        obs={k:[] for k in ARMS}
        targets=[]
        seeds=[DEV_SEED] if dev else CFG["seeds"]
        identity=frozen["bank"] if dev else json.loads((base/"test_banks.json").read_text())
        for seed in seeds:
            bank=draws(len(ids),seed)
            assert draw_identity(bank)==(identity if dev else identity[str(seed)])
            for snr in (4,10):
                for sl in batches(len(ids)):
                    zz=z[sl].cuda()
                    h,noise=(v[sl].cuda() for v in bank[:2])
                    if dev:targets.extend(source_scale(zz).cpu().tolist())
                    for method in ARMS:
                        signal=method=="signaled_and_inferred"
                        frame=packet("spread",model,zz,refs,(),constant=None if signal else mu)
                        assert (power(frame)-1).abs().max()<4e-5
                        state=retrieve(model,receive(frame,h,noise,snr,"spread",()))
                        v=features(state,signal)
                        if dev:
                            obs[method].append(v)
                            continue
                        scale=predict(v,cfgs[method])
                        ps,ds=quality(rgb(im,sl),render(model,state,scale),dists)
                        delta=(scale-source_scale(zz)).abs().cpu().tolist()
                        preds.extend(dict(snr=snr,draw=seed,source=sl.start+j,arm=method,psnr=float(p),dists=float(d),scale_mae=e)
                                     for j,(p,d,e) in enumerate(zip(ps,ds,delta)))
        if dev:
            for name in ARMS:
                cfgs[name]=fit(np.concatenate(obs[name]),np.asarray(targets))
            save(out/"freeze.json",{"parent":digest(base/"freeze.json"),"coefficients":cfgs,"fit_images":256,"observations_per_arm":512})
    assert len(preds)==4000
    rows(out/"events.csv",preds)
    from analyze import scope,tensor,CONTRASTS,SUMMARY
    with (base/"common500_events.csv").open() as f:
        parent=[r for r in csv.DictReader(f) if r["arm"] in ("spread_raw","constant")]
    assert len(parent)==4000
    metrics=("psnr","dists","scale_mae")
    pairs=(("inferred","constant"),("spread_raw","inferred"),("signaled_and_inferred","inferred"),("signaled_and_inferred","spread_raw"))
    for snr in ("4","10"):
        z=tensor([r for r in parent+preds if str(r["snr"])==snr],metrics,lambda r:r["arm"],"source",500,2)
        scope("source_scale_secondary",snr,z,metrics,pairs)
    rows(out/"summary.csv",SUMMARY);rows(out/"paired.csv",CONTRASTS)
    finish(out,new_test_image_decodes=4000,exact_baseline_scores_reused=4000,new_eval_source_encodings=756,
           training_neural_updates=0,validation_ridge_fits=2,stage_protocol=digest(HERE/"source_predictor_protocol.json"))


if __name__=="__main__":
    main()

