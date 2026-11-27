"""Receiver-only phase slope inference with fixed power and a stationary negative control."""
import math

import numpy as np
import torch

from single_link import (DEV_SEED, MODELS, basis, batches, codec, draw_identity, draws, latents,
                         nmse, power, render, retrieve, rgb, tx)
from study import CFG, finish, quality, rows, save, scorer, setup, source

GRID = [k*.15 for k in range(-10,11)]
DETECTORS = ("reference", "reference_minus_null")


def trial_scores(y, references, angles):
    out = []
    for phasor in angles:
        coeff = (y*phasor) @ references[:2].T.contiguous()
        energy = coeff.abs().square().mean(1)
        out.append(torch.stack((energy[:,0], energy[:,0]-energy[:,1]), 1))
    scores = torch.stack(out, dim=1)
    best, idx = scores.max(1)
    strength = (best-scores[:,10]) / scores[:,10,0:1].clamp_min(1e-6)
    return strength, idx


@torch.inference_mode()
def main():
    out = setup("drift")
    refs = basis()
    grid = torch.tensor(GRID, device="cuda")
    positions = torch.linspace(-.5,.5,384,device="cuda")
    derotate = torch.exp(-1j * grid[:,None]*positions)
    assert torch.allclose(derotate[10], torch.ones_like(positions, dtype=torch.complex64))
    val, _ = source("val256", out)
    dev_bank = draws(len(val), DEV_SEED)
    configs = {}
    for name in MODELS:
        model = codec(name)
        z = latents(model,val)
        energies = torch.cat([power(tx(model,z[sl].cuda())).cpu() for sl in batches(len(z))])
        gain = 1./math.sqrt(float(energies.mean())) if name=="qam64" else 1.
        checks = []
        for snr in CFG["siso"]["snr_db"]:
            for sl in batches(len(z)):
                h,noise = (v[sl].cuda() for v in dev_bank[:2])
                frame = tx(model,z[sl].cuda())
                y = frame*h[:,:,None] + noise*(10**(-snr/20)/gain)
                checks.append(trial_scores(y,refs,derotate)[0].cpu())
        thresholds = torch.quantile(torch.cat(checks), .95, dim=0).tolist()
        configs[name] = dict(gain=gain, native_mean_energy=float(energies.mean()), thresholds=thresholds,
                             model_sha256=MODELS[name][2])
        del model,z
    save(out/"freeze.json", {"criteria":"pooled stationary validation 95th percentile, by modem and objective",
                       "dev_bank":draw_identity(dev_bank),"configurations":configs,"grid_rad":GRID})
    pixels, ids = source("common500",out)
    dists = scorer()
    evaluations = 0
    bank_hashes = {}
    for name in MODELS:
        model = codec(name)
        z = latents(model,pixels)
        cfg = configs[name]
        gain = cfg["gain"]
        threshold = torch.tensor(cfg["thresholds"],device="cuda")
        records = []
        for seed in CFG["seeds"]:
            bank = draws(len(ids),seed)
            bank_hashes[str(seed)] = draw_identity(bank)
            for snr in CFG["siso"]["snr_db"]:
                for sweep in (0.,1.2):
                    for sl in batches(len(ids)):
                        h, noise, sign = (v[sl].cuda() for v in bank)
                        x = rgb(pixels,sl)
                        packet = tx(model,z[sl].cuda())
                        actual = sweep*sign
                        phase = torch.exp(1j*actual[:,None]*positions)
                        y = packet*h[:,:,None]*phase[:,None,:] + noise*(10**(-snr/20)/gain)
                        strength, idx = trial_scores(y,refs,derotate)
                        corrections = torch.where(strength>threshold,idx,10)
                        choices = [torch.full_like(idx[:,0],10),corrections[:,0],corrections[:,1]]
                        cache = []
                        for method,k in zip(("unchanged",*DETECTORS),choices):
                            # An identical vector of per-image decisions produces exactly the same IQ and RGB.
                            match = next((vals for old,vals in cache if torch.equal(old,k)),None)
                            if match is None:
                                state = retrieve(model,y*derotate[k,None,:])
                                pred = render(model,state)
                                ps,ds = quality(x,pred,dists)
                                match = (ps,ds,nmse(state[3],h).cpu().numpy())
                                cache.append((k,match))
                                evaluations += len(k)
                            ps,ds,err = match
                            slope = grid[k]
                            per_image = torch.stack([slope,actual,(slope-actual).abs(),power(packet)*gain**2],1).cpu().tolist()
                            for j,(est,truth,ae,energy) in enumerate(per_image):
                                records.append(dict(modem=name,snr=snr,sweep=sweep,draw=seed,source=sl.start+j,
                                      receiver=method,psnr=float(ps[j]),dists=float(ds[j]),nmse=float(err[j]),
                                      estimated_rad=est,true_rad=truth,abs_rad=ae,corrected=int(abs(est)>1e-7),energy=energy))
                    print(name,seed,snr,sweep,flush=True)
        rows(out/(name+"_events.csv"),records)
        del model,z
    save(out/"test_banks.json",bank_hashes)
    finish(out,logical_reconstructions=len(ids)*2*2*2*2*3,unique_batched_image_decodes=evaluations,
           receiver_inputs="received IQ only; single QAM gain from validation, known by both ends",
           image_encoder_examples=len(MODELS)*(len(val)+len(pixels)))


if __name__=="__main__":
    main()

