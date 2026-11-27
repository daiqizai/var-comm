"""Equal resource layout, source-scale signaling and reference-energy controls."""
import math

import numpy as np
import torch

from single_link import (DEV_SEED, basis, batches, codec, draw_identity, draws, latents,
                         nmse, power, render, retrieve, rgb, source_scale, theta, tx)
from study import CFG, finish, quality, rows, save, scorer, setup, source

LAYOUTS = ("spread","dedicated","superposed")
SHARES = ("native","share20","share40")


def reflect(x,v):
    return x - 2*(x*v.conj()).sum(-1,keepdim=True)*v


def frame_transform(refs):
    vectors = []
    current = refs.clone()
    target = torch.zeros_like(refs)
    for i,k in enumerate((0,191,383)):
        target[i,k] = 1.
        v = current[i]-target[i]
        v = v/v.norm()
        current = reflect(current,v)
        vectors.append(v)
    assert (current-target).abs().max()<3e-5
    return vectors


def rotate(x,vs):
    for v in vs:
        x = reflect(x,v)
    return x


def packet(layout,model,z,refs,vs,constant=None):
    wave = tx(model,z)
    if layout=="dedicated":
        return rotate(wave,vs)
    if layout=="spread" and constant is None:
        return wave
    m = model.modem
    amp1,amp3 = (384*m.eta1).sqrt(), math.sqrt(384*m.eta3)
    p = source_scale(z)
    ph = torch.exp(1j*theta(p))[:,None,None]
    if constant is not None:
        fixed = torch.exp(1j*theta(torch.full_like(p,constant)))[:,None,None]
        return wave+amp3*(fixed-ph)*refs[2]
    # Permit the data to overlap both active signatures. Keep the exact same quiet subspace.
    b = z.shape[0]
    re,im = (z/p[:,None,None]).reshape(b,-1).chunk(2,dim=1)
    data = torch.complex(re,im).reshape(b,32,384)
    data -= (data*refs[1].conj()).sum(-1,keepdim=True)*refs[1]
    pilots = amp1*refs[0]+amp3*ph*refs[2]
    a = power(data)
    c = (data.conj()*pilots).real.mean((1,2))
    available = 1.-m.eta1-m.eta3
    scale = (-c+(c.square()+a*available).sqrt())/a
    return data*scale[:,None,None]+pilots


def set_share(model,name,original):
    modem=model.modem
    raw,eta3=original
    modem.eta1_raw.copy_(raw)
    modem.eta3=eta3
    if name!="native":
        total=float(modem.eta1)+eta3
        share={"share20":.2,"share40":.4}[name]
        modem.eta3=total*share
        chan=total-modem.eta3
        modem.eta1_raw.copy_(torch.logit(torch.full_like(raw,(chan-.012)/.058)))
    return float(modem.eta1),modem.eta3


def receive(wave,h,noise,snr,layout,vs):
    y = h[:,:,None]*wave + 10**(-snr/20)*noise
    return rotate(y,reversed(vs)) if layout=="dedicated" else y


def posterior(y,refs,model,prior):
    channel,quiet,amplitude = (y @ refs.T.contiguous()).unbind(-1)
    cross = (amplitude*channel.conj()).sum(1)
    noise = quiet.abs().square().mean(1).clamp_min(1e-5)
    a,b = (384*model.modem.eta1).sqrt(), math.sqrt(384*model.modem.eta3)
    grid = torch.linspace(.4,7,401,device=y.device)
    loglike = (2*a*b/(a*a+b*b)) * (cross[:,None]*torch.exp(-1j*theta(grid))).real/noise[:,None]
    mu,var=prior
    return torch.softmax(loglike-(grid-mu).square()/(2*var),dim=1) @ grid


def batch_metrics(x,z,wave,state,est,h,model,dists):
    adjusted = state[1].pwr.sqrt() if est is None else est
    ps,ds=quality(x,render(model,state,est),dists)
    en=power(wave)
    papr=10*(wave.abs().square().amax((1,2))/en).log10()
    other=torch.stack(((adjusted-source_scale(z)).abs(),nmse(state[3],h),papr,en),1).cpu().tolist()
    return [(float(p),float(d),*rest) for p,d,rest in zip(ps,ds,other)]


@torch.inference_mode()
def main():
    out=setup("frontend")
    model=codec("continuous")
    refs=basis()
    vs=frame_transform(refs)
    dists=scorer()
    original=(model.modem.eta1_raw.clone(),model.modem.eta3)
    x,ids=source("val256",out)
    z=latents(model,x)
    scales=source_scale(z)
    prior=(float(scales.mean()),float(scales.var(unbiased=False)))
    assert prior[1]>0
    bank=draws(len(ids),DEV_SEED)
    fit={layout:[] for layout in LAYOUTS}
    scores={key:[] for key in SHARES}
    shares={key:set_share(model,key,original) for key in SHARES}
    set_share(model,"native",original)
    qz=z[:8].cuda()
    h,noise=(v[:8].cuda() for v in bank[:2])
    base=packet("spread",model,qz,refs,vs)
    mapped=packet("dedicated",model,qz,refs,vs)
    y0=h[:,:,None]*base+noise
    equiv=(rotate(h[:,:,None]*mapped+rotate(noise,vs),reversed(vs))-y0).abs().max().item()
    assert equiv < 1e-4
    qual={}
    for k in LAYOUTS:
        w=packet(k,model,qz,refs,vs)
        assert (power(w)-1).abs().max()<3e-5
        qual[k]={"max_energy_error":float((power(w)-1).abs().max())}
    save(out/"qualification.json",{"paired_unitary_max_abs_IQ_error":equiv,"layouts":qual,"slots_per_block":384,"blocks":32})
    for snr in CFG["siso"]["snr_db"]:
        for sl in batches(len(z)):
            zz=z[sl].cuda()
            h,noise=(v[sl].cuda() for v in bank[:2])
            for k in LAYOUTS:
                w=packet(k,model,zz,refs,vs)
                y=receive(w,h,noise,snr,k,vs)
                state=retrieve(model,y)
                fit[k].extend(torch.stack((state[1].pwr.sqrt(),source_scale(zz)),1).cpu().tolist())
                if k=="spread" and sl.start<64:
                    p,_=quality(rgb(x,sl),render(model,state),dists)
                    scores["native"].extend(p.tolist())
            if sl.start<64:
                for key in SHARES[1:]:
                    set_share(model,key,original)
                    w=tx(model,zz)
                    y=receive(w,h,noise,snr,"spread",vs)
                    p,_=quality(rgb(x,sl),render(model,retrieve(model,y)),dists)
                    scores[key].extend(p.tolist())
                set_share(model,"native",original)
    terms={}
    for k in LAYOUTS:
        a=np.asarray(fit[k])
        terms[k]=np.linalg.lstsq(np.c_[a[:,0],np.ones(len(a))],a[:,1],rcond=None)[0].tolist()
    assert all(len(v)==128 for v in scores.values())
    scoremeans={k:float(np.mean(v)) for k,v in scores.items()}
    chosen=max(SHARES,key=scoremeans.get)
    save(out/"freeze.json",{"affine_slope_intercept":terms,"empirical_scale_mean_var":prior,"energy_split":shares,
         "energy_val_mean_psnr":scoremeans,"selected":chosen,"bank":draw_identity(bank)})
    x,ids=source("common500",out)
    z=latents(model,x)
    records=[]
    hashes={}
    for seed in CFG["seeds"]:
        bank=draws(len(z),seed)
        hashes[str(seed)]=draw_identity(bank)
        for snr in CFG["siso"]["snr_db"]:
            for sl in batches(len(z)):
                zz=z[sl].cuda()
                im=rgb(x,sl)
                h,noise=(v[sl].cuda() for v in bank[:2])
                values={}
                for k in LAYOUTS:
                    w=packet(k,model,zz,refs,vs)
                    y=receive(w,h,noise,snr,k,vs)
                    state=retrieve(model,y)
                    slope,offset=terms[k]
                    adj=(slope*state[1].pwr.sqrt()+offset).clamp(.4,7.)
                    values[k+"_raw"]=batch_metrics(im,zz,w,state,None,h,model,dists)
                    values[k+"_affine"]=batch_metrics(im,zz,w,state,adj,h,model,dists)
                    if k=="spread":
                        est=posterior(y,refs,model,prior)
                        values["posterior"]=batch_metrics(im,zz,w,state,est,h,model,dists)
                w=packet("spread",model,zz,refs,vs,constant=prior[0])
                state=retrieve(model,receive(w,h,noise,snr,"spread",vs))
                values["constant"]=batch_metrics(im,zz,w,state,torch.full_like(state[1].pwr,prior[0]),h,model,dists)
                for key in SHARES[1:]:
                    set_share(model,key,original)
                    w=tx(model,zz)
                    state=retrieve(model,receive(w,h,noise,snr,"spread",vs))
                    values[key]=batch_metrics(im,zz,w,state,None,h,model,dists)
                set_share(model,"native",original)
                for arm,v in values.items():
                    for j,(p,d,ae,ne,papr,en) in enumerate(v):
                        assert abs(en-1.)<4e-5
                        records.append(dict(snr=snr,draw=seed,source=sl.start+j,arm=arm,psnr=p,dists=d,
                                            scale_mae=ae,channel_nmse=ne,papr_db=papr,energy=en))
            print("formal",seed,snr,flush=True)
    rows(out/"common500_events.csv",records)
    save(out/"test_banks.json",hashes)
    finish(out,formal_receiver_images=len(records),formal_unique_transmissions=len(ids)*2*2*6,
           image_encoder_examples=756, validation_energy_image_scores=384,calibration_observations_per_layout=512,
           energy_selected=chosen, native_scores_shared_across_routes=True)


if __name__=="__main__":
    main()

