"""Source-paired test of three 5B receivers on common500 and Kodak24."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from dir5_training_pair import ARMS, GPU, ROOT, RUN, digest, paths
from study import TEMP, dump_once, regular_path

sys.path[:0]=paths()
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir5_og.genvar_ultralow_blind import (
    DEFAULT_VAR_VQVAE_CKPT, GenHoloVARTransceiver, SNR_CENTER, SNR_SCALE, load_frozen_var_vqvae,
)

CELLS=("train_on_test_on","train_on_test_off","train_off_test_off")
METRICS=("psnr_db","dists","lpips_alex")
SEEDS=(5001,5002,5003)


def write_table(path,rows):
    regular_path(path)
    with path.open("x",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def tensor_hash(x):
    return hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    out=ROOT/"evaluation"
    regular_path(out);out.mkdir(exist_ok=False)
    configure_runtime_env(cache_root=TEMP/"cache/dir5_train_control/evaluation",num_threads=4)
    configure_torch_determinism()
    hw=admit_gpu(GPU)
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    vae=load_frozen_var_vqvae(DEFAULT_VAR_VQVAE_CKPT,device="cuda:0")
    models={};weights={}
    for arm in ARMS:
        assert json.loads((RUN/(arm+"_wait.json")).read_text())["actual_wait"]==0
        meta=json.loads((ROOT/"training"/arm/("summary_"+arm+".json")).read_text())
        cp=Path(meta["best_ckpt_path"])
        assert digest(cp)==meta["best_ckpt_sha256"]
        model=GenHoloVARTransceiver(vae,cbr_denom=256,method="genholo_var",hidden_ch=160).cuda().eval()
        state=torch.load(cp,map_location="cpu")
        assert state["step"]==meta["best_step"] and set(state["state_dict"])=={
            k for k,p in model.named_parameters() if p.requires_grad}
        missing,unexpected=model.load_state_dict(state["state_dict"],strict=False)
        assert not unexpected and not set(state["state_dict"]).intersection(missing)
        model.vq_proj.disable_projection=arm=="projection_off"
        models[arm]=model
        weights[arm]={"sha256":meta["best_ckpt_sha256"],"step":state["step"]}
    scores=OfflineVisualEvaluator(device=torch.device("cuda"),include_lpips=True)
    records=[];sources=[];means=[];paired=[];hashes={}
    for split,n in (("common500",500),("kodak24",24)):
        ds=load_split_u8(split,allow_holdout=True);ids=ds.source_ids
        assert ds.is_holdout and len(ds.images_u8)==len(set(ids))==len(ids)==n
        x=ds.images_u8.float()/255.
        feats=torch.cat([vae.quant_conv(vae.encoder(v.cuda()*2.-1.)).cpu() for v in x.split(64)])
        hashes[split]={"images_ordered":tensor_hash(ds.images_u8),"source_ids":ids,"events":{}}
        values=np.full((3,3,n,3),np.nan,dtype=np.float64)
        for si,seed in enumerate(SEEDS):
            gh=torch.Generator(device="cpu").manual_seed(seed*2311+10*19)
            hr,hi=sample_correlated_rbs(n,8,.85,device="cpu",generator=gh)
            h=torch.stack([hr,hi],dim=-1)
            gn=torch.Generator(device="cpu").manual_seed(seed*3517+256+10*29)
            noise=torch.randn((n,8,96,2),generator=gn,dtype=torch.float32)
            hashes[split]["events"][str(seed)]={"h":tensor_hash(h),"noise":tensor_hash(noise)}
            for i in range(0,n,32):
                end=min(i+32,n);target=x[i:end].cuda()
                latent=feats[i:end].cuda();hb=h[i:end].cuda();eb=noise[i:end].cuda()
                alive=torch.ones((end-i,8),device="cuda")
                for arm,m in models.items():
                    tx=m.forward_from_fenc(latent,10.,alive_rb_mask=alive,h_rb_override=hb,standard_noise_iq=eb,decode_pixels=False)
                    f=tx["f_pre_vq"]
                    cr=((tx["c_r_hat_db"]-SNR_CENTER)/SNR_SCALE)[:,None,None,None].expand(-1,1,16,16)
                    af=tx["alive_frac"][:,None,None,None].expand_as(cr)
                    disabled=f+m.vq_proj.post_refine(torch.cat([f,f,cr,af],dim=1))
                    if arm=="projection_on":
                        variants=((0,tx["f_out"]),(1,disabled))
                    else:
                        assert torch.allclose(tx["f_out"],disabled,rtol=1e-6,atol=1e-6)
                        variants=((2,tx["f_out"]),)
                    for ci,estimate in variants:
                        mb=scores.evaluate_batch(target,m.decode_fhat_to_01(estimate))
                        values[ci,si,i:end,:]=np.column_stack([mb[k] for k in METRICS])
        assert np.isfinite(values).all()
        draws=np.random.default_rng(2026112306).integers(0,n,size=(5000,n))
        def estimate(vector):
            lo,hi=np.quantile(vector[draws].mean(axis=1),[.025,.975])
            return {"n_sources":n,"mean":float(vector.mean()),"ci95_lo":float(lo),"ci95_hi":float(hi)}
        averages=values.mean(axis=1)
        for ci,cell in enumerate(CELLS):
            for i,source in enumerate(ids):
                sources.append({"split":split,"source_id":source,"cell":cell,**dict(zip(METRICS,averages[ci,i]))})
                for si,seed in enumerate(SEEDS):
                    records.append({"split":split,"source_id":source,"cell":cell,"seed":seed,**dict(zip(METRICS,values[ci,si,i]))})
            for mi,key in enumerate(METRICS):
                means.append({"split":split,"cell":cell,"metric":key,**estimate(averages[ci,:,mi])})
        for ca,cb in ((0,2),(1,2),(0,1)):
            for mi,key in enumerate(METRICS):
                paired.append({"split":split,"left":CELLS[ca],"right":CELLS[cb],"metric":key,
                               **estimate(averages[ca,:,mi]-averages[cb,:,mi])})
        print(split,"complete",n,"sources",flush=True)
    assert len(records)==4716 and len(sources)==1572 and len(means)==len(paired)==18
    for name,table in (("repeat_scores",records),("source_means",sources),("means",means),("paired",paired)):
        write_table(out/(name+".csv"),table)
    dump_once(out/"identity.json",{"gpu":hw,"weights":weights,"frozen_vae_sha256":digest(DEFAULT_VAR_VQVAE_CKPT),
        "source_and_channel":hashes,"distinct_source_channel_events":1572,"model_transmissions":3144,
        "renders":len(records),"bootstrap_seed":2026112306,"bootstrap_size":5000})
    print("Completed all three registered comparisons",flush=True)


if __name__=="__main__":
    main()
