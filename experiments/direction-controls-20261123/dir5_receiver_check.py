"""One frozen-receiver intervention on 5B v2, with native reference and shared polishing."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from study import D2, D3, REPO, STUDY, TEMP, dump_once, regular_path

PROJECT = TEMP/"producer_snapshots/observed_v1/experiments/dir5-otfs-genvar-20261122"
sys.path[:0] = [str(PROJECT/"src"), *(str(TEMP/"source/experiments"/d/"src") for d in (D2,D3))]
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, sha256_file
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir2_zpac.stats import paired_bootstrap_ci
from dir5_og.genvar_ultralow_blind import DEFAULT_VAR_VQVAE_CKPT, GenHoloVARTransceiver, load_frozen_var_vqvae

KEYS = ["psnr_db","dists","lpips_alex"]
ARMS = ["native","gates_off_same_polisher","continuous_only"]
PROTOCOL = {"purpose":"Separate the two codebook correction paths from the post-refiner at fixed v2 weights.",
    "data":"existing development val256, CBR 1/256, 10 dB, all 8 RBs, original fixed validation noise",
    "arms":ARMS,"metrics":KEYS,"repeats":1,"bootstrap_source_seed":2026112305,"bootstrap_draws":5000,
    "max_gpu_seconds":900,"checkpoints":1,"unique_source_transmissions":256,"maximum_decoder_outputs":768,
    "no_training":True,"analysis":"paired direct differences, native minus no codebook; no codebook minus continuous; native minus continuous. Conditional intervention on the selected receiver."}


def write_csv(p, rows):
    regular_path(p)
    with p.open("x",newline="") as f:
        w=csv.DictWriter(f,list(rows[0])); w.writeheader(); w.writerows(rows)


@torch.inference_mode()
def run(gpu, expected_weight):
    assert json.loads((STUDY/"dir5/receiver_protocol.json").read_text())==PROTOCOL
    out=STUDY/"dir5/receiver_val256"
    regular_path(out)
    out.mkdir(exist_ok=False)
    t0=time.time()
    cache=TEMP/"cache/dir5_control"
    regular_path(cache)
    configure_runtime_env(cache_root=cache,num_threads=4)
    configure_torch_determinism()
    hw=admit_gpu(gpu)
    tag="genholo_var_cbr256"
    train_path=REPO/"results/dir5_otfs_genvar_20261122/train"/f"summary_{tag}.json"
    train=json.loads(train_path.read_text())
    cp=Path(train["best_ckpt_path"])
    regular_path(cp)
    assert expected_weight==train["best_ckpt_sha256"]==sha256_file(cp)
    eval_v1=json.loads((TEMP/"producer_snapshots/observed_v1/results/dir5_otfs_genvar_20261122/eval/summary_dir5_common500.json").read_text())
    assert expected_weight!=eval_v1["checkpoints"][tag]["sha256"]
    regular_path(Path(DEFAULT_VAR_VQVAE_CKPT))
    assert sha256_file(DEFAULT_VAR_VQVAE_CKPT)==eval_v1["checkpoints"]["var_vqvae"]["sha256"]
    vae=load_frozen_var_vqvae(DEFAULT_VAR_VQVAE_CKPT,device="cuda:0")
    model=GenHoloVARTransceiver(vae,cbr_denom=256,method="genholo_var",hidden_ch=160).cuda().eval()
    assert model.tx_fine is None and hasattr(model,"tx_pyr_down")
    sd=torch.load(cp,map_location="cpu")["state_dict"]
    trainable={k for k,p in model.named_parameters() if p.requires_grad}
    missing,unexpected=model.load_state_dict(sd,strict=False)
    assert not unexpected and not trainable.intersection(missing)
    evaluator=OfflineVisualEvaluator(device=torch.device("cuda:0"),include_lpips=True)
    ds=load_split_u8("val256")
    assert not ds.is_holdout and len(set(ds.source_ids))==len(ds.images_u8)==256
    target=ds.images_u8.float()/255.
    # Use the frozen encoder in the actual 64-image cache-generation batches.
    latent=torch.cat([vae.quant_conv(vae.encoder(x.cuda()*2.-1.)).cpu() for x in target.split(64)])
    gen=torch.Generator(device="cpu").manual_seed(9811+256)
    hr,hi=sample_correlated_rbs(256,8,.85,device="cpu",generator=gen)
    h=torch.stack([hr,hi],dim=-1)
    eps=torch.randn((256,8,96,2),generator=gen,dtype=torch.float32)
    rows=[]
    for s in range(0,256,32):
        e=s+32
        y=model.forward_from_fenc(latent[s:e].cuda(),10.,alive_rb_mask=torch.ones((32,8),device="cuda"),
                h_rb_override=h[s:e].cuda(),standard_noise_iq=eps[s:e].cuda(),decode_pixels=False)
        f=y["f_pre_vq"]
        # Same post-refinement network and blind RX conditioning; set the contributions of both
        # frozen codebook branches to zero at the fusion point. The native channel is evaluated once.
        from dir5_og.genvar_ultralow_blind import SNR_CENTER,SNR_SCALE
        cr=((y["c_r_hat_db"]-SNR_CENTER)/SNR_SCALE)[:,None,None,None].expand(-1,1,16,16)
        af=y["alive_frac"][:,None,None,None].expand_as(cr)
        without=f+model.vq_proj.post_refine(torch.cat([f,f,cr,af],dim=1))
        for arm,rec in zip(ARMS,(y["f_out"],without,f)):
            vals=evaluator.evaluate_batch(target[s:e].cuda(),model.decode_fhat_to_01(rec))
            for i in range(e-s):
                rows.append({"source_index":s+i,"source_id":ds.source_ids[s+i],"arm":arm,
                             **{k:float(vals[k][i]) for k in KEYS}})
    write_csv(out/"source_scores.csv",rows)
    means,paired=[],[]
    for k in KEYS:
        vals={a:np.array([x[k] for x in rows if x["arm"]==a]) for a in ARMS}
        for a in ARMS:
            assert len(vals[a])==256 and np.isfinite(vals[a]).all()
            v=paired_bootstrap_ci(vals[a],num_resamples=5000,seed=2026112305)
            means.append({"arm":a,"metric":k,**{x:v[x] for x in ("n","mean","ci95_lo","ci95_hi")}})
        for a,b in ((ARMS[0],ARMS[1]),(ARMS[1],ARMS[2]),(ARMS[0],ARMS[2])):
            v=paired_bootstrap_ci(vals[a],vals[b],num_resamples=5000,seed=2026112305)
            paired.append({"left":a,"right":b,"metric":k,**{x:v[x] for x in ("n","mean","ci95_lo","ci95_hi")}})
    write_csv(out/"means.csv",means);write_csv(out/"paired.csv",paired)
    names={"psnr_db":"psnr","dists":"dists","lpips_alex":"lpips"}
    replay={z["metric"]:z["mean"]-train["best_val"]["ul_10dB_"+names[z["metric"]]] for z in means if z["arm"]=="native"}
    assert sha256_file(cp)==expected_weight
    dump_once(out/"identity.json",{"checkpoint_sha256":expected_weight,"best_step":train["best_step"],
        "frozen_vae_sha256":eval_v1["checkpoints"]["var_vqvae"]["sha256"],"gpu":hw,"transmissions":256,"renders":768,
        "forward_from_fenc_batch_calls":8,"frozen_encoder_batch_calls":4,"decoder_batch_calls":24,
        "native_minus_saved_validation":replay,"wall_seconds":time.time()-t0,
        "pixel_source_order_sha256":hashlib.sha256(ds.images_u8.contiguous().numpy().tobytes()).hexdigest(),
        "channel_sha256":hashlib.sha256(h.numpy().tobytes()).hexdigest(),"noise_sha256":hashlib.sha256(eps.numpy().tobytes()).hexdigest()})
    if abs(replay["psnr_db"])>.01 or max(abs(replay[k]) for k in KEYS[1:])>.0003:
        raise RuntimeError(f"Original validation discrepancy, inspect: {replay}")
    print("PASS: 256 identical received observations, 768 outputs",replay,flush=True)


if __name__=="__main__":
    if sys.argv[1]=="register":
        dump_once(STUDY/"dir5/receiver_protocol.json",PROTOCOL)
    else:
        run(int(sys.argv[1]),sys.argv[2])
