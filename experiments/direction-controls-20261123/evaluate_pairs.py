"""Evaluate the registered mechanism controls on the original channels, paired by source."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from study import D2, D3, D4, SHA, SPEC, STUDY, SWIN, SWIN_SHA, TEMP, dump_once, git, regular_path

for project in (D2, D3, D4):
    sys.path.insert(0, str(TEMP / "source/experiments" / project / "src"))

import numpy as np
import torch
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, seed_everything, sha256_file
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir3_mh.holorate_blind import HoloRateBlindTransceiver
from dir4_rn.noma_semsic_blind import GrantFreeNOMATransceiver, sample_multiuser_uplink_channels

METRICS = ("psnr_db", "dists")
REMOTE_RESULTS = {
    "noma": ("dir4_rf1bit_noma_20261122", "dir4", "dir4b_noma", "semsic_stage1_only", "SemSIC_Stage1_Only"),
    "holo": ("dir3_mimo_holorate_20261121", "dir3", "dir3b_holorate", "raster_blind_resilient", "Raster_Blind_Resilient"),
}
SNRS = {"over3x2_13_8_4": [13, 8, 4], "balanced_10_10": [10, 10], "nearfar_12_3": [12, 3]}
OLD_CONDITIONS = {
    "over3x2_13_8_4": "over3x2_nearfar_13_8_4dB", "balanced_10_10": "noma2x2_bal_10dB",
    "nearfar_12_3": "noma2x2_nearfar_12_3dB", "first8": "rateless_8rb_25pct",
    "first16": "rateless_16rb_50pct", "full": "rateless_32rb_100pct", "contiguous16": "burst_erasure_50pct",
}


def write_csv(path, rows):
    if not rows:
        raise ValueError(path)
    regular_path(path)
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_sha(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def old_eval_file(track, name):
    return git("show", f"{SHA}:results/{REMOTE_RESULTS[track][0]}/eval/{name}").decode()


def make_models(track, device):
    spec = SPEC[track]
    _, num, _, old_tag, _ = REMOTE_RESULTS[track]
    regular_path(SWIN)
    assert sha256_file(SWIN) == SWIN_SHA
    backbone = load_swinjscc_backbone(SWIN, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    record = {}
    for split in ("common500", "kodak24"):
        old = json.loads(old_eval_file(track, f"eval_{num}_summary_{split}.json"))["loaded_checkpoints"][old_tag]
        if "reference" in record and record["reference"] != old:
            raise ValueError("Archived checkpoint identity differs between splits")
        record["reference"] = old
    models = {}
    for tag in [*spec["arms"], "reference"]:
        if tag != "reference":
            summary = STUDY / "training" / track / tag / f"train_{num}_{tag}.json"
            obj = json.loads(summary.read_text())
            assert obj["backbone_ckpt"]["sha256"] == SWIN_SHA and not obj["holdout_touched"]
            record[tag] = obj["best_checkpoint"]
        ckpt = Path(record[tag]["path"])
        regular_path(ckpt)
        assert sha256_file(ckpt) == record[tag]["sha256"]
        method = old_tag if tag == "reference" else spec["method"]
        if track == "noma":
            m = GrantFreeNOMATransceiver(backbone, method=method, n_rx=2, num_rbs=32, hidden_ch=160)
            m.modem.own_denoise_only = tag == "own_denoise"
        else:
            m = HoloRateBlindTransceiver(backbone, method=method, num_rbs=32, hidden_ch=160)
            m.modem.identity_spatial = tag == "identity"
        state = torch.load(ckpt, map_location="cpu")["state_dict"]
        trainable = {k for k, v in m.named_parameters() if v.requires_grad}
        missing, unexpected = m.load_state_dict(state, strict=False)
        if unexpected or trainable.intersection(missing) or not trainable:
            raise ValueError(f"Checkpoint incomplete for {tag}")
        models[tag] = m.to(device).eval()
        record[tag] = {**record[tag], "trainable_params": sum(p.numel() for p in m.parameters() if p.requires_grad)}
    assert record[spec["arms"][0]]["trainable_params"] == record["full"]["trainable_params"]
    return models, record


def score_condition(track, condition, split, x, stems, ids, models, evaluator):
    n = len(ids)
    q = SPEC[track]
    snrs = SNRS.get(condition, [10])
    u = len(snrs)
    arr = {k: np.full((3, u, n, 2), np.nan) for k in models}
    noise_record = []
    roles = [f"User{i+1}_{snrs[i]}dB" for i in range(u)] if track == "noma" else ["image"]
    shifts = [0, n//3, 2*n//3][:u]
    for ri, seed in enumerate(q["seeds"]):
        gen = torch.Generator(device="cpu")
        if track == "noma":
            gen.manual_seed(seed*2089 + u*97 + round(snrs[0]*10))
            hr, hi = sample_multiuser_uplink_channels(n, u, 2, 32, .85, generator=gen)
            eps = torch.randn((n, 2, 32, 384, 2), generator=gen, dtype=torch.float32)
            noise = [hr, hi, eps]
        else:
            gen.manual_seed(seed*2029)
            eps = torch.randn((n, 32, 384, 2), generator=gen, dtype=torch.float32)
            extra = torch.randn((n, 32, 384, 2), generator=gen, dtype=torch.float32)
            hr, hi = sample_correlated_rbs(n, 32, .85, device="cpu", generator=gen)
            h = torch.stack([hr, hi], dim=-1)
            mask = torch.ones((n,32))
            if condition == "contiguous16":
                mask[:,8:24] = 0
            else:
                keep = {"first8":8, "first16":16, "full":32}[condition]
                mask[:,keep:] = 0
            noise = [h, eps, extra, mask]
        noise_record.append({"seed": seed, "tensor_hashes": [tensor_sha(z) for z in noise]})
        bs = 25 if track=="noma" else 50
        for a in range(0, n, bs):
            b = min(n,a+bs)
            order = [(np.arange(a,b)+sh)%n for sh in shifts]
            xb = [x[idx].cuda() for idx in order]
            sb = [stems[idx].cuda() for idx in order]
            ch = [z[a:b].cuda() for z in noise]
            for name, model in models.items():
                if track=="noma":
                    strength = torch.tensor(snrs,device="cuda",dtype=torch.float32).expand(b-a,u)
                    y = model.forward_noma_uplink(sb,strength,h_mu_override=(ch[0],ch[1]),
                            standard_noise_iq=ch[2],input_is_stem=True)["recons"]
                else:
                    rec = model.forward_hostile_blind(sb[0],10.,rho_f=.85,alive_rb_mask=ch[3],
                            jam_rb_mask=None,jam_inr_db=0.,h_rb_override=ch[0],
                            standard_noise_iq=ch[1],jam_noise_iq=ch[2],input_is_stem=True)
                    y = [rec["recon"]]
                for i in range(u):
                    m = evaluator.evaluate_batch(xb[i],y[i])
                    arr[name][ri,i,a:b,:] = np.column_stack([m[key] for key in METRICS])
    assert all(np.isfinite(v).all() for v in arr.values())
    if len(set(ids)) != n:
        raise ValueError("Source IDs are not unique")
    frames, sources, summary, paired = [], [], [], []
    statseed = 20261124 if track=="noma" else 20261123
    draws = np.random.default_rng(statseed).integers(0,n,size=(5000,n))

    def summarize(arr):
        boots = arr[draws].mean(axis=1)
        lo,hi=np.quantile(boots,[.025,.975])
        return {"mean":float(arr.mean()),"ci95_lo":float(lo),"ci95_hi":float(hi)}

    for name, raw in arr.items():
        for i,role in enumerate(roles):
            for ri,seed in enumerate(q["seeds"]):
                for j,(ps,di) in enumerate(raw[ri,i]):
                    frames.append(dict(split=split,condition=condition,arm=name,seed=seed,tuple_anchor=ids[j],
                                      role=role,source_id=ids[(j+shifts[i])%n],psnr_db=ps,dists=di))
    source_vals = {k:v.mean(axis=0) for k,v in arr.items()}
    if track=="noma":
        source_vals = {k:np.concatenate([v,v.mean(axis=0,keepdims=True)]) for k,v in source_vals.items()}
        roles = [*roles,"ALL_USERS_MEAN"]
    for i,role in enumerate(roles):
        group=dict(split=split,condition=condition,role=role)
        for k,v in source_vals.items():
            for j,(ps,di) in enumerate(v[i]):
                sources.append(dict(**group,arm=k,tuple_anchor=ids[j],
                                    source_ids="|".join(ids[(j+sh)%n] for sh in shifts) if role=="ALL_USERS_MEAN" else ids[(j+shifts[i])%n],
                                    psnr_db=ps,dists=di))
            for col,metric in enumerate(METRICS):
                summary.append(dict(**group,arm=k,metric=metric,n=n,**summarize(v[i,:,col])))
        for left,right in [("full",q["arms"][0]),("full","reference"),(q["arms"][0],"reference")]:
            for col,metric in enumerate(METRICS):
                paired.append(dict(**group,left=left,right=right,metric=metric,n=n,
                                   **summarize(source_vals[left][i,:,col]-source_vals[right][i,:,col])))
    return frames, sources, summary, paired, noise_record


@torch.inference_mode()
def evaluate(track):
    q = SPEC[track]
    for arm in q["arms"]:
        assert json.loads((TEMP/"runs"/track/f"{arm}_wait.json").read_text())["actual_wait"] == 0
    cache = TEMP/"cache"/track/"evaluation"
    regular_path(cache)
    configure_runtime_env(cache_root=cache,num_threads=4)
    seed_everything(20261124 if track=="noma" else 20261123)
    configure_torch_determinism()
    gpu = admit_gpu(q["gpu"])
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    models, checkpoints = make_models(track,torch.device("cuda:0"))
    metric = OfflineVisualEvaluator(device=torch.device("cuda:0"),include_lpips=False)
    out = STUDY/"evaluation"/track
    regular_path(out)
    out.mkdir(parents=True,exist_ok=False)
    count = {}
    audit = []
    all_summary,all_paired = [],[]
    _, _, prefix, _, oldname = REMOTE_RESULTS[track]
    for split,n in (("common500",500),("kodak24",24)):
        ds = load_split_u8(split,allow_holdout=True)
        ids = ds.source_ids
        assert len(ds.images_u8)==len(ids)==n and ds.is_holdout
        dump_once(out/f"{split}_source_manifest.json", [{"source_id":k,"chw_u8_sha256":tensor_sha(x)} for k,x in zip(ids,ds.images_u8)])
        targets = ds.images_u8.float()/255.
        model = models["full"]
        stems = torch.cat([model.extract_encoder_stem(x.cuda()).cpu() for x in targets.split(32)])
        hist = list(csv.DictReader(old_eval_file(track,f"{prefix}_results_{split}.csv").splitlines()))
        frames,sources=[],[]
        count[split] = {}
        for c in q["conditions"]:
            fr,ss,sm,pa,seeds = score_condition(track,c,split,targets,stems,ids,models,metric)
            print(split,c,len(fr),"finite actual per-user/image scores",flush=True)
            frames.extend(fr); sources.extend(ss); all_summary.extend(sm); all_paired.extend(pa)
            count[split][c] = {"scored_images":len(fr),"noise":seeds}
            for row in [v for v in sm if v["arm"]=="reference"]:
                match = [v for v in hist if v["model"]==oldname and v["condition"]==OLD_CONDITIONS[c]
                         and v.get("user_role","image")==row["role"]]
                assert len(match)==1
                expected=float(match[0][row["metric"]+"_mean"])
                audit.append({**row,"historical":expected,"difference":row["mean"]-expected})
        write_csv(out/f"{split}_frame_scores.csv",frames)
        write_csv(out/f"{split}_source_means.csv",sources)
    write_csv(out/"means.csv",all_summary)
    write_csv(out/"paired.csv",all_paired)
    write_csv(out/"historical_replay.csv",audit)
    max_diff={m:max(abs(z["difference"]) for z in audit if z["metric"]==m) for m in METRICS}
    dump_once(out/"identity.json",{"gpu":gpu,"checkpoints":checkpoints,"events":count,"historical_max_abs":max_diff,
             "bootstrap_unit": "source for HoloRate; fixed source tuple anchor for NOMA",
             "source_commit":SHA,"controller_pid":os.getpid()})
    if max_diff["psnr_db"]>.005 or max_diff["dists"]>.0002:
        raise RuntimeError(f"Historical replay disagreement requires review: {max_diff}")
    print("PASS",track,"historical replay",max_diff,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("track",choices=SPEC)
    evaluate(parser.parse_args().track)
