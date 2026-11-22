"""Register and run a single matched training comparison of the two 5B codebook corrections."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from study import D2, D3, PYTHON, STUDY, TEMP, WEIGHTS, dump_once, edit, regular_path, save_once

ROOT = STUDY / "dir5/train_control_256"
RUN = TEMP / "runs/dir5_train_control"
PROJECT = "dir5-otfs-genvar-20261122"
SNAP = TEMP / "producer_snapshots/published_v2"
SRC = TEMP / "dir5_matched_training/source" / PROJECT
SHARED = TEMP.parent / "dir5_otfs_genvar_v1/cache"
GPU = 3
SEED = 2026112305
ARMS = ("projection_on", "projection_off")
LIMIT = 6000


def digest(path):
    regular_path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paths():
    return [str(SRC/"src"), *(str(TEMP/"source/experiments"/pj/"src") for pj in (D2,D3))]


def prepare():
    capture = json.loads((STUDY/"dir5/published_v2_identity.json").read_text())
    assert capture["checkpoints_matching_all"] == 14
    source = SNAP/"experiments"/PROJECT
    sources = {}
    trainer = "train_dir5_otfs_genvar.py"
    for p in sorted(source.rglob("*.py")):
        if "src" not in p.relative_to(source).parts and p.name != trainer:
            continue
        src = p.read_bytes()
        assert hashlib.sha256(src).hexdigest() == capture["files"][str(p.relative_to(SNAP))]
        text = src.decode()
        if p.name == "genvar_ultralow_blind.py":
            anchor = "        w_vq, w_cb = gates.chunk(2, dim=-1)"
            text = edit(text,anchor,"        if getattr(self, 'disable_projection', False):\n            gates = gates * 0.0\n"+anchor)
        if p.name == trainer:
            text = edit(text,"    run_cache = args.shared_cache_dir / args.run_tag",
                        "    run_cache = Path("+repr(str(TEMP/"cache/dir5_train_control"))+") / args.run_tag")
            anchor = "        model = GenHoloVARTransceiver(vae, cbr_denom=args.cbr_denom, method=args.method).to(device)"
            text = edit(text,anchor,anchor+"\n"+'''        model.vq_proj.disable_projection = args.run_tag == "projection_off"
        import hashlib
        init = hashlib.sha256()
        for name, par in model.named_parameters():
            if par.requires_grad:
                init.update(name.encode() + b"\\0")
                init.update(par.detach().cpu().contiguous().numpy().tobytes())
        save_json(args.out_dir / "initialization.json", {"trainable_state_sha256":init.hexdigest(), "disable_projection":model.vq_proj.disable_projection})''')
        encoded = text.encode()
        compile(encoded, str(p), "exec")
        relative = p.relative_to(source)
        save_once(SRC/relative,encoded)
        sources[str(relative)] = {"snapshot":hashlib.sha256(src).hexdigest(),"run":hashlib.sha256(encoded).hexdigest()}
    dump_once(ROOT/"protocol.json",{
        "registered_before_new_training": True, "trigger": "The published-v2 1/256 receiver intervention on val256 favored disabling projections; both training roles are now compared prospectively on known test sets.",
        "arms": list(ARMS), "seed":SEED, "physical_gpu":GPU, "time_limit_seconds_all_stages":LIMIT,
        "train": "Same 8x8 base, frozen VAE, latent targets, initialization, AdamW schedules, corruption, polisher and original val256 composite. 6000 latent steps + 1500 image/LPIPS steps; only both projection contributions are forced to zero in projection_off. Nominal parameter set equal; projection-specific gradients vanish in that arm.",
        "test": "CBR 1/256, 768 complex symbols, all 8 blocks, 10 dB, common500 and Kodak24; source means over original draws 5001,5002,5003. Three cells: trained-on tested on/off, and trained-off tested off. PSNR, DISTS (primary), LPIPS-Alex. 5000 pointwise bootstrap draws per split; no tuning on the new results.",
        "comparisons": ["trained_on/test_on minus trained_off/test_off","trained_on/test_off minus trained_off/test_off","trained_on/test_on minus trained_on/test_off"],
        "split_inference":"One new matched training seed. Test sources already used by the producer. Image intervals are conditional on these models.",
        "cache_sha256":digest(SHARED/"var_latents_fp16.pt"),"source_sha256":sources,
        "workers":{name:digest(STUDY/name) for name in ("dir5_training_pair.py","dir5_training_evaluation.py")}})


def runtime():
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU),CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONPATH=os.pathsep.join(paths()),PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")


def check_cache():
    import torch
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, seed_everything
    from dir5_og.genvar_ultralow_blind import load_frozen_var_vqvae, DEFAULT_VAR_VQVAE_CKPT
    configure_runtime_env(cache_root=TEMP/"cache/dir5_train_control/qualification",num_threads=4)
    seed_everything(SEED);configure_torch_determinism()
    gpu=admit_gpu(GPU)
    ident=json.loads((STUDY/"dir5/receiver_val256/identity.json").read_text())
    assert digest(DEFAULT_VAR_VQVAE_CKPT)==ident["frozen_vae_sha256"]
    cache=torch.load(SHARED/"var_latents_fp16.pt",map_location="cpu",weights_only=True)
    vae=load_frozen_var_vqvae(DEFAULT_VAR_VQVAE_CKPT,device=torch.device("cuda:0")).eval()
    found=[]
    with torch.inference_mode():
        for split,indices in (("train20k",(0,9600,19968)),("val256",(0,192))):
            ds=load_split_u8(split)
            cached=cache["train_f_enc_fp16" if split=="train20k" else "val_f_enc_fp32"]
            assert len(ds.source_ids)==len(ds.images_u8)==len(cached)
            for start in indices:
                img=ds.images_u8[start:start+64].cuda().float()/255.
                feats=vae.quant_conv(vae.encoder(img*2.-1.)).cpu()
                expected=cached[start:start+len(img)].float()
                diff=(feats-expected).abs()
                adjacent=(feats-expected.roll(1,0)).abs().mean().item()
                assert diff.mean().item()<0.001 and diff.max().item()<0.02 and adjacent>10*diff.mean().item()
                found.append({"split":split,"first_index":start,"n":len(img),"first_source":ds.source_ids[start],
                              "mean_abs":diff.mean().item(),"max_abs":diff.max().item(),"wrong_order_mean_abs":adjacent})
    dump_once(ROOT/"qualification.json",{"gpu":gpu,"cache_matches":found,"vae_sha256":ident["frozen_vae_sha256"]})


def worker(label,cmd,deadline):
    started=time.time()
    remaining=deadline-started
    if remaining<=0:raise TimeoutError("Registered single-GPU window expired")
    with (RUN/(label+".log")).open("x") as f:
        p=subprocess.Popen(cmd,cwd=STUDY,env=runtime(),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        dump_once(RUN/(label+"_start.json"),{"pid":p.pid,"ticks":Path("/proc",str(p.pid),"stat").read_text().split(") ")[1].split()[19],"time":started,"physical_gpu":GPU})
        try:rc=p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.terminate()
            rc=p.wait(timeout=45)
        dump_once(RUN/(label+"_wait.json"),{"pid":p.pid,"actual_wait":rc,"device_seconds":time.time()-started})
    if rc:raise RuntimeError(f"{label}: exit {rc}")


def run():
    regular_path(RUN);RUN.mkdir(parents=True,exist_ok=False)
    start=time.time()
    dump_once(RUN/"claim.json",{"owner":os.getpid(),"time":start,"protocol":digest(ROOT/"protocol.json")})
    deadline=start+LIMIT
    try:
        worker("qualification",[str(PYTHON),"-B",str(STUDY/"dir5_training_pair.py"),"qualify"],deadline)
        for arm in ARMS:
            worker(arm,[str(PYTHON),"-B",str(SRC/"scripts/train_dir5_otfs_genvar.py"),"--track","genvar","--method","genholo_var","--cbr-denom","256","--run-tag",arm,"--physical-gpu",str(GPU),"--stage1-steps","6000","--stage2-steps","1500","--seed",str(SEED),"--shared-cache-dir",str(SHARED),"--ckpt-dir",str(WEIGHTS/"dir5_train_control"/arm),"--out-dir",str(ROOT/"training"/arm)],deadline)
        initial=[json.loads((ROOT/"training"/arm/"initialization.json").read_text()) for arm in ARMS]
        assert initial[0]["trainable_state_sha256"]==initial[1]["trainable_state_sha256"]
        assert [a["disable_projection"] for a in initial]==[False,True]
        worker("evaluation",[str(PYTHON),"-B",str(STUDY/"dir5_training_evaluation.py")],deadline)
    except Exception as e:
        dump_once(RUN/"failure.json",{"reason":repr(e),"at":time.time()})
        raise
    ledger={label:json.loads((RUN/(label+"_wait.json")).read_text()) for label in ("qualification",*ARMS,"evaluation")}
    dump_once(ROOT/"resource.json",{"jobs":ledger,"device_hours":sum(x["device_seconds"] for x in ledger.values())/3600})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","qualify","run"));arg=parser.parse_args()
    {"prepare":prepare,"qualify":check_cache,"run":run}[arg.action]()
