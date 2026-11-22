"""Cross the 5A waveform and receiver-reliability switches at two frozen published weights."""
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from study import D2, D3, STUDY, SWIN, SWIN_SHA, TEMP, dump_once, regular_path, save_once

sys.path[:0]=[str(TEMP/"source/experiments"/d/"src") for d in (D2,D3)]
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, sha256_file
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.stats import paired_bootstrap_ci

SNAP=TEMP/"producer_snapshots/observed_v1"
ORIG=SNAP/"experiments/dir5-otfs-genvar-20261122/src/dir5_og/otfs_blind.py"
OLD_RESULTS=SNAP/"results/dir5_otfs_genvar_20261122"
WEIGHTS=["symp_otfs","symp_ofdm_2d"]
FACTORS=[(w,h) for w in (False,True) for h in (False,True)]
HASH="329c902279cfde07a5b45229ff1d6723bd02707c0a60f444afc71653f31f010f"
PROTOCOL={"data":"val256, 10 dB, k_d 0.10 and 0.18, one original validation noise seed 9701",
          "weights":WEIGHTS,"waveform":"ISFFT/SFFT on/off","RX reliability":"uniform/per-resource-element",
          "images_per_cell":256,"max_reconstructions":4096,"max_physical_observations":2048,
          "paired":"PSNR and DISTS; on/off within each checkpoint and fixed reliability; reliability within each spreading state; check interaction",
          "bootstrap_source_seed":2026112306,"bootstrap_draws":5000,"input_sha256":HASH,
          "max_device_seconds":1200,"training":0,"cohort":"development aggregate, checkpoint selected using these sources"}


def register():
    assert sha256_file(ORIG)==HASH
    src=ORIG.read_text()
    old="        if self.use_otfs:\n"
    assert src.count(old)==1
    altered=src.replace(old,'        if getattr(self, "uniform_reliability", self.use_otfs):\n')
    compile(altered,str(ORIG),"exec")
    save_once(TEMP/"waveform_control_src/otfs_factorial.py",altered.encode())
    dump_once(STUDY/"dir5/waveform_protocol.json",PROTOCOL)


def write_csv(p,rows):
    regular_path(p)
    with p.open("x",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


@torch.inference_mode()
def run(gpu):
    assert json.loads((STUDY/"dir5/waveform_protocol.json").read_text())==PROTOCOL
    out=STUDY/"dir5/waveform_val256"
    regular_path(out)
    out.mkdir(exist_ok=False)
    start=time.time()
    cache=TEMP/"cache/dir5_waveform"
    regular_path(cache)
    configure_runtime_env(cache_root=cache,num_threads=4)
    configure_torch_determinism()
    hw=admit_gpu(gpu)
    altered=TEMP/"waveform_control_src/otfs_factorial.py"
    assert altered.read_text()==ORIG.read_text().replace("        if self.use_otfs:\n",
                    '        if getattr(self, "uniform_reliability", self.use_otfs):\n')
    spec=importlib.util.spec_from_file_location("otfs_factorial",altered)
    ot=importlib.util.module_from_spec(spec);spec.loader.exec_module(ot)
    assert sha256_file(SWIN)==SWIN_SHA
    bb=load_swinjscc_backbone(SWIN,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device="cuda:0")
    ds=load_split_u8("val256")
    assert not ds.is_holdout and len(set(ds.source_ids))==256
    x=ds.images_u8.float()/255.
    metric=OfflineVisualEvaluator(device=torch.device("cuda:0"),include_lpips=False)
    gen=torch.Generator(device="cpu").manual_seed(9701)
    eps=torch.randn((256,48,256,2),generator=gen,dtype=torch.float32)
    ot.sample_doubly_selective_multipath(256,.05,generator=gen)
    channels={k:ot.sample_doubly_selective_multipath(256,k,generator=gen) for k in (.1,.18)}
    old=json.loads((OLD_RESULTS/"eval/summary_dir5_common500.json").read_text())
    rows,stats,pairs=[],[],[]
    replay,weight_ids={},{}
    stems=None
    for tag in WEIGHTS:
        cp=old["checkpoints"][tag]
        regular_path(Path(cp["path"]))
        assert sha256_file(cp["path"])==cp["sha256"]
        weight_ids[tag]=cp["sha256"]
        m=ot.SympOTFSBlindTransceiver(bb,method=tag,hidden_ch=160).cuda().eval()
        sd=torch.load(cp["path"],map_location="cpu")["state_dict"]
        missing,unexpected=m.load_state_dict(sd,strict=False)
        assert not unexpected and not set(missing).intersection(n for n,p in m.named_parameters() if p.requires_grad)
        if stems is None:
            stems=torch.cat([m.extract_encoder_stem(batch.cuda()).cpu() for batch in x.split(32)])
        for kd,(g,nu) in channels.items():
            vals={(w,h):np.full((256,2),np.nan) for w,h in FACTORS}
            for i in range(0,256,32):
                j=i+32
                for w in (False,True):
                    m.modem.use_otfs=w
                    tx=m.encode_otfs_tx(stems[i:j].cuda())
                    y,_=ot.physical_doubly_selective_waveform_channel(tx,10.,kd_max=kd,
                          multipath_override=(g[i:j].cuda(),nu[i:j].cuda()),standard_noise_td_iq=eps[i:j].cuda())
                    for h in (False,True):
                        m.modem.uniform_reliability=h
                        rec,_,_=m.decode_otfs_rx(y)
                        scores=metric.evaluate_batch(x[i:j].cuda(),rec)
                        vals[w,h][i:j]=np.column_stack([scores["psnr_db"],scores["dists"]])
            cell=dict(weight=tag,kd=kd)
            for w,h in FACTORS:
                a=vals[w,h]
                assert np.isfinite(a).all()
                for src,(ps,di) in zip(ds.source_ids,a):
                    rows.append(dict(**cell,spread=w,uniform=h,source_id=src,psnr_db=ps,dists=di))
                for k,metric_name in enumerate(("psnr_db","dists")):
                    stats.append(dict(**cell,spread=w,uniform=h,metric=metric_name,mean=float(a[:,k].mean())))
            if kd==.18:
                native=tag=="symp_otfs"
                prod=json.loads((OLD_RESULTS/f"train/summary_{tag}.json").read_text())
                replay[tag]=float(vals[native,native][:,0].mean()-prod["best_val"]["kd018_ood_10dB"])
            contrasts = {"spread_at_local":vals[True,False]-vals[False,False],
                         "spread_at_uniform":vals[True,True]-vals[False,True],
                         "uniform_at_spread":vals[True,True]-vals[True,False],
                         "uniform_at_no_spread":vals[False,True]-vals[False,False],
                         "joint_on_minus_off":vals[True,True]-vals[False,False]}
            contrasts["interaction"]=contrasts["spread_at_uniform"]-contrasts["spread_at_local"]
            for label,delta in contrasts.items():
                for k,metric_name in enumerate(("psnr_db","dists")):
                    v=paired_bootstrap_ci(delta[:,k],num_resamples=5000,seed=2026112306)
                    pairs.append(dict(**cell,contrast=label,metric=metric_name,**{a:v[a] for a in ("n","mean","ci95_lo","ci95_hi")}))
            print(tag,kd,"evaluated",flush=True)
    write_csv(out/"source_scores.csv",rows);write_csv(out/"means.csv",stats);write_csv(out/"paired.csv",pairs)
    dump_once(out/"identity.json",{"original_source_sha256":HASH,"modified_source_sha256":sha256_file(altered),
              "gpu":hw,"weight_sha256":weight_ids,"wall_s":time.time()-start,"replay_psnr_difference":replay,
              "transmissions":2048,"reconstructions":4096,"unique_sources":256,"noise_repeats":1})
    if max(abs(v) for v in replay.values())>.008:
        raise RuntimeError(f"Native validation replay requires inspection: {replay}")
    print("PASS",replay,flush=True)


if __name__=="__main__":
    if sys.argv[1]=="register": register()
    else: run(int(sys.argv[1]))
