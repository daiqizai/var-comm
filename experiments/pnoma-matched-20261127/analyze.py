"""Read the complete fixed populations, verify pairing, and calculate group-bootstrap intervals."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SPEC=json.loads((ROOT/"protocol.json").read_text())
CONDITIONS=SPEC["evaluation"]["conditions"]
DRAWS=SPEC["evaluation"]["channel_seeds"]
ARMS=("pnoma_q2","semsic_three","semsic_latent","semsic_full")
METRICS=("weak_psnr","weak_dists","mean_psnr")

def read(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))

def write(p,rows):
    with p.open("x",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def percentiles(values):
    return np.quantile(values,[.025,.975]).tolist()

def summary(values,bank):
    lower,upper=percentiles(values[bank].mean(1))
    return dict(estimate=float(values.mean()),ci_low=lower,ci_high=upper)

def main():
    root=ROOT/"evaluate"
    out=ROOT/"analysis"
    out.mkdir(exist_ok=False)
    done=json.loads((root/"done.json").read_text())
    freeze=json.loads((ROOT/"train/freeze.json").read_text())
    assert done["registered_weight"]==freeze["weight_sha256"]
    sources=json.loads((root/"sources.json").read_text())
    assert len(sources["ids"])==len(set(sources["pixels"]))==500
    rows=read(root/"group_scores.csv")
    assert len(rows)==3000
    keys={(r["condition"],int(r["draw"]),int(r["group"]),r["method"]) for r in rows}
    assert keys=={(c,s,g,m) for c in CONDITIONS for s in DRAWS for g in range(125) for m in ARMS}
    src=defaultdict(set)
    a=np.empty((len(CONDITIONS),len(ARMS),3,125,len(METRICS)),dtype=float)
    for r in rows:
        c,s,g,m=r["condition"],int(r["draw"]),int(r["group"]),r["method"]
        slot=int(r["weak_slot"])
        assert sources["ids"][g*4+slot]==r["weak_source"]
        src[c,s,g].add(slot)
        a[list(CONDITIONS).index(c),ARMS.index(m),DRAWS.index(s),g]=[float(r[n]) for n in METRICS]
    assert np.isfinite(a).all() and all(len(v)==1 for v in src.values())
    extra=read(root/"external_user_psnr.csv")
    assert len(extra)==750
    for r in extra:
        c,s,g=list(CONDITIONS).index(r["condition"]),DRAWS.index(int(r["draw"])),int(r["group"])
        scores=np.asarray([float(r[f"slot_{u}_psnr"]) for u in range(4)])
        assert abs(scores.mean()-a[c,0,s,g,2])<1e-6
        assert abs(scores[int(r["weak_slot"])]-a[c,0,s,g,0])<1e-6
    rng=np.random.default_rng(271127)
    paired_bank=rng.integers(0,125,(10000,125))
    time_bank=rng.integers(0,32,(10000,32))
    means=[]
    deltas=[]
    source_means=a.mean(2)
    for c,cond in enumerate(CONDITIONS):
        for m,method in enumerate(ARMS):
            for k,metric in enumerate(METRICS):
                vec=source_means[c,m,:,k]
                means.append(dict(condition=cond,method=method,metric=metric,quartets=125,**summary(vec,paired_bank)))
                if m:
                    diff=vec-source_means[c,0,:,k]
                    deltas.append(dict(condition=cond,contrast=method+" minus pnoma_q2",metric=metric,
                        quartets=125,**summary(diff,paired_bank)))
    write(out/"quality_means.csv",means)
    write(out/"quality_paired.csv",deltas)
    times=read(root/"timing.csv")
    t=np.full((2,2,32,2),np.nan)
    for r in times:
        idx=(list(CONDITIONS).index(r["condition"]),ARMS[:2].index(r["method"]),int(r["group"]),int(r["repeat"]))
        assert np.isnan(t[idx]);t[idx]=float(r["rx_ms"])
    assert len(times)==256 and np.isfinite(t).all() and (t>0).all()
    avg=t.mean(-1)
    time_rows=[]
    time_deltas=[]
    for c,cond in enumerate(CONDITIONS):
        for m,name in enumerate(ARMS[:2]):
            time_rows.append(dict(condition=cond,method=name,quartets=32,**summary(avg[c,m],time_bank)))
        delta=summary(avg[c,1]-avg[c,0],time_bank)
        ratio=avg[c,1][time_bank].mean(1)/avg[c,0][time_bank].mean(1)
        lo,hi=percentiles(ratio)
        time_deltas.append(dict(condition=cond,contrast="semsic_three minus pnoma_q2",quartets=32,
                         **delta,ratio=float(avg[c,1].mean()/avg[c,0].mean()),ratio_low=lo,ratio_high=hi))
    write(out/"timing_means.csv",time_rows)
    write(out/"timing_paired.csv",time_deltas)
    val=read(ROOT/"train/validation.csv")
    assert int(max(val,key=lambda rr:float(rr["selection"]))["step"])==freeze["step"]
    data=dict(quality_rows=len(rows),per_user_rows=len(extra),timing_rows=len(times),weak_slot_match_all=True,
             distinct_groups=125,source_id_count=len(sources["ids"]),pixel_duplicates=0,
             all_source_images_returned=True,train_test_overlap=sources["train_intersection"],val_test_overlap=sources["val_intersection"],
             selected_with_validation_only=freeze["step"],bootstrap_draws=10000,bootstrap_unit="disjoint quartet",
             training_seeds_external=1,rx_timing_anchors=len(read(root/"timing_anchors.csv")))
    (out/"verification.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data))

if __name__=="__main__":
    main()
