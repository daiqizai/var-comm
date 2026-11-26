"""Paired inferences use an entire four-source collision as the sampling unit."""
import argparse
import csv
import itertools
import json
import hashlib
from pathlib import Path
import numpy as np

SEEDS=(1001,2002,3003)
CONDITIONS=("nearfar_14db","nearfar_18db_ood")
METRICS=("antenna_db","combiner_db","interference_noise_db","latent_db","frontend_db")
ROUTING=("weak_psnr","weak_dists","mean_psnr","rgb_calls")
rng=np.random.default_rng(271127)
banks={n:rng.integers(0,n,size=(10000,n),dtype=np.int32) for n in (125,32)}

def read(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def write(path,records):
    with path.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

def dump(path,item):
    with path.open("x",encoding="utf-8") as f:
        json.dump(item,f,indent=2,allow_nan=False)

def stat(x):
    assert x.ndim==1 and np.isfinite(x).all()
    y=x[banks[len(x)]].mean(1)
    return dict(estimate=float(x.mean()),lo=float(np.quantile(y,.025)),hi=float(np.quantile(y,.975)))

def ratio(a,b):
    bank=banks[len(a)]
    den=b[bank].mean(1)
    assert np.isfinite(den).all() and (np.abs(den)>1.e-10).all()
    y=a[bank].mean(1)/den
    return dict(estimate=float(a.mean()/b.mean()),lo=float(np.quantile(y,.025)),hi=float(np.quantile(y,.975)))

def collect(z,spec,metric):
    items={ (int(row["seed"]),int(row["group"])):float(row[metric]) for row in z if spec(row)}
    assert len(items)==len(SEEDS)*125
    return np.array([[items[s,g] for g in range(125)] for s in SEEDS]).mean(0)

def mechanism(study,out):
    means,contrasts=[],[]
    for cond in CONDITIONS:
        z=sum((read(study/"mechanism"/f"{cond}_{s}.csv") for s in SEEDS),[])
        assert len(z)==3375
        for rnd in (1,2,3):
            metrics=METRICS+(("weak_psnr","weak_dists") if rnd==3 else ())
            for met in metrics:
                arms={a:collect(z,lambda v:v["replica"]==a and int(v["round"])==rnd,met) for a in ("linear","latent","rgb")}
                keys=dict(condition=cond,round=rnd,metric=met)
                for arm,arr in arms.items():
                    means.append(dict(**keys,arm=arm,**stat(arr)))
                for a,b in (("latent","linear"),("rgb","latent"),("rgb","linear")):
                    diff=arms[a]-arms[b]
                    contrasts.append(dict(**keys,comparison=f"{a}-{b}",**stat(diff),positive_quartets=int((diff>0).sum()),n=125))
    write(out/"mechanism_means.csv",means)
    write(out/"mechanism_paired.csv",contrasts)

def routing(study,out):
    path=study/"development/frozen.json"
    freeze=json.loads(path.read_text())
    fmap={k:"/".join(v) for k,v in freeze["policies"].items()}
    means,pairs,pct,case=[],[],[],[]
    counts={}
    timings=[]
    for layout,cond in itertools.product(("original","rotated"),CONDITIONS):
        z=sum((read(study/"formal"/f"{layout}_{cond}_{s}.csv") for s in SEEDS),[])
        keys=dict(layout=layout,condition=cond)
        assert len(z)==375*len(set(fmap.values()))
        for rz in z:
            if rz["spec"] not in ("-/latent","-/full"):
                assert 0<=int(rz["rgb_calls"])<=3
        metric_arrays={}
        for met in ROUTING:
            arms={a: collect(z,lambda v:v["spec"]==spec,met) for a,spec in fmap.items()}
            metric_arrays[met]=arms
            for a,arr in arms.items():
                means.append(dict(**keys,metric=met,arm=a,**stat(arr)))
            for a,b in (("adaptive","best_fixed"),("adaptive","matched_fixed"),("adaptive","latent"),("adaptive","full"),
                        ("best_fixed","latent"),("full","latent"),("full","best_fixed")):
                diff=arms[a]-arms[b]
                pairs.append(dict(**keys,metric=met,comparison=f"{a}-{b}",**stat(diff),positive_quartets=int((diff>0).sum()),n=125))
        for arm in ("best_fixed","adaptive"):
            arr=metric_arrays["weak_psnr"]
            pct.append(dict(**keys,arm=arm,metric="fraction_of_full_minus_latent_weak_psnr",
                            **ratio(arr[arm]-arr["latent"],arr["full"]-arr["latent"])))
        for a,spec in fmap.items():
            obs=[r for r in z if r["spec"]==spec]
            counts[f"{layout}/{cond}/{a}"]={"active_hat_4":sum(int(v["active_hat"])==4 for v in obs),
                 "all":len(obs),"calls":sum(int(v["rgb_calls"]) for v in obs)}
        for aa in ("best_fixed","matched_fixed"):
            ids=lambda r:(r["seed"],r["group"])
            masks={ids(r):r["users_each_round"] for r in z if r["spec"]==fmap[aa]}
            ad=[r for r in z if r["spec"]==fmap["adaptive"]]
            eq=sum(masks[ids(row)]==row["users_each_round"] for row in ad)
            case.append(dict(**keys,comparator=aa,identical_user_selections=eq,total=len(ad)))
    write(out/"quality_means.csv",means)
    write(out/"quality_paired.csv",pairs)
    write(out/"quality_recovery_fraction.csv",pct)
    write(out/"same_choices.csv",case)
    raw=read(study/"timing/batch1.csv")
    for cond in CONDITIONS:
        arms={}
        for name,spec in fmap.items():
            v={(int(r["group"]),int(r["repeat"])):float(r["rx_ms"]) for r in raw if r["condition"]==cond and r["spec"]==spec}
            assert len(v)==64
            arr=np.array([[v[g,t] for t in range(2)] for g in range(32)]).mean(1)
            arms[name]=arr
            timings.append(dict(condition=cond,quantity="mean_rx_ms",comparison=name,**stat(arr)))
        for a,b in (("adaptive","best_fixed"),("adaptive","matched_fixed"),("adaptive","latent"),("adaptive","full"),("best_fixed","full"),("best_fixed","latent")):
            timings.append(dict(condition=cond,quantity="difference_ms",comparison=a+"-"+b,**stat(arms[a]-arms[b])))
            timings.append(dict(condition=cond,quantity="ratio",comparison=a+"/"+b,**ratio(arms[a],arms[b])))
    write(out/"timing.csv",timings)
    verdict=[]
    for cond in CONDITIONS:
        p=next(r for r in pairs if r["layout"]=="rotated" and r["condition"]==cond and r["metric"]=="weak_psnr" and r["comparison"]=="adaptive-best_fixed")
        t=next(r for r in timings if r["condition"]==cond and r["quantity"]=="ratio" and r["comparison"]=="adaptive/best_fixed")
        verdict.append(dict(condition=cond,quality_lower_gt_zero=p["lo"]>0,latency_upper_le_1_05=t["hi"]<=1.05,
                            paired_primary_endpoint_pass=p["lo"]>0 and t["hi"]<=1.05))
    dump(out/"audit.json",dict(bootstrap_quartet_draws=10000,seed=271127,source_pop=500,groups=125,
         timing_groups=32,bank_hashes={str(k):hashlib.sha256(v.tobytes()).hexdigest() for k,v in banks.items()},
         policy_freeze_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rule_comparison=verdict,activity_and_call_counts=counts))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("mode",choices=["mechanism","routing"])
    a=p.parse_args()
    dst=a.study/"analysis"
    dst.mkdir(exist_ok=True)
    if a.mode=="mechanism":
        assert (a.study/"mechanism/closed.json").exists()
        mechanism(a.study,dst)
    else:
        assert all((a.study/d/"closed.json").is_file() for d in ("development","formal","timing"))
        routing(a.study,dst)
