"""Aggregate first within each independent image or disjoint quartet, then resample paired units."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE/"protocol.json").read_text())
UNITS = {}
SUMMARY, CONTRASTS = [], []


def json_read(file):
    return json.loads(file.read_text())


def write(file, rows):
    with file.open("x", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def weight_bank(n):
    if n not in UNITS:
        cfg=PROTOCOL["bootstrap"]
        rng=np.random.default_rng(cfg["seed"])
        ids=rng.integers(0,n,size=(cfg["draws"],n))
        weights=np.array([np.bincount(row,minlength=n) for row in ids], dtype=np.float64)/n
        UNITS[n]=weights
    return UNITS[n]


def moments(x):
    assert np.isfinite(x).all()
    means=weight_bank(len(x)) @ x
    return x.mean(0), *np.percentile(means, [2.5,97.5], axis=0)


def tensor(rows, metrics, arm, unit, n, draws):
    by=defaultdict(lambda:defaultdict(list))
    for r in rows:
        by[arm(r)][int(r[unit])].append([float(r[k]) for k in metrics])
    grouped={}
    for a,units in by.items():
        assert set(units)==set(range(n))
        assert all(len(v)==draws for v in units.values()),a
        grouped[a]=np.array([np.mean(units[k],axis=0) for k in range(n)])
    return grouped


def scope(route, condition, arrays, metrics, pairs):
    for arm,x in arrays.items():
        results=moments(x)
        for j,metric in enumerate(metrics):
            SUMMARY.append(dict(route=route,condition=condition,arm=arm,metric=metric,units=len(x),
                                mean=results[0][j],lower95=results[1][j],upper95=results[2][j]))
    for a,b in pairs:
        delta=arrays[a]-arrays[b]
        stats=moments(delta)
        for j,metric in enumerate(metrics):
            CONTRASTS.append(dict(route=route,condition=condition,first=a,second=b,metric=metric,units=len(delta),
                                  delta=stats[0][j],lower95=stats[1][j],upper95=stats[2][j]))


def read(file):
    with file.open() as f:
        return list(csv.DictReader(f))


def front():
    arr=read(HERE/"frontend/common500_events.csv")
    assert len(arr)==20000 and len({(r["arm"],r["snr"],r["draw"],r["source"]) for r in arr})==len(arr)
    selection=json_read(HERE/"frontend/freeze.json")["selected"]
    winner = "spread_raw" if selection=="native" else selection
    metrics=("psnr","dists","scale_mae","channel_nmse","papr_db","energy")
    for snr in ("4","10"):
        x=tensor((r for r in arr if r["snr"]==snr),metrics,lambda r:r["arm"],"source",500,2)
        assert len(x)==10
        for route,arms,pairs in [
            ("layout",[k+t for k in ("spread","dedicated","superposed") for t in ("_raw","_affine")],
             [(a+t,b+t) for a,b in (("spread","dedicated"),("spread","superposed")) for t in ("_affine","_raw")]),
            ("source_scale",["spread_raw","constant","posterior"],[("spread_raw","constant"),("posterior","spread_raw")]),
            ("energy",["spread_raw","share20","share40"],[(winner,"spread_raw"),("share20","spread_raw"),("share40","spread_raw")])]:
            scope(route,snr,{k:x[k] for k in arms},metrics,list(dict.fromkeys(pairs)))


def drift():
    measures=("psnr","dists","nmse","abs_rad","corrected","energy")
    for modem in ("continuous","qam64"):
        arr=read(HERE/("drift/"+modem+"_events.csv"))
        assert len(arr)==12000 and len({(r["snr"],r["sweep"],r["receiver"],r["draw"],r["source"]) for r in arr})==len(arr)
        for snr in ("4","10"):
            for sw in ("0.0","1.2"):
                x=tensor((r for r in arr if r["snr"]==snr and r["sweep"]==sw), measures,
                         lambda r:r["receiver"],"source",500,2)
                scope("phase_drift",modem+"/"+snr+"/"+sw,x,measures,[("reference","unchanged"),
                      ("reference_minus_null","unchanged"),("reference_minus_null","reference")])


def activity():
    arr=read(HERE/"activity/mask_decisions.csv")
    assert len(arr)==8000 and len({(r["condition"],r["draw"],r["group"],r["mask"]) for r in arr})==8000
    zero={}; allevents=defaultdict(list)
    methods=("force_one","allow_empty","calibrated_F")
    for r in arr:
        mask=int(r["mask"]); load=mask.bit_count()
        if load==0:
            ident=(r["draw"],r["group"])
            codes=[r[m] for m in methods]
            if ident in zero:assert zero[ident]==codes
            zero[ident]=codes
        for method in methods:
            pred=int(r[method])
            item=dict(group=r["group"],arm=method,exact=int(pred==mask),any_false=int(bool(pred & (15^mask))),
                      any_miss=int(bool(mask & (15^pred))), missed=(mask & (15^pred)).bit_count(),
                      false_users=(pred & (15^mask)).bit_count(), loaded=load, inactive=4-load)
            allevents[(r["condition"],str(load))].append(item)
            if load:
                allevents[(r["condition"],"nonempty")].append(item)
    for (co,load),values in allevents.items():
        if load=="0" and co!="spread14":continue
        metrics=["exact","any_false","any_miss"]
        # Every quartet sees each mask twice. Compute denominators across active / inactive user slots.
        mean_load=np.mean([x["loaded"] for x in values])
        if mean_load:
            for x in values:x["miss_per_active"]=x["missed"]/mean_load
            metrics.append("miss_per_active")
        if mean_load<4:
            for x in values:x["false_per_inactive"]=x["false_users"]/(4-mean_load)
            metrics.append("false_per_inactive")
        count=len(values)//(125*3)
        arrays=tensor(values,metrics,lambda r:r["arm"],"group",125,count)
        condition="common_noise/0" if load=="0" else co+"/"+load
        scope("activity",condition,arrays,metrics,[("calibrated_F","force_one"),("calibrated_F","allow_empty"),("allow_empty","force_one")])


def separation():
    arr=read(HERE/"separation/common500_groups.csv")
    assert len(arr)==4500 and len({(r["draw"],r["condition"],r["geometry"],r["receiver"],r["group"]) for r in arr})==4500
    solo=read(HERE/"separation/solo.csv")
    selection="/".join(json_read(HERE/"separation/freeze.json")["selected"])
    m=("weak_psnr","weak_dists","mean_psnr")
    for co in ("spread14","spread18"):
        x=tensor((r for r in arr if r["condition"]==co),m,lambda r:r["geometry"]+"/"+r["receiver"],"group",125,2)
        pairs=[(selection,"learned/matched"),("learned/overlap_mmse","learned/full_mmse"),
               ("orthogonal/matched","learned/matched"),("coincident/matched","learned/matched"),
               ("coincident/full_mmse","coincident/matched"),("orthogonal/overlap_mmse","orthogonal/matched")]
        scope("separation",co,x,m,list(dict.fromkeys(pairs)))
        only=tensor((r for r in solo if r["condition"]==co),["weak_psnr"],lambda r:r["geometry"],"group",125,1)
        scope("separation_solo",co,only,["weak_psnr"],[("orthogonal","learned"),("coincident","learned")])
        both=tensor((r for r in arr if r["condition"]==co and r["draw"]=="1001" and r["receiver"]=="matched"),
                    ["weak_psnr"],lambda r:r["geometry"],"group",125,1)
        differences={g:both[g]-only[g] for g in both}
        scope("separation_collision_cost",co,differences,["weak_psnr"],[("orthogonal","learned"),("coincident","learned")])


def main():
    for role in ("frontend","drift","activity","separation"):
        assert (HERE/role/"complete.json").is_file(), role
    assert json_read(HERE/"frontend/test_banks.json")==json_read(HERE/"drift/test_banks.json")
    front()
    drift()
    activity()
    separation()
    out=HERE/"analysis"
    out.mkdir(exist_ok=False)
    write(out/"summary.csv",SUMMARY)
    write(out/"paired.csv",CONTRASTS)
    banks={n:hashlib.sha256(x.astype("<f8").tobytes()).hexdigest() for n,x in UNITS.items()}
    with (out/"statistics.json").open("x") as f:
        json.dump(dict(bootstrap=PROTOCOL["bootstrap"],weights_sha256=banks,summary=len(SUMMARY),
                       paired=len(CONTRASTS),all_four_workers_completed=True,
                       common_physical_banks_across_single_link_roles=True),f,indent=2)


if __name__=="__main__":
    main()

