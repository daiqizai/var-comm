"""Independent, standard-library spot reconstruction of the published estimands and identities."""
from pathlib import Path
import csv
import json
import statistics
import math
import hashlib
import sys

def read(path):
    with path.open(newline="",encoding="utf8") as f: return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(base):
    z=base/"analysis"
    formal=base/"formal"
    aliases=json.loads((base/"development/frozen.json").read_text())["policies"]
    maps={k:"/".join(v) for k,v in aliases.items()}
    mech=read(z/"mechanism_paired.csv")
    route=read(z/"quality_paired.csv")
    times=read(z/"timing.csv")
    srows=read(base/"secondary/matched.csv")
    sstats=read(z/"secondary.csv")
    assert len(mech)==102 and len(route)==112 and len(srows)==750
    best=maps["best_fixed"]
    arm=maps["adaptive"]
    anchors={}
    ids=json.loads((formal/"sources.json").read_text())["ids"]
    grps=[set(ids[4*g:4*g+4]) for g in range(125)]
    assert len(set.union(*grps))==sum(map(len,grps))==500
    diffs=[]
    for condition in ("nearfar_14db","nearfar_18db_ood"):
        rz=sum((read(formal/f"rotated_{condition}_{s}.csv") for s in (1001,2002,3003)),[])
        for v in rz:
            assert ids[4*int(v["group"])+int(v["slot"])]==v["weak_source"]
        obs={(r["spec"],r["seed"],r["group"]):float(r["weak_psnr"]) for r in rz}
        def eq(x,y):
            diffs.append(abs(x-y))
            assert math.isclose(x,y,abs_tol=2.e-8,rel_tol=2.e-9),(x,y)
        delta=statistics.mean(obs[arm,s,str(g)]-obs[best,s,str(g)] for s in ("1001","2002","3003") for g in range(125))
        p=next(r for r in route if r["layout"]=="rotated" and r["comparison"]=="adaptive-best_fixed" and
               r["condition"]==condition and r["metric"]=="weak_psnr")
        eq(delta,float(p["estimate"]))
        rows=[r for r in srows if r["condition"]==condition]
        for v in rows:
            eq(float(v["fixed_psnr"]),obs[best,v["seed"],v["group"]])
        eq(statistics.mean(float(v["adaptive_psnr"])-float(v["fixed_psnr"]) for v in rows),
           float(next(v["estimate"] for v in sstats if v["condition"]==condition and v["metric"]=="psnr")))
        t=read(base/"timing/batch1.csv")
        ref=statistics.mean(float(v["rx_ms"]) for v in t if v["condition"]==condition and v["spec"]==best)
        eq(ref,float(next(v["estimate"] for v in times if v["condition"]==condition and v["comparison"]=="best_fixed")))
        for s in (1001,2002,3003):
            f=formal/f"rotated_{condition}_{s}.csv"
            anchors[f.name]=sha(f)
    b=json.loads((base/"secondary/identities.json").read_text())
    assert anchors==b["fixed_csv_sha256"]
    for f in ("mechanism","development","formal","timing","secondary"):
        assert json.loads((base/f/"closed.json").read_text())["optimization_steps"]==0
    print(json.dumps(dict(checked_scalar_identities=len(diffs),maximum_absolute_difference=max(diffs),
                    original_fixed_file_hashes_matched=len(anchors),common500_unique_images=len(ids),
                    nonoverlapping_groups=125,actual_cpu_reduction="Python statistics.mean"),indent=2))

if __name__=="__main__":
    main(Path(sys.argv[1]))
