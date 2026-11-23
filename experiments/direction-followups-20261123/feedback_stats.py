"""Source and trajectory-block sensitivity intervals for the frozen feedback study."""
import csv
from collections import defaultdict
import importlib.util
from pathlib import Path
import numpy as np
from common import PRIMARY, STUDY, digest, read, record, table

_spec=importlib.util.spec_from_file_location("paired_study_statistics",Path(__file__).with_name("statistics.py"))
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
interval=_module.interval


def main():
    root=STUDY/"feedback"
    assert read(root/"wait.json")["actual_wait"]==0
    sources,means,pairs=[],[],[]
    frame_sha={}
    for split,n in (("common500",500),("kodak24",24)):
        path=root/(split+"_frames.csv")
        frame_sha[split]=digest(path)
        raw=defaultdict(dict)
        ids=[]
        seen=set()
        for row in csv.DictReader(path.open()):
            sid=row["source_id"]
            if sid not in seen:
                seen.add(sid)
                ids.append(sid)
            arm=row["model"]+"/"+row["mode"]
            cell=raw[row["scenario"],arm,sid]
            assert int(row["seed"]) not in cell
            cell[int(row["seed"])]=[float(row[k]) for k in PRIMARY]
        assert len(seen)==n and len(raw)==n*6*2
        assert all(set(c)=={2001,2002,2003} for c in raw.values())
        rng=np.random.default_rng(2026112308)
        draw=rng.integers(0,n,(5000,n))
        # Contiguous images share one serial channel trajectory in each realization.
        # Resample aligned blocks after averaging the same 3 trajectories in both arms.
        starts=rng.integers(0,n,(5000,(n+7)//8))
        blocks=((starts[:,:,None]+np.arange(8))%n).reshape(5000,-1)[:,:n]
        arms=[m+"/"+t for m in ("rx","joint") for t in ("known","constant","history")]
        terms={f"joint_minus_rx_{m}":{"joint/"+m:1,"rx/"+m:-1} for m in ("known","constant","history")}
        for method in ("rx","joint"):
            for a,b in (("history","known"),("constant","known"),("history","constant")):
                terms[method+"_"+a+"_minus_"+b]={method+"/"+a:1,method+"/"+b:-1}
        terms["joint_gain_history_minus_known"]={"joint/history":1,"rx/history":-1,"joint/known":-1,"rx/known":1}
        for sc in ("snr10_rho090","snr4_rho085"):
            arr={}
            for arm in arms:
                a=np.asarray([np.mean([raw[sc,arm,i][z] for z in (2001,2002,2003)],axis=0) for i in ids])
                assert a.shape==(n,2) and np.isfinite(a).all()
                arr[arm]=a
                for k,(p,d) in zip(ids,a):
                    sources.append(dict(split=split,condition=sc,unit=k,arm=arm,psnr_db=p,dists=d))
                for j,metric in enumerate(PRIMARY):
                    means.append(dict(split=split,condition=sc,arm=arm,metric=metric,**interval(a[:,j],draw)))
            for label,weights in terms.items():
                d=sum(v*arr[a] for a,v in weights.items())
                for j,metric in enumerate(PRIMARY):
                    circ=interval(d[:,j],blocks)
                    pairs.append(dict(split=split,condition=sc,contrast=label,metric=metric,
                        **interval(d[:,j],draw),block8_ci95_lo=circ["ci95_lo"],block8_ci95_hi=circ["ci95_hi"]))
    for name,rows in (("source_means",sources),("means",means),("paired",pairs)):
        table(root/(name+".csv"),rows)
    record(root/"statistics_identity.json",{"frame_sha256":frame_sha,"mean_intervals":len(means),
          "paired_intervals":len(pairs),"additional_block8_intervals":len(pairs),"draws":5000,"seed":2026112308,
          "logical_frame_scores":{"common500":18000,"kodak24":864}})


if __name__=="__main__":
    main()
