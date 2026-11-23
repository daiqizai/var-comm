"""Source-tuple contrasts for prior-based and model-based parallel cancellation."""
import csv
import numpy as np
from common import PRIMARY, PRIOR, STUDY, digest, read, record, table

CONTRASTS=(("full","plain"),("full","raw_gate"),("raw_gate","plain"))


def main():
    root=STUDY/"pic"
    assert read(root/"wait.json")["actual_wait"]==0
    sources,means,pairs=[],[],[]
    files={}
    for split,n in (("common500",500),("kodak24",24)):
        path=root/(split+"_frames.csv")
        files[split]=digest(path)
        ids=[k["source_id"] for k in read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))]
        idx={s:i for i,s in enumerate(ids)}
        raw={}
        for z in csv.DictReader(path.open()):
            group=z["role"],z["arm"]
            if group not in raw: raw[group]=np.full((3,n,2),np.nan)
            row=int(z["seed"])-3001
            pos=idx[z["tuple_anchor"]]
            assert 0<=row<3 and np.isnan(raw[group][row,pos]).all()
            raw[group][row,pos]=[float(z[k]) for k in PRIMARY]
        assert len(raw)==9 and all(np.isfinite(v).all() for v in raw.values())
        value={k:v.mean(axis=0) for k,v in raw.items()}
        roles=("User1_13dB","User2_8dB","User3_4dB")
        for a in ("full","plain","raw_gate"):
            value["all_users",a]=sum(value[g,a] for g in roles)/3
        draws=np.random.default_rng(2026112308).integers(0,n,size=(5000,n))
        disjoint=n//3
        ddraws=np.random.default_rng(2026112308).integers(0,disjoint,size=(5000,disjoint))
        for (g,a),v in value.items():
            ci=np.quantile(v[draws].mean(axis=1),[.025,.975],axis=0)
            for mk,mean,lo,hi in zip(PRIMARY,v.mean(0),ci[0],ci[1]):
                means.append(dict(split=split,condition=g,arm=a,metric=mk,n=n,mean=mean,ci95_lo=lo,ci95_hi=hi))
            for j,sid in enumerate(ids):
                sources.append(dict(split=split,condition=g,arm=a,unit=sid,psnr_db=v[j,0],dists=v[j,1]))
        for g in (*roles,"all_users"):
            for a,b in CONTRASTS:
                d=value[g,a]-value[g,b]
                ci=np.quantile(d[draws].mean(axis=1),[.025,.975],axis=0)
                ci2=np.quantile(d[:disjoint][ddraws].mean(axis=1),[.025,.975],axis=0)
                for j,mk in enumerate(PRIMARY):
                    pairs.append(dict(split=split,condition=g,contrast=a+"_minus_"+b,metric=mk,n=n,
                        mean=d[:,j].mean(),ci95_lo=ci[0,j],ci95_hi=ci[1,j],disjoint_n=disjoint,
                        disjoint_mean=d[:disjoint,j].mean(),disjoint_ci95_lo=ci2[0,j],disjoint_ci95_hi=ci2[1,j]))
    for stem,rows in (("means",means),("paired",pairs),("source_means",sources)):
        table(root/(stem+".csv"),rows)
    record(root/"statistics_identity.json",dict(frame_sha256=files,mean_intervals=len(means),paired_intervals=len(pairs),
          disjoint_intervals=len(pairs),noise_seeds=[3001,3002,3003],bootstrap_seed=2026112308,draws=5000,
          split_frames={"common500":13500,"kodak24":648}))


if __name__=="__main__":
    main()
