"""Validate each complete factorial cell and compute paired, equal-source bootstrap intervals."""
import argparse
import csv
from collections import defaultdict
import numpy as np
from common import PRIMARY, SIC_ARMS, STUDY, WAVE_ARMS, digest, read, record, table


def interval(v, draws):
    lo,hi = np.quantile(v[draws].mean(axis=1),[.025,.975])
    return dict(n=len(v),mean=float(v.mean()),ci95_lo=float(lo),ci95_hi=float(hi))


def contrast_map(track):
    if track == "noma":
        return {
            "both_minus_neither":{"both":1,"neither":-1},
            "both_minus_covariance":{"both":1,"covariance":-1},
            "both_minus_cancel":{"both":1,"cancel":-1},
            "cancel_minus_neither":{"cancel":1,"neither":-1},
            "covariance_minus_neither":{"covariance":1,"neither":-1},
            "interaction":{"both":1,"cancel":-1,"covariance":-1,"neither":1}}
    if track == "wave_test":
        return {
            "matched_spread_minus_no_spread":{"spread/spread":1,"no_spread/no_spread":-1},
            "spread_on_spread_weights":{"spread/spread":1,"spread/no_spread":-1},
            "spread_on_no_spread_weights":{"no_spread/spread":1,"no_spread/no_spread":-1},
            "train_spread_at_spread":{"spread/spread":1,"no_spread/spread":-1},
            "train_spread_at_no_spread":{"spread/no_spread":1,"no_spread/no_spread":-1},
            "interaction":{"spread/spread":1,"spread/no_spread":-1,"no_spread/spread":-1,"no_spread/no_spread":1}}
    out = {k+"_rf_minus_blackbox":{"rf_bussgang/"+k:1,"blackbox_rf/"+k:-1}
           for k in ("frame_rms","fixed","causal")}
    for method in ("rf_bussgang","blackbox_rf"):
        for a,b in (("fixed","frame_rms"),("causal","frame_rms"),("causal","fixed")):
            out[method+"_"+a+"_minus_"+b] = {method+"/"+a:1,method+"/"+b:-1}
    out["interaction_fixed_vs_frame_rms"] = {"rf_bussgang/fixed":1,"blackbox_rf/fixed":-1,
                                             "rf_bussgang/frame_rms":-1,"blackbox_rf/frame_rms":1}
    return out


def layout(track, split):
    if track=="noma":
        return ["User1_13dB","User2_8dB","User3_4dB"], list(SIC_ARMS), {3001,3002,3003}
    if track=="wave_test":
        return ["0.1","0.18"], [a+"/"+b for a in WAVE_ARMS for b in WAVE_ARMS], {5001,5002,5003}
    return (["0","-6","6"] if split=="kodak24" else ["0"]), [
        a+"/"+b for a in ("rf_bussgang","blackbox_rf") for b in ("frame_rms","fixed","causal")], {3001,3002,3003}


def aggregate(track):
    root = STUDY/track
    assert read(root/("test_wait.json" if track=="rf" else "wait.json"))["actual_wait"] == 0
    means,pairs,sources = [],[],[]
    inputs, frame_counts = {},{}
    for split,n in (("common500",500),("kodak24",24)):
        groups,arms,seeds = layout(track,split)
        path = root/(split+"_frames.csv")
        inputs[split] = digest(path)
        raw = defaultdict(dict)
        ids = []
        seen_ids = set()
        total = 0
        with path.open() as file:
            for row in csv.DictReader(file):
                assert row["split"] == split
                source = row["tuple_anchor"] if track=="noma" else row["source_id"]
                if source not in seen_ids:
                    ids.append(source)
                    seen_ids.add(source)
                if track=="noma":
                    group,arm = row["role"],row["arm"]
                elif track=="wave_test":
                    group,arm = row["kd"],row["trained"]+"/"+row["deploy"]
                else:
                    group,arm = row["gain_db"],row["model"]+"/"+row["mode"]
                seed = int(row["seed"])
                cell = raw[group,arm,source]
                if seed in cell or group not in groups or arm not in arms or seed not in seeds:
                    raise ValueError((group,arm,source,seed))
                cell[seed] = [float(row[metric]) for metric in PRIMARY]
                total += 1
        assert len(ids)==n and total==n*len(groups)*len(arms)*3
        assert all(set(cell)==seeds for cell in raw.values())
        frame_counts[split] = total
        arrays = {g:{a:np.asarray([np.mean([raw[g,a,i][j] for j in sorted(seeds)],axis=0) for i in ids])
                     for a in arms} for g in groups}
        if track=="noma":
            arrays["all_users"] = {a:np.mean([arrays[g][a] for g in groups],axis=0) for a in arms}
        draws = np.random.default_rng(2026112308).integers(0,n,size=(5000,n))
        for g,values in arrays.items():
            for a,v in values.items():
                assert v.shape==(n,2) and np.isfinite(v).all()
                for i,k in enumerate(ids):
                    sources.append(dict(split=split,condition=g,unit=k,arm=a,psnr_db=v[i,0],dists=v[i,1]))
                for m,key in enumerate(PRIMARY):
                    means.append(dict(split=split,condition=g,arm=a,metric=key,**interval(v[:,m],draws)))
            for label,terms in contrast_map(track).items():
                d=sum(weight*values[arm] for arm,weight in terms.items())
                for m,key in enumerate(PRIMARY):
                    pairs.append(dict(split=split,condition=g,contrast=label,metric=key,**interval(d[:,m],draws)))
    for stem,rows in (("source_means",sources),("means",means),("paired",pairs)):
        table(root/(stem+".csv"),rows)
    record(root/"statistics_identity.json",dict(frame_sha256=inputs,logical_frame_scores=frame_counts,
           bootstrap_seed=2026112308,source_bootstrap_draws=5000,noise_realizations=3,
           source_unit="tuple anchor; users averaged within anchor for all_users" if track=="noma" else "source image",
           mean_intervals=len(means),paired_intervals=len(pairs),source_mean_rows=len(sources)))
    print(track,frame_counts,len(pairs),"paired intervals",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("track",choices=["noma","wave_test","rf"])
    aggregate(p.parse_args().track)
