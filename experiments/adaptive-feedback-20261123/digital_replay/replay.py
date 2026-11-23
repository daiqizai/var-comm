"""Cost an observable feedback-to-action rule using a complete pre-existing digital-frame outcome matrix."""
import csv
from collections import Counter
import hashlib
import io
from itertools import combinations
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from study import ROOT, SPEC, TEMP_BASE, USER, digest, read, record, table

SPEC_D = read(HERE / "protocol.json")
OUT = ROOT / "digital"
OLD_RUN = TEMP_BASE / "nestcom_unified_calibration_20261103_v1"
STATES = [4,7,10,19]


def matrix(registration_dir=OUT):
    files = {"logical":"cal_pilot_plan/logical.json","physical":"cal_pilot_metric_plan/physical.json",
             "scores":"cal_pilot_dino/scores.json","choices":"cal_pilot_plan/choices.json",
             "sources":"cal_pilot_plan/sources.json"}
    repo = USER / "code/VAR_COMM"
    rel = "results/nestcom_unified_calibration_20261103_v1/pilot_calibration/candidate_scores.csv"
    reference = subprocess.check_output(["git","-C",str(repo),"show",SPEC["base_commit"]+":"+rel])
    record(registration_dir / "registration.json", {"protocol_sha256":digest(HERE/"protocol.json"),
        "code_sha256":digest(Path(__file__)), "inputs":{k:{"path":str(OLD_RUN/v),"sha256":digest(OLD_RUN/v)}
        for k,v in files.items()}, "aggregate_csv_sha256":hashlib.sha256(reference).hexdigest()})
    reference = [r for r in csv.DictReader(io.StringIO(reference.decode())) if r["method"]=="NeST"]
    ids = [s["source_id"] for s in read(OLD_RUN/files["sources"])]
    candidates = sorted(set(r["candidate_id"] for r in reference))
    assert len(ids)==len(set(ids))==100 and len(candidates)==300 and len(reference)==1200
    sid, cid = ({k:i for i,k in enumerate(keys)} for keys in (ids,candidates))
    phy, scores, choices = (read(OLD_RUN/files[k]) for k in ("physical","scores","choices"))
    val = np.full((100,4,300,2),np.nan)
    null_count, phy_used, pair_used = Counter(), set(), set()
    for row in read(OLD_RUN/files["logical"]):
        if row["method"]!="NeST":
            continue
        source, action, snr = row["source_id"],row["candidate_id"],row["snr_db"]
        assert row["N"]==1024 and row["noise_seed"]==4101 and row["channel_repeat"]==0
        idx = sid[source], STATES.index(snr), cid[action]
        assert np.isnan(val[idx]).all()
        fr = phy[row["physical_key"]]
        rec = scores[fr["pair_key"]]
        assert rec["source_id"]==source and fr["pixel_sha256"]==rec["pixel_sha256"]
        val[idx] = [rec["value"],float(fr["status"]!="ACCEPTED_SOURCE")]
        null_count[snr,action] += int(choices[row["choice_id"]]["NULL"])
        phy_used.add(row["physical_key"]); pair_used.add(fr["pair_key"])
    assert np.isfinite(val).all()
    for row in reference:
        s, a = int(row["snr_db"]),row["candidate_id"]
        col = val[:,STATES.index(s),cid[a]]
        assert int(row["source_count"])==100 and int(row["logical_count"])==100
        assert abs(col[:,0].mean()-float(row["mean_dino_cosine"]))<1e-12
        assert int(col[:,1].sum())==int(row["failed_or_null"]) and null_count[s,a]==int(row["paid_NULL"])
    record(registration_dir/"matrix_identity.json", {"logical_positions":120000,"unique_actual_physical_keys":len(phy_used),
           "unique_stored_image_pairs":len(pair_used),"candidate_cell_checks":len(reference),
           "new_frames":0,"new_receivers":0,"new_image_scores":0})
    return ids, candidates, val


def action_mix(utility, candidates, persistence, mode, params):
    bits,period,delay = params
    groups = [list(range(4))] if not bits else ([[0,1],[2,3]] if bits==1 else [[i] for i in range(4)])
    weight = np.zeros((4,len(candidates)))
    rules = []
    for phase in range(period):
        age = delay + phase
        for sym, indices in enumerate(groups):
            stay = persistence**age
            law = np.full(4,(1-stay)/4)
            law[indices] += stay/len(indices)
            scores = law @ utility
            action = int(np.flatnonzero(scores >= scores.max()-1e-12)[0])
            weight[:,action] += law*len(indices)/(4*period)
            rules.append(dict(p=persistence,mode=mode,phase=phase,age_slots=age if bits else "",
                         report=sym if bits else "", action=candidates[action],train_expected_cosine=scores[action]))
    assert np.allclose(weight.sum(axis=1),.25)
    return weight, rules


def run():
    ids, candidates, val = matrix()
    order = sorted(range(len(ids)),key=lambda i:hashlib.sha256(("digital-feedback-20261123|"+ids[i]).encode()).digest())
    train, test = order[:50],order[50:]
    record(OUT/"split.json",{"calibration":[ids[i] for i in train],"evaluation":[ids[i] for i in test]})
    train_quality = val[train,:,:,0].mean(axis=0)
    modes = list(SPEC_D["policies"])
    draws = np.random.default_rng(SPEC_D["bootstrap_seed"]).integers(len(test),size=(5000,len(test)))
    summary, pairs, rules, by_snr, sources = [], [], [], [], []
    for persistence in (.6,.95):
        outputs = {}
        for mode, params in SPEC_D["policies"].items():
            w, policy = action_mix(train_quality,candidates,persistence,mode,params)
            rules.extend(policy)
            conditional = np.einsum("sa,isam->ism",4*w,val[test])
            totals = conditional.mean(axis=1)
            outputs[mode] = totals
            for k, metric in enumerate(("dino_cosine","rejected_or_null_fraction")):
                mean = totals[:,k].mean()
                lo,hi = np.quantile(totals[draws,k].mean(axis=1),[.025,.975])
                summary.append(dict(p=persistence,mode=mode,metric=metric,mean=mean,ci_low=lo,ci_high=hi))
                for i,row in enumerate(test):
                    sources.append(dict(p=persistence,mode=mode,source_id=ids[row],metric=metric,value=totals[i,k]))
                for i,s in enumerate(STATES):
                    by_snr.append(dict(p=persistence,mode=mode,true_snr_db=s,metric=metric,mean=conditional[:,i,k].mean()))
        for left,right in combinations(modes,2):
            for k,metric in enumerate(("dino_cosine","rejected_or_null_fraction")):
                diff=outputs[left][:,k]-outputs[right][:,k]
                lo,hi=np.quantile(diff[draws].mean(axis=1),[.025,.975])
                pairs.append(dict(p=persistence,left=left,right=right,metric=metric,
                                  mean_delta=diff.mean(),ci_low=lo,ci_high=hi))
    for n,rows in [("policies",rules),("summary",summary),("paired",pairs),("by_snr",by_snr),("source_means",sources)]:
        table(OUT/(n+".csv"),rows)
    record(OUT/"completion.json",{"calibration_sources":50,"evaluation_sources":50,"means":len(summary),
          "paired":len(pairs),"policy_decisions":len(rules),"bootstrap_draws":len(draws)})


if __name__=="__main__":
    run()
