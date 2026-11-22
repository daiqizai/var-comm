"""Independently reproduce the new training-control statistics and total device ledger."""
import collections
import csv
import json

import numpy as np
from dir5_training_pair import ARMS, GPU, LIMIT, ROOT, RUN, SRC
from finish_checks import files_audit
from study import STUDY, TEMP, WEIGHTS, dump_once


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    out=ROOT/"evaluation"
    keys=("psnr_db","dists","lpips_alex")
    identity=json.loads((out/"identity.json").read_text())
    frames=rows(out/"repeat_scores.csv")
    sources=rows(out/"source_means.csv")
    summary=rows(out/"means.csv")
    paired=rows(out/"paired.csv")
    assert (len(frames),len(sources),len(summary),len(paired))==(4716,1572,18,18)
    largest=0.
    for split,n in (("common500",500),("kodak24",24)):
        ids=identity["source_and_channel"][split]["source_ids"]
        assert len(ids)==len(set(ids))==n
        groups=collections.defaultdict(dict)
        for row in frames:
            if row["split"]!=split:continue
            key=(row["cell"],row["source_id"]);seed=int(row["seed"])
            assert seed not in groups[key]
            groups[key][seed]=np.array([float(row[k]) for k in keys])
        assert len(groups)==3*n
        averages={}
        for row in sources:
            if row["split"]!=split:continue
            key=(row["cell"],row["source_id"]);g=groups.pop(key)
            assert set(g)=={5001,5002,5003}
            x=np.mean([g[j] for j in (5001,5002,5003)],axis=0)
            largest=max(largest,float(np.abs(x-[float(row[k]) for k in keys]).max()))
            averages[key]=x
        assert not groups
        index=np.random.default_rng(2026112306).integers(0,n,size=(5000,n))
        def vector(cell,metric):
            col=keys.index(metric)
            return np.array([averages[cell,i][col] for i in ids])
        for row in summary+paired:
            if row["split"]!=split:continue
            metric=row["metric"]
            values=vector(row["cell"],metric) if "cell" in row else vector(row["left"],metric)-vector(row["right"],metric)
            found=np.r_[values.mean(),np.quantile(values[index].mean(axis=1),[.025,.975])]
            original=np.array([float(row[k]) for k in ("mean","ci95_lo","ci95_hi")])
            largest=max(largest,float(np.abs(found-original).max()))
    assert largest<1e-10
    init=[json.loads((ROOT/"training"/arm/"initialization.json").read_text()) for arm in ARMS]
    assert init[0]["trainable_state_sha256"]==init[1]["trainable_state_sha256"]
    assert [s["disable_projection"] for s in init]==[False,True]
    ledger=json.loads((ROOT/"resource.json").read_text())
    assert len(ledger["jobs"])==4
    elapsed=0.
    for key,rec in ledger["jobs"].items():
        start=json.loads((RUN/(key+"_start.json")).read_text())
        assert rec["actual_wait"]==0 and rec["pid"]==start["pid"] and start["physical_gpu"]==GPU
        elapsed+=rec["device_seconds"]
    assert elapsed<=LIMIT and abs(elapsed/3600-ledger["device_hours"])<1e-10
    dump_once(ROOT/"review.json",{"source_rows":len(sources),"repeat_scores":len(frames),
        "mean_CIs":len(summary),"paired_CIs":len(paired),"max_absolute_statistic_recalculation_error":largest,
        "identical_initial_trainable_tensor_sha256":init[0]["trainable_state_sha256"],
        "file_checks":[files_audit(f) for f in (ROOT,RUN,SRC,WEIGHTS/"dir5_train_control",TEMP/"cache/dir5_train_control")]})
    first=json.loads((STUDY/"resource_ledger.json").read_text())
    dump_once(STUDY/"combined_resource.json",{"initial_mechanism_and_frozen_receiver_studies":first["all_new_gpu_device_hours"],
        "additional_matched_5B_training_and_test":ledger["device_hours"],
        "total_actual_A800_device_hours":first["all_new_gpu_device_hours"]+ledger["device_hours"],
        "existing_producer_training_and_formal_reevaluations_charged_here":0,
        "unit":"Wall time of each occupied GPU worker, added across devices. The 5B extension used one GPU sequentially."})
    print("5B intervals and paired arm identities verified",flush=True)


if __name__=="__main__": main()
