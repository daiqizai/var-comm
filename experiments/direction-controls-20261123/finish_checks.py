"""Recompute the finite mechanism-control summaries, identities and actual device ledger."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import numpy as np
from study import SPEC, STUDY, TEMP, WEIGHTS, dump_once, regular_path


def read_json(p):
    return json.loads(p.read_text())


def read_rows(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))


def files_audit(root):
    regular_path(root)
    files=byte_count=0
    for folder,dirs,names in os.walk(root,followlinks=False):
        for n in dirs+names:
            p=Path(folder)/n
            m=p.lstat()
            if stat.S_ISLNK(m.st_mode) or stat.S_ISREG(m.st_mode) and m.st_nlink!=1:
                raise RuntimeError(f"Linked object: {p}")
            if stat.S_ISREG(m.st_mode): files+=1;byte_count+=m.st_size
    return {"path":str(root),"regular_files":files,"regular_bytes":byte_count,
            "symlinks":0,"regular_files_with_multiple_names":0}


def mechanism(track):
    q=SPEC[track];p=STUDY/"evaluation"/track
    maxdiff=maxci=0.
    summary=read_rows(p/"means.csv");paired=read_rows(p/"paired.csv")
    report={}
    for split,n in (("common500",500),("kodak24",24)):
        sources=read_rows(p/(split+"_source_means.csv"))
        frames=read_rows(p/(split+"_frame_scores.csv"))
        ids=[z["source_id"] for z in read_json(p/(split+"_source_manifest.json"))]
        assert len(ids)==len(set(ids))==n
        seen=set(); groups=collections.defaultdict(list); vals=collections.defaultdict(dict)
        for z in frames:
            key=(z["condition"],z["arm"],z["role"],z["tuple_anchor"])
            assert (key,z["seed"]) not in seen; seen.add((key,z["seed"]))
            groups[key].append((int(z["seed"]),z["source_id"],np.array([float(z["psnr_db"]),float(z["dists"])])))
        alltags=set(q["arms"])|{"reference"}
        assert {z["condition"] for z in frames}==set(q["conditions"]) and {z["arm"] for z in frames}==alltags
        for z in sources:
            if z["role"]=="ALL_USERS_MEAN": continue
            key=(z["condition"],z["arm"],z["role"],z["tuple_anchor"])
            g=groups.pop(key)
            assert {t[0] for t in g}==set(q["seeds"]) and len(g)==3
            assert {t[1] for t in g}=={z["source_ids"]}
            expected=np.mean([t[2] for t in g],axis=0)
            assert np.max(np.abs(expected-[float(z["psnr_db"]),float(z["dists"])]))<1e-10
            vals[key[:3]][key[3]]=expected
        assert not groups
        for z in sources:
            if z["role"]!="ALL_USERS_MEAN":continue
            key=(z["condition"],z["arm"],z["role"])
            match=[v[z["tuple_anchor"]] for k,v in vals.items() if k[:2]==key[:2] and k[2]!="ALL_USERS_MEAN"]
            exp=np.mean(match,axis=0)
            assert np.max(np.abs(exp-[float(z["psnr_db"]),float(z["dists"])]))<1e-10
            vals[key][z["tuple_anchor"]]=exp
        arrays={k:np.stack([v[j] for j in ids]) for k,v in vals.items()}
        rng=np.random.default_rng(20261124 if track=="noma" else 20261123)
        draws=rng.integers(0,n,size=(5000,n))
        idx={"psnr_db":0,"dists":1}
        for row in [z for z in summary+paired if z["split"]==split]:
            col=idx[row["metric"]]
            def arm(name):
                return arrays[(row["condition"],name,row["role"])][:,col]
            a=arm(row["arm"]) if "arm" in row else arm(row["left"])-arm(row["right"])
            ci=np.quantile(np.mean(a[draws],axis=1),[.025,.975])
            maxdiff=max(maxdiff,abs(a.mean()-float(row["mean"])))
            maxci=max(maxci,float(np.max(np.abs(ci-[float(row["ci95_lo"]),float(row["ci95_hi"])]))))
        report[split]={"actual_per_image_and_user_scores":len(frames),"source_mean_rows":len(sources),
                      "sources":n,"repeats":3,"methods":len(alltags),"conditions":len(q["conditions"])}
    assert max(maxdiff,maxci)<1e-10
    return {"splits":report,"max_mean_recalculation_error":maxdiff,"max_paired_or_mean_CI_error":maxci,
            "mean_CIs":len(summary),"paired_CIs":len(paired)}


def main():
    checks={k:mechanism(k) for k in SPEC}
    resource=[]
    for name,q in SPEC.items():
        run=TEMP/"runs"/name
        for part in q["arms"]+["evaluation"]:
            a=read_json(run/f"{part}_start.json");b=read_json(run/f"{part}_wait.json")
            assert b["actual_wait"]==0 and a["pid"]==b["pid"]
            resource.append({"task":name+"/"+part,"physical_gpu":q["gpu"],"actual_wait":b["actual_wait"],
                             "pid":b["pid"],"device_seconds":b["elapsed_s"],"start":a["start_unix"]})
    for name in ("dir5_receiver","dir5_waveform"):
        run=TEMP/"runs"/name
        a=read_json(run/"start.json");b=read_json(run/"wait.json");assert b["actual_wait"]==0 and a["pid"]==b["pid"]
        resource.append({"task":name,"physical_gpu":a["gpu"],"actual_wait":b["actual_wait"],
                         "pid":b["pid"],"device_seconds":b["seconds"],"start":a["start"]})
    main_hrs=sum(z["device_seconds"] for z in resource[:6])/3600
    assert main_hrs<=12
    assert read_json(STUDY/"dir5/published_v2_identity.json")["checkpoints_matching_all"]==14
    for tag in ("observed_v1","published_v2"):
        assert read_json(STUDY/"dir5"/tag/"audit.json")["paired_quantities_checked"]==436
    for f in ("channel_check.json","receiver_val256/identity.json","waveform_val256/identity.json"):
        assert (STUDY/"dir5"/f).is_file()
    dump_once(STUDY/"data_checks.json",checks)
    dump_once(STUDY/"resource_ledger.json",{"jobs":resource,"matched_study_device_hours":main_hrs,
          "all_new_gpu_device_hours":sum(z["device_seconds"] for z in resource)/3600,
          "historical_producer_runs_charged_here":0,"cpu_import_check_failures":1,
          "time_basis":"wall time per actual GPU worker; simultaneous jobs counted separately"})
    roots=[STUDY,WEIGHTS,TEMP/"runs",TEMP/"cache",TEMP/"source",TEMP/"producer_snapshots",TEMP/"waveform_control_src"]
    dump_once(STUDY/"storage_audit.json",{"recorded_at_unix":time.time(),"checks":[files_audit(p) for p in roots],
          "scope":"Study-owned scientific files, weights, scratch and captured evidence. Other liulu projects are outside this check."})
    print("All frame-to-source means and CIs reproduced",checks,flush=True)


if __name__=="__main__": main()
