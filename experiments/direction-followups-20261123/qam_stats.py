"""Source-paired V4 comparisons under measured-frame and fixed receiver noise."""
import csv
import os
import subprocess
import sys
import time
import numpy as np
from common import PRIMARY, PRIOR, PY, STUDY, TEMP, digest, include_sources, read, record, table

ROOT=STUDY/"qam"


def statistics():
    assert read(ROOT/"wait.json")["actual_wait"]==0
    sources,means,pairs=[],[],[]
    for split,n in (("common500",500),("kodak24",24)):
        ids=[r["source_id"] for r in read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))]
        index={v:i for i,v in enumerate(ids)}
        cells={(method,mode):np.full((3,n,2),np.nan) for method in ("dd","blind","pilot") for mode in ("native","fixed")}
        for r in csv.DictReader((ROOT/(split+"_frames.csv")).open()):
            pos=cells[r["method"],r["mode"]][int(r["seed"])-2001,index[r["source_id"]]]
            assert np.isnan(pos).all()
            pos[:]=[float(r[k]) for k in PRIMARY]
        assert all(np.isfinite(v).all() for v in cells.values())
        images={key:v.mean(axis=0) for key,v in cells.items()}
        draws=np.random.default_rng(2026112308).integers(0,n,size=(5000,n))
        def summarize(arr):
            ci=np.percentile(arr[draws].mean(axis=1),[2.5,97.5],axis=0)
            return [dict(split=split,condition="64QAM_10dB",metric=m,n=n,
                         mean=arr[:,j].mean(),ci95_lo=ci[0,j],ci95_hi=ci[1,j]) for j,m in enumerate(PRIMARY)]
        for (method,mode),arr in images.items():
            arm=method+"/"+mode
            means.extend(dict(**r,arm=arm) for r in summarize(arr))
            sources.extend(dict(split=split,condition="64QAM_10dB",arm=arm,unit=ids[j],
                                psnr_db=p,dists=d) for j,(p,d) in enumerate(arr))
        for ref in ("blind","pilot"):
            diffs={m:images["dd",m]-images[ref,m] for m in ("native","fixed")}
            for mode,arr in diffs.items():
                pairs.extend(dict(**r,contrast=f"dd_minus_{ref}_{mode}") for r in summarize(arr))
            pairs.extend(dict(**r,contrast=f"fixed_minus_native_dd_gain_over_{ref}")
                         for r in summarize(diffs["fixed"]-diffs["native"]))
        for method in ("dd","blind","pilot"):
            pairs.extend(dict(**r,contrast=method+"_fixed_minus_native") for r in
                         summarize(images[method,"fixed"]-images[method,"native"]))
    for stem,rows in (("means",means),("paired",pairs),("source_means",sources)):
        table(ROOT/(stem+".csv"),rows)
    record(ROOT/"statistics_identity.json",{"draws":5000,"seed":2026112308,"paired_intervals":len(pairs),
           "mean_intervals":len(means),"frame_files":{s:digest(ROOT/(s+"_frames.csv")) for s in ("common500","kodak24")}})


def continue_after_wave(gpu):
    from after_training import wait_for_roles
    from worker import admission
    assert read(ROOT/"preflight.json")["images"]==0
    wait_for_roles(["wave_test"])
    time.sleep(12)
    samples=admission(gpu)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
             PYTHONPATH=include_sources(),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",
             CUBLAS_WORKSPACE_CONFIG=":4096:8")
    cmd=[str(PY),"-B",str(STUDY/"qam.py"),"run"]
    with (TEMP/"logs/qam_test.log").open("xb") as log:
        start=time.time()
        p=subprocess.Popen(cmd,env=env,cwd=STUDY,stdout=log,stderr=subprocess.STDOUT)
        with open(f"/proc/{p.pid}/stat") as f:
            ticks=f.read().split(") ")[1].split()[19]
        record(ROOT/"start.json",dict(pid=p.pid,start_ticks=ticks,started_unix=start,command=cmd,
               admission=samples,physical_gpu=gpu,script_sha256=digest(STUDY/"qam.py"),protocol_sha256=digest(ROOT/"protocol.json")))
        try:
            rc=p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            p.terminate()
            try: rc=p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                rc=p.wait()
        record(ROOT/"wait.json",dict(pid=p.pid,actual_wait=rc,device_seconds=time.time()-start))
        if rc: raise RuntimeError(f"V4 QAM receiver comparison exited {rc}")
    statistics()


if __name__=="__main__":
    if len(sys.argv)>1:
        continue_after_wave(int(sys.argv[1]))
    else:
        statistics()
