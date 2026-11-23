"""Recover V4 scoring after a missing promisor blob, with verified offline inputs and durable batches."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from common import (D2, PY, STUDY, TEMP, USER, digest, include_sources, path_check, read,
                    record, save, single_replace)

ROOT=STUDY/"qam"
MIRROR=TEMP/"qam_input_mirror"
DRIVER=TEMP/"qam_recovery/qam_fixed_io.py"
STATS=TEMP/"qam_recovery/qam_fixed_stats.py"
REV="65a5a8e14b92eb7fad92eb6e6444418e52d9a073"
BASELINE="results/swinjscc_mismatch_zpac_20261120/v4_universal_eval/"


def git_blob_hash(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def prepare():
    failed=read(ROOT/"wait.json")
    assert failed["actual_wait"]==1 and not (ROOT/"common500_frames.csv").exists()
    assert read(ROOT/"frozen_gains.json")["validation_sources"]==256
    log=(TEMP/"logs/qam_test.log").read_text()
    assert "could not fetch" in log and "old=list(csv.DictReader(blob(RESULT+" in log
    paths=list(read(ROOT/"protocol.json")["bound_sources"])
    paths+=[BASELINE+f for f in ("eval_v4_summary_common500.json","v4_s3_digital_qam_common500.csv","v4_s3_digital_qam_kodak24.csv")]
    source_repo=USER/"code/VAR_COMM.dir2-swinjscc-mismatch-v1"
    identities={}
    for name in paths:
        item=subprocess.check_output(["git","-C",str(source_repo),"ls-tree",REV,name]).decode().strip()
        info,actual_name=item.split("\t")
        mode,kind,oid=info.split()
        assert (mode,kind,actual_name)==("100644","blob",name)
        src=path_check(source_repo/name)
        data=src.read_bytes() if src.is_file() else b""
        if git_blob_hash(data)!=oid:
            data=subprocess.check_output(["git","-C",str(source_repo),"cat-file","blob",oid])
        assert git_blob_hash(data)==oid
        save(MIRROR/name,data)
        identities[name]=dict(git_blob=oid,sha256=digest(MIRROR/name))
    parent=STUDY/"qam.py"
    assert digest(parent)==read(ROOT/"start.json")["script_sha256"]
    text=parent.read_text()
    old='    return subprocess.check_output(["git","-C",str(USER/"code/VAR_COMM"),"show",REV+":"+rel])'
    text=single_replace(text,old,'''    p=TEMP/"qam_input_mirror"/rel
    assert digest(p)==read(ROOT/"io_recovery.json")["offline_inputs"][rel]["sha256"]
    return p.read_bytes()''')
    text=single_replace(text,'for split in ("val256","common500","kodak24"):','for split in ("common500","kodak24"):')
    text=single_replace(text,'            x=ds.images_u8[a:b].cuda().float()/255',
                        '            begin=len(rows)\n            x=ds.images_u8[a:b].cuda().float()/255')
    text=single_replace(text,"        if cal:\n            avg=",'''            table(ROOT/"recovered_batches"/f"{split}_{a:04d}.csv",rows[begin:])
        if cal:
            avg=''')
    text=single_replace(text,'        table(ROOT/(split+"_frames.csv"),rows)\n','')
    text=single_replace(text,'        old=list(csv.DictReader','        table(ROOT/(split+"_frames.csv"),rows)\n        old=list(csv.DictReader')
    compile(text,str(DRIVER),"exec")
    save(DRIVER,text.encode())
    stat_source=single_replace((STUDY/"qam_stats.py").read_text(),'assert read(ROOT/"wait.json")["actual_wait"]==0',
                               'assert read(ROOT/"recovery_wait.json")["actual_wait"]==0')
    compile(stat_source,str(STATS),"exec")
    save(STATS,stat_source.encode())
    record(ROOT/"io_recovery.json",{"registered_original":digest(parent),"recovery_script":digest(DRIVER),
        "offline_inputs":identities,"statistics_script":digest(STATS),"frozen_gain_sha256":digest(ROOT/"frozen_gains.json"),
        "failed_attempt":{"actual_wait":1,"pid":failed["pid"],"device_seconds":failed["device_seconds"],
            "position":"After all 9000 common500 image scores and before the first split output; no per-image scores survived the process exit.",
            "unretained_image_scores":9000,"calibration_transmissions_retained":768},
        "recovery":"Use the six verified Git blobs on PFS, reuse frozen calibration, write completed batches before historical cross-check. Channel, receiver, metrics, seeds and test cohort unchanged.",
        "test_only":True,"unchanged_total_role_limit_seconds":1800})


def run(gpu):
    from worker import admission
    assert not (ROOT/"recovery_start.json").exists()
    meta=read(ROOT/"io_recovery.json")
    assert digest(DRIVER)==meta["recovery_script"] and digest(ROOT/"frozen_gains.json")==meta["frozen_gain_sha256"]
    for name,h in meta["offline_inputs"].items():
        assert digest(MIRROR/name)==h["sha256"]
    samples=admission(gpu)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
             PYTHONPATH=os.pathsep.join([str(STUDY),include_sources()]),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
             MKL_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    cmd=[str(PY),"-B",str(DRIVER),"run"]
    with (TEMP/"logs/qam_recovery.log").open("xb") as log:
        start=time.time()
        p=subprocess.Popen(cmd,env=env,cwd=STUDY,stdout=log,stderr=subprocess.STDOUT)
        with open(f"/proc/{p.pid}/stat") as statfile:
            ticks=statfile.read().split(") ")[1].split()[19]
        record(ROOT/"recovery_start.json",dict(pid=p.pid,start_ticks=ticks,started_unix=start,command=cmd,
               admission=samples,physical_gpu=gpu,script_sha256=digest(DRIVER),protocol_sha256=digest(ROOT/"protocol.json")))
        remaining=1800-meta["failed_attempt"]["device_seconds"]
        try:
            rc=p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.terminate()
            try: rc=p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                rc=p.wait()
        record(ROOT/"recovery_wait.json",dict(pid=p.pid,actual_wait=rc,device_seconds=time.time()-start))
        if rc: raise RuntimeError(f"Offline QAM evaluation exited {rc}")
    assert digest(STATS)==meta["statistics_script"]
    subprocess.run([str(PY),"-B",str(STATS)],cwd=STUDY,env=dict(env,CUDA_VISIBLE_DEVICES=""),check=True)


if __name__=="__main__":
    if sys.argv[1]=="prepare":
        prepare()
    else:
        run(int(sys.argv[1]))
