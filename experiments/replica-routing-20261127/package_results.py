"""Publishable execution ledger and file list; large arrays and process logs stay on PFS."""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

def filesafe(root):
    count=0
    for p in [root,*root.rglob("*")]:
        s=p.lstat()
        assert not stat.S_ISLNK(s.st_mode),p
        if stat.S_ISREG(s.st_mode):
            assert s.st_nlink==1,p
            count+=1
    return count

def put(p,data):
    with p.open("x",encoding="utf-8") as f: json.dump(data,f,indent=2,ensure_ascii=False)

def main(study,scratch):
    j=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    stages=("mechanism","development","formal","timing","secondary")
    events={}
    wall=0.
    receipts=study/"execution"
    receipts.mkdir(exist_ok=False)
    for name in stages:
        job=scratch/("job_"+name)
        meta=j(job/"registration.json")
        pid=j(job/"pid.json")
        wait=j(job/"wait.json")
        ids=j(study/name/"identity.json")
        done=j(study/name/"closed.json")
        assert wait["actual_wait"]==0
        current=Path(f'/proc/{pid["worker"]}/stat')
        assert not current.exists() or current.read_text().split()[21]!=pid["start_ticks"]
        interval=wait["end_unix"]-meta["start_unix"]
        wall+=interval
        events[name]=dict(registration=meta,process=pid,exit=wait,gpu_uuid=ids["gpu"]["uuid"],
            device_wall_seconds=interval,closed_science_seconds=done["elapsed_s"],
            peak_reserved_mib=done["peak_reserved_mib"],closed=done)
    put(receipts/"job_ledger.json",dict(roles=events,total_device_wall_hours=wall/3600,
        definition="Sum of elapsed wall time from role registration through actual child wait on one assigned A800 per role. Overlapping roles count separately. This is allocated time, not SM-active time.",
        environment="Frozen SwinJSCC, full FP32 (TF32 off), four physical CPU cores per process, three checkpoints. Development GPU2; other GPU roles GPU3. No training."))
    put(receipts/"source_snapshot.json",j(scratch/"snapshot/manifest.json"))
    put(receipts/"integrity.json",dict(physical_files_before_integrity_receipt=filesafe(study),
        snapshot_files=filesafe(scratch/"snapshot"),symlinks=0,multiply_linked_regular_files=0,
        receiver_only_inputs="Estimated activity, channel powers and LMMSE weights, denoiser output, stage index.",
        main_freeze_sha256=sha(study/"development/frozen.json"),secondary_freeze_sha256=sha(study/"nondegenerate_freeze.json"),
        weak_gate_replays=750,weak_lmmse_matrices_per_replay=4,
        figure_pdf_page_size_points={"replica_geometry":[522,212.4],"quality_cost":[522,237.6]},pages_per_figure=1,
        figure_review="Both PDFs rendered with Poppler at 150 dpi and visually inspected. Legends, stage labels, panel titles, axes, and error bars are visible. The revised three-page research draft has a separate print review.",
        fold="Nonoverlapping four-source clusters. Same source clusters across checkpoints and layouts."))

if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
