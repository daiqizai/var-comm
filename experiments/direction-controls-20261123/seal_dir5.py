"""Wait for both producer evaluations to return before identifying the final 5B weights."""
import csv
import hashlib
import json
from pathlib import Path
import time
from audit_dir5 import audit
from capture_dir5 import LOGDIR, RESULT, main as capture
from study import REPO, STUDY, TEMP, dump_once


def active():
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try: a=(p/"cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if any(v.endswith(b"/evaluate_dir5_otfs_genvar.py") for v in a[:4]) and b"--skip-dir5a" in a: return True
    return False


def main():
    for split in ("common500","kodak24"):
        p=LOGDIR/f"eval_{split}_dir5b_v2.log"
        while not p.is_file() or f"[Eval-Dir5:{split}] ALL DONE" not in p.read_text(errors="replace"):
            if not active():
                raise RuntimeError(f"Producer exited before writing completion for {split}")
            time.sleep(20)
    while active(): time.sleep(5)
    for split in ("common500","kodak24"):
        ev=json.loads((REPO/RESULT/f"eval/summary_dir5_{split}.json").read_text())["checkpoints"]
        old=json.loads((TEMP/"producer_snapshots/observed_v1"/RESULT/f"eval/summary_dir5_{split}.json").read_text())["checkpoints"]
        for tag in ("genholo_var_cbr128","genholo_var_cbr256"):
            h=json.loads((REPO/RESULT/f"train/summary_{tag}.json").read_text())["best_ckpt_sha256"]
            assert ev[tag]["sha256"]==h!=old[tag]["sha256"]
    capture("published_v2")
    audit("published_v2")
    root=STUDY/"dir5"
    result={}
    for split in ("common500","kodak24"):
        for name in ("dir5a_otfs","dir5b_genvar"):
            files=[root/ver/f"{name}_{split}.csv" for ver in ("observed_v1","published_v2")]
            rows=[]
            for p in files:
                with p.open() as stream: rows.append({(v["condition"],v["model"]):v for v in csv.DictReader(stream)})
            assert rows[0].keys()==rows[1].keys()
            diff=0.; bycondition={}
            for k in rows[0]:
                delta={m:float(rows[1][k][m+"_mean"])-float(rows[0][k][m+"_mean"])
                       for m in ("psnr_db","ms_ssim","dists","lpips_alex")}
                if k[1]=="GenHolo_VAR_JSCC": bycondition[k[0]]=delta
                else: diff=max(diff,max(abs(v) for v in delta.values()))
            result[name+"_"+split]={"unchanged_reference_max_abs_mean_diff":diff,
                   "raw_csv_equal":files[0].read_bytes()==files[1].read_bytes(), "v2_minus_v1_genholo_descriptive":bycondition}
    dump_once(root/"release_comparison.json",result)
    print("Producer v2 sealed, checkpoint and cross-version checks complete",flush=True)


if __name__=="__main__":main()
