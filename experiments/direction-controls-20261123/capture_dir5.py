"""Capture producer evidence as regular immutable files; detect changes during reading."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from study import REPO, STUDY, TEMP, dump_once, git, regular_path, save_once

PROJ = "experiments/dir5-otfs-genvar-20261122"
RESULT = "results/dir5_otfs_genvar_20261122"
LOGDIR = TEMP.parent / "dir5_otfs_genvar_v1/logs"


def hashed_file(p):
    h = hashlib.sha256()
    with p.open("rb") as fd:
        for block in iter(lambda: fd.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main(label):
    assert label.replace("_","").isalnum()
    dest = TEMP / "producer_snapshots" / label
    dest.mkdir(parents=True, exist_ok=False)
    files = [p for folder in (REPO/PROJ, REPO/RESULT) for p in folder.rglob("*")
             if p.is_file() and p.suffix in {".py",".sh",".json",".csv",".md"}]
    files += [p for p in LOGDIR.glob("*.log") if p.is_file() and p.stat().st_size <= 128*1024]
    entries = {}
    head = git("rev-parse", "HEAD").decode().strip()
    for p in sorted(files):
        regular_path(p)
        name = str(p.relative_to(REPO)) if p.is_relative_to(REPO) else "execution_logs/"+p.name
        data = p.read_bytes()
        save_once(dest/name, data)
        entries[name] = hashlib.sha256(data).hexdigest()
    cp = {}
    for p in sorted((dest/RESULT/"eval").glob("*.json")):
        for k,v in json.loads(p.read_text())["checkpoints"].items():
            loc = Path(v["path"])
            regular_path(loc)
            if str(loc) not in cp:
                cp[str(loc)] = {"observed_now":hashed_file(loc), "expected":{}}
            cp[str(loc)]["expected"][p.name]=v["sha256"]
    receipt = {"head_at_capture":head, "observed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
               "label":label, "files":entries, "checkpoints":cp,
               "checkpoints_matching_all":sum(all(h==v["observed_now"] for h in v["expected"].values()) for v in cp.values())}
    dump_once(dest/"manifest.json",receipt)
    dump_once(STUDY/"dir5"/(label+"_identity.json"),receipt)
    print("Snapshot",label,"source/result/log files",len(entries),"matching current ckpts",receipt["checkpoints_matching_all"],"/",len(cp))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("label")
    main(parser.parse_args().label)
