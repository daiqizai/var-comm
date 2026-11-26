"""Freeze the best genuinely selective validation ranker using only development scores."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main(study):
    cfg=json.loads((study/"protocol.json").read_text())
    dev=study/"development"
    with (dev/"candidates.csv").open(newline="") as f:
        candidates=list(csv.DictReader(f))
    viable=[v for v in candidates if any(k in (1,2) for k in cfg["schedules"][v["schedule"]])]
    adaptive=max((v for v in viable if v["rule"] in cfg["adaptive_rules"]),key=lambda v:float(v["weak_psnr"]))
    fixed=max((v for v in viable if v["schedule"]==adaptive["schedule"] and v["rule"] in cfg["fixed_rules"]),
              key=lambda v:float(v["weak_psnr"]))
    values=dict(protocol=sha(study/"nondegenerate_protocol.json"),parent=sha(study/"protocol.json"),
        freeze=sha(dev/"frozen.json"),candidates=sha(dev/"candidates.csv"),adaptive=adaptive,fixed=fixed)
    dest=study/"nondegenerate_freeze.json"
    if dest.exists():
        old=json.loads(dest.read_text())
        assert all(old[k]==v for k,v in values.items()),"Existing decision disagrees with validation rule"
        return
    values.update(issued_unix=time.time(),completed_formal_conditions_at_freeze=len(list((study/"formal").glob("*.csv"))),
                  basis="Development-only. Formal quality files not opened for this decision.")
    with dest.open("x") as f: json.dump(values,f,indent=2)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    main(p.parse_args().study)
