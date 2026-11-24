"""Paired contrasts on the same 125 quadruples and resampling bank as the parent."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_replicas import estimate, write_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for a in ("events", "gate_events", "parent_analysis", "wait", "output"):
        p.add_argument(a, type=Path)
    a = p.parse_args()
    assert json.loads(a.wait.read_text())["actual_wait"] == 0
    assert not list(a.gate_events.glob("*.partial"))
    gate_meta = json.loads((a.gate_events / "compute.json").read_text())
    assert gate_meta["weak_user_scores"] == 750
    check = json.loads((a.parent_analysis / "data_checks.json").read_text())
    bank = np.random.default_rng(2026112401).integers(125, size=(10000,125), dtype=np.int32)
    assert hashlib.sha256(bank.astype("<i4").tobytes()).hexdigest() == check["bootstrap"]["common500"]["int32_little_endian_sha256"]
    metric_keys = [("weak_psnr_db","psnr_db"), ("weak_dists","dists"), ("interference_nmse_db_r3","interference_nmse_db_r3")]
    rows = []
    for cond in ("nearfar_14db", "nearfar_18db_ood"):
        measured = np.full((3,125,3,3),np.nan)
        for i, seed in enumerate((1001,2002,3003)):
            name = f"common500_{cond}_{seed}.csv"
            with (a.events / name).open(newline="") as f:
                old = {(int(r["group"]),r["arm"]):r for r in csv.DictReader(f) if r["slot"]=="3"}
            with (a.gate_events / name).open(newline="") as f:
                for r in csv.DictReader(f):
                    g = int(r["group"])
                    assert r["condition"] == cond and int(r["seed"]) == seed and np.isnan(measured[i,g]).all()
                    native, rgb = old[g,"latent"], old[g,"rgb_reencode"]
                    assert native["source"] == r["source"] == rgb["source"]
                    for m,(new_key,old_key) in enumerate(metric_keys):
                        measured[i,g,:,m] = [native[old_key], r[new_key], rgb[old_key]]
        assert np.isfinite(measured).all()
        for m,(key,_) in enumerate(metric_keys):
            group = measured[:,:,:,m].mean(axis=0)
            for label,delta in (("rgb_fixed_minus_latent",group[:,1]-group[:,0]),
                                ("rgb_own_minus_rgb_fixed",group[:,2]-group[:,1])):
                rows.append(dict(split="common500",condition=cond,groups=125,seed_realizations=3,
                                 metric=key,comparison=label,**estimate(delta,bank)))
    write_rows(a.output,rows)


if __name__ == "__main__":
    main()
