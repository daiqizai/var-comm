"""Check the exported gate against per-source metrics and the registered bootstrap."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from verify_pfs_cache import filehash, require


def load_csv(path):
    with path.open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("evidence", type=Path)
    args = ap.parse_args()
    d = args.evidence
    read = lambda name: json.loads((d / name).read_bytes())
    for f, h in read("evidence_hashes.json").items():
        require(filehash(d/f) == h, ("exported evidence", f))
    plan = read("sp1_plan_identity.json")
    protocol = read("sp1_protocol.json")
    gate = read("decision.json")
    raw = read("actual_compute.json")
    roster = [r["source_id"] for r in read("source_manifest.json")]
    require(roster == plan["roster"] and len(set(roster)) == 50, "pilot ordering")
    candidates = load_csv(d / "all_policy_means.csv")
    require(Counter(r["family"] for r in candidates) == plan["policies"], "all registered policies")
    require(raw["physical_waves"] == plan["physical_waves"] and raw["receiver_observations"] == plan["receiver_observations"], "all radio observations")
    stored = {}
    fields = ("dino", "dists", "dreamsim", "psnr_db")
    grouped = defaultdict(list)
    for row in load_csv(d / "selected_source_means.csv"):
        grouped[row["pair"], row["family"]].append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda r:int(r["slot"]))
        require([r["source_id"] for r in rows] == roster, ("source means", key))
        stored[key] = np.array([[(float(row[m+"_weak"]) + float(row[m+"_strong"])) / 2
                                for m in fields] for row in rows])
    rules = protocol["bootstrap"]
    bank = np.random.default_rng(rules["seed"]).integers(0,50,(rules["replicates"],50),dtype=np.int16)
    mem = io.BytesIO()
    np.save(mem, bank, allow_pickle=False)
    require(hashlib.sha256(mem.getvalue()).hexdigest() == gate["bank_sha256"], "identical bootstrap indices")
    checks = 0
    for key, entry in gate["pairs"].items():
        for name, estimates in entry["comparisons"].items():
            diff = stored[key,"SP"] - stored[key,name]
            mu = diff.mean(0)
            ci = np.quantile(diff[bank].mean(1),(.025,.975),axis=0)
            for j, metric in enumerate(fields):
                want = [estimates[metric][v] for v in ("mean","lo","hi")]
                require(np.allclose(want,[mu[j],ci[0,j],ci[1,j]],atol=1.e-12,rtol=0), (key,name,metric))
                checks += 1
        # Independent family search directly over the delivered, full candidate means.
        weak, strong = key.split("_")
        selected = read("operating_points.json")[key]["0.5"]
        for label, family in (("pure_sp","sp"),("best_oma","oma"),("best_single","single")):
            pool = [r for r in candidates if r["family"] == family and (family != "sp" or r["grid"] == "quarter")]
            best = min(pool,key=lambda r:(-(float(r["dino_"+weak])+float(r["dino_"+strong]))/2,r["policy"]))
            require(best["policy"] == selected[label], (key, family, "equal-weight selection"))
        require(entry["passed"] == all(entry["tests"].values()), ("pair gate", key))
    require((gate["status"] == "GO") == all(p["passed"] for p in gate["pairs"].values()), "combined gate")
    print(json.dumps(dict(status="PASS",registered_sources=len(roster),candidate_rows=len(candidates),
        independent_mean_and_paired_intervals=checks,complete_radio_receptions=raw["receiver_observations"],
        decision=gate["status"]),indent=2))


if __name__ == "__main__":
    main()
