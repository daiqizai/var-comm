"""Read-only checks of already completed CSVs and process receipts; no inference/bootstrap."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ("psnr_db", "lpips_alex", "dists", "dreamsim")
BUDGETS = (512, 768, 1024, 1536, 2048, 3072, 4096, 8192)


def rows(name):
    with (ROOT / name).open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    frame, means, paired, capacity = map(rows, (
        "frame_metrics.csv", "summary.csv", "paired_deltas.csv", "capacity.csv"))
    snapshot = json.loads((ROOT / "completion_snapshot.json").read_text(encoding="utf8"))
    bank = json.loads((ROOT / "statistics_identity.json").read_text(encoding="utf8"))
    counts = json.loads((ROOT / "actual_compute_and_reuse.json").read_text(encoding="utf8"))
    sources = bank["sources"]
    assert len(sources) == len(set(sources)) == 500
    assert bank["resamples"] == 10000
    assert (len(frame), len(means), len(paired), len(capacity)) == (8000, 64, 32, 16)
    keyed = {(r["source"], int(r["budget_bits"]), r["method"]): r for r in frame}
    assert len(keyed) == len(frame)
    groups = {}
    for b in BUDGETS:
        for method in ("original", "new"):
            g = [keyed[s, b, method] for s in sources]
            assert all(int(r["size"]) == 256 and int(r["null"]) == 0 for r in g)
            assert all(int(r["packet_bits"]) <= b for r in g)
            assert all(math.isfinite(float(r[m])) for r in g for m in METRICS)
            groups[b, method] = g
        assert all(keyed[s, b, "new"]["reference_sha256"] ==
                   keyed[s, b, "original"]["reference_sha256"] for s in sources)
    for r in means:
        g = groups[int(r["budget_bits"]), r["method"]]
        actual = math.fsum(float(x[r["metric"]]) for x in g) / 500
        assert abs(actual - float(r["mean"])) < 1e-10
        assert float(r["ci_low"]) <= float(r["mean"]) <= float(r["ci_high"])
        assert int(r["sources"]) == 500
    for r in paired:
        b, m = int(r["budget_bits"]), r["metric"]
        actual = math.fsum(float(keyed[s, b, "new"][m]) -
                           float(keyed[s, b, "original"][m]) for s in sources) / 500
        assert abs(actual - float(r["mean"])) < 1e-10
        assert r["contrast"] == "new minus original" and int(r["sources"]) == 500
        assert float(r["ci_low"]) <= float(r["mean"]) <= float(r["ci_high"])
    for r in capacity:
        g = groups[int(r["budget_bits"]), r["method"]]
        total_tokens = 1360 if r["method"] == "new" else 336
        assert int(r["null_count"]) == int(r["hard_budget_violations"]) == 0
        assert int(r["full_count"]) == sum(int(x["received_tokens"]) == total_tokens for x in g)
        for col, raw in (("mean_packet_bits", "packet_bits"), ("mean_received_tokens", "received_tokens")):
            assert abs(float(r[col]) - math.fsum(float(x[raw]) for x in g) / 500) < 1e-10
    roles = snapshot["roles"]
    assert len(roles) == 25
    receipts = {role: value["actual_wait.json"] for role, value in roles.items()}
    assert all(r["returncode"] == 0 and r["child_gone"] for r in receipts.values())
    outer = snapshot["snapshot"]["ledger"]["eval256_evaluation_v2"]
    assert outer["actual_wait"]["returncode"] == 0
    assert not outer["owner_live"] and not outer["child_live"]
    assert all(v["all_native_finite"] and v["budgets_valid"] and
               v["baseline256_source_native_identity_checked"] for v in snapshot["shard_audits"].values())
    assert sum(v["received_prefixes_exact"] for v in snapshot["shard_audits"].values()) == 4000
    assert len(snapshot["metric_reuse_checks"]) == 8
    assert all(v["passed"] for v in snapshot["metric_reuse_checks"].values())
    totals = Counter()
    for v in counts.values():
        totals.update(v)
    for k in ("new_encoder", "new_rx", "new_metric_pairs", "baseline_cache_logical_reuse"):
        assert totals[k] == 4000, (k, totals[k])
    assert totals["baseline_verification_metric_pairs"] == 8
    assert totals["baseline_encoder"] == totals["baseline_rx"] == totals["baseline_metric_pairs"] == 0
    result = {
        "passed": True, "resolution": 256, "sources": 500, "budgets_bits": BUDGETS,
        "logical_positions": 8000, "mean_intervals": 64, "paired_intervals": 32,
        "same_source_reference_hashes": True, "complete_denominators": True,
        "mean_and_direct_paired_point_estimates_reconciled": True,
        "bootstrap_recomputed_by_this_audit": False,
        "bootstrap_identity": {k: v for k, v in bank.items() if k != "sources"},
        "all_25_leaf_roles_exit_zero": True, "controller_exit_zero_and_gone": True,
        "independent_received_prefixes_exact": 4000,
        "null_positions": 0, "hard_budget_violations": 0,
        "compute_counts": dict(totals),
        "evaluation_gpu_device_hours": sum(r["charged_gpu_device_hours"] for r in receipts.values()),
        "evaluation_leaf_cpu_process_hours": sum(r["cpu_process_seconds"] for r in receipts.values()) / 3600,
        "resource_accounting": "25 leaf receipts only; controller child-inclusive CPU time is not added again",
        "controller_wall_seconds": outer["actual_wait"]["elapsed_seconds"],
        "source_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in ROOT.iterdir() if p.suffix == ".csv"},
        "new_scientific_calls_by_this_audit": 0,
    }
    (ROOT / "data_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    (ROOT / "resource_ledger_final.json").write_text(json.dumps({
        "leaf_receipts": receipts, "controller_receipt_not_added_to_leaf_cpu": outer["actual_wait"],
        "gpu_device_hours": result["evaluation_gpu_device_hours"],
        "cpu_process_hours": result["evaluation_leaf_cpu_process_hours"],
    }, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
