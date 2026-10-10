"""Bind copied report/figure artifacts without resolving scientific input paths.

prepare runs locally; verify runs on the actual destination; seal requires its
SHA-pinned successful receipt. All three stages use the standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = "N1024_REPORT_FIGURE_REMOTE_COPY_VERIFICATION_REQUEST_V1"
SCOPE = "local_authored_N1024_report_or_figure"
REMOTE_ROOT = "/home/liulu/projects/VAR_COMM"
BASE = "results/wcl_evidence_closure_20261009"
SCRIPTS = "experiments/wcl-evidence-closure-20261009/scripts"


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fresh(p):
    p = Path(p).resolve()
    if p.exists():
        raise ValueError("Fresh output directory required: " + str(p))
    p.mkdir(parents=True)
    return p


def classification(group, relative):
    """Only the locally authored report output layer is mapped.

    Copied historical science records and frozen policies are verified separately.
    No nested path appearing in an artifact is traversed or rewritten.
    """
    if group.startswith("T5_") or group in ("T3", "N1024_INDEX", "REPORT_FIGURE_SCRIPT"):
        return True, "locally authored report/figure artifact; contained source references remain unchanged"
    if group == "T2" and not relative.endswith("/expanded_whole_policy.json"):
        return True, "locally assembled read-only calibration audit report; not a scientific gate"
    if group == "T4_external" and "/original_" not in relative:
        return True, "locally assembled external timing report; original science references unchanged"
    if relative.endswith(("/frozen_policy.json", "/expanded_whole_policy.json", "/frozen_configurations.csv")):
        return False, "verification only: frozen scientific policy/configuration; prohibited as report path resolver"
    if "/original_" in relative:
        return False, "verification only: copied historical scientific outputs/receipts, not locally authored"
    return False, "verification only: remote-authored scientific/report export; no local-authorship assertion"


def validate_request(request):
    assert request["schema"] == SCHEMA
    assert request["scope"] == SCOPE
    assert request["remote_root"] == REMOTE_ROOT
    assert request["status"] == "AWAITING_ACTUAL_REMOTE_VERIFICATION"
    assert request["not_for_scientific_input_resolution"] is True
    assert request["rewrite_original_file_bytes"] is False
    assert request["original_index_row_count"] == 314
    entries = request["entries"]
    assert len(entries) == request["file_count"]
    assert len({r["original_local_path"] for r in entries}) == len(entries)
    assert len({r["remote_copy"]["path"] for r in entries}) == len(entries)
    for row in entries:
        rel = PurePosixPath(row["remote_relative_path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        assert str(rel).startswith((BASE + "/", SCRIPTS + "/"))
        assert row["remote_copy"]["path"] == REMOTE_ROOT + "/" + str(rel)
        assert row["remote_copy"]["sha256"] == row["sha256"]
        assert len(row["sha256"]) == 64 and int(row["sha256"], 16) >= 0
        assert row["bytes"] > 0
        eligible, reason = classification(row["group"], row["repo_relative_path"])
        assert eligible == row["map_eligible"] and reason == row["scope_reason"]
        assert row["scope"] == (SCOPE if eligible else "verification_only_not_a_path_mapping")
    return entries


def prepare(args):
    root = args.root.resolve()
    index = root / BASE / "N1024_delivery_index_v1"
    completion = read_json(index / "completion.json")
    assert completion["status"] == "N1024_DELIVERY_INDEX_COMPLETE_T6_PENDING"
    for name, binding in completion["outputs"].items():
        assert digest(index / name) == binding["sha256"]
    with (index / "delivery_files.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 314
    entries = []
    def add(path, group, remote_relative=None, expected=None):
        p = Path(path).resolve()
        assert p.is_relative_to(root)
        h = digest(p)
        if expected:
            assert h == expected, str(p)
        rel = p.relative_to(root).as_posix()
        remote_relative = remote_relative or rel
        eligible, reason = classification(group, rel)
        entries.append({"original_local_path": str(p), "repo_relative_path": rel,
                        "sha256": h, "bytes": p.stat().st_size, "group": group,
                        "remote_relative_path": remote_relative,
                        "remote_copy": {"path": REMOTE_ROOT + "/" + remote_relative, "sha256": h},
                        "map_eligible": eligible, "scope": SCOPE if eligible else "verification_only_not_a_path_mapping",
                        "scope_reason": reason})
    for row in rows:
        add(row["local_absolute_path"], row["group"], expected=row["sha256"])
    index_files = sorted(p for p in index.iterdir() if p.is_file())
    assert len(index_files) == 9
    for p in index_files:
        add(p, "N1024_INDEX")
    for name in ["plot_existing_t5.py", "plot_t5_completed_examples.py", "plot_t5_entropy_quality.py", "plot_t1_quality_tx_cost.py", "build_n1024_delivery_index.py", "n1024_report_path_map.py"]:
        remote_name = "plot_t5_completed_examples_delivery_279a229c.py" if name == "plot_t5_completed_examples.py" else name
        p = root / SCRIPTS / name
        if name == "plot_t5_completed_examples.py":
            assert digest(p) == "279a229c7c50b5d17c7f3215aad4f743fe943f3c16241a07da2016e52f74a5f4"
        add(p, "REPORT_FIGURE_SCRIPT", SCRIPTS + "/" + remote_name)
    entries.sort(key=lambda r: r["remote_relative_path"])
    request = {"schema": SCHEMA, "status": "AWAITING_ACTUAL_REMOTE_VERIFICATION", "scope": SCOPE,
               "not_for_scientific_input_resolution": True, "rewrite_original_file_bytes": False,
               "remote_root": REMOTE_ROOT, "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_index_completion": {"path": str(index / "completion.json"), "sha256": digest(index / "completion.json")},
               "source_index_csv": {"path": str(index / "delivery_files.csv"), "sha256": digest(index / "delivery_files.csv")},
               "original_index_row_count": len(rows), "extra_index_file_count": len(index_files),
               "extra_report_script_count": 6, "file_count": len(entries),
               "map_eligible_count": sum(r["map_eligible"] for r in entries),
               "verification_only_count": sum(not r["map_eligible"] for r in entries),
               "verifier_code_sha256": digest(__file__), "entries": entries}
    validate_request(request)
    out = fresh(args.out)
    write_json(out / "remote_verification_request.json", request)
    proposal = {"status": "UNSEALED_AWAITING_ACTUAL_REMOTE_RECEIPT", "scope": SCOPE,
                "not_for_scientific_input_resolution": True, "request_sha256": digest(out / "remote_verification_request.json"),
                "entries": [r for r in entries if r["map_eligible"]]}
    write_json(out / "path_mapping_proposal.UNSEALED.json", proposal)
    fields = ["group", "original_local_path", "repo_relative_path", "sha256", "bytes", "remote_relative_path", "map_eligible", "scope", "scope_reason"]
    with (out / "coverage.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(entries)
    h = digest(out / "remote_verification_request.json")
    readme = f"""# Unsealed local report/figure path map request

All 314 files from N1024_delivery_index_v1 are requested for actual remote SHA verification, plus its nine own files and six explicitly bound report/figure scripts: {len(entries)} files total. {request['map_eligible_count']} entries are eligible for the local report/figure output path map; {request['verification_only_count']} are verification-only. The copied frozen policy, remote-authored scientific exports and historical scientific receipts are never path-resolved through this map. No scientific gate, checkpoint, model, input image, registered source record or .research input receives a mapping. Source references inside each artifact are preserved byte for byte.

The examples v2 local script has remote name plot_t5_completed_examples_delivery_279a229c.py. The remote historical same-name script is not replaced or silently mapped. Other remote paths are the exact copied delivery paths explicitly listed by the source index, not guesses about old scientific inputs.

Status is UNSEALED. No successful remote verification or copied-file availability is claimed before the remote process returns. A missing or mismatched file yields a failed receipt, and seal refuses it. Existing scientific/report outputs are never edited; the verifier reads only the enumerated files and writes only a new receipt.

Upload the request and n1024_report_path_map.py through the root-owned connection. Root runs:

```text
python experiments/wcl-evidence-closure-20261009/scripts/n1024_report_path_map.py verify --request <uploaded_request.json> --request-sha {h} --receipt <new_remote_receipt.json>
```

Download that actual receipt, independently obtain its remote SHA, and run locally into a new directory:

```text
python experiments/wcl-evidence-closure-20261009/scripts/n1024_report_path_map.py seal --request <this_request.json> --request-sha {h} --receipt <downloaded_receipt.json> --receipt-sha <actual_remote_receipt_SHA> --out <new_sealed_mapping_directory>
```

This map is exclusively local-authored N1024 report/figure navigation and restore provenance. It is not an authority to redirect any scientific dependency, relax any hash check, reinterpret a science gate, or declare T6 complete. No raw C:\\ reference embedded in a report is edited or replaced. No SSH, model, channel, bootstrap or timing calls are made by this workflow.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"out": str(out), "request_sha256": h, "files": len(entries), "eligible": request["map_eligible_count"], "verification_only": request["verification_only_count"], "status": "UNSEALED"}, indent=2))


def verify(args):
    assert digest(args.request) == args.request_sha
    request = read_json(args.request)
    entries = validate_request(request)
    assert digest(__file__) == request["verifier_code_sha256"]
    assert platform.system() == "Linux", "Run verification on the actual Linux destination"
    remote_root = Path(REMOTE_ROOT).resolve(strict=True)
    receipt_path = args.receipt.resolve()
    assert not receipt_path.exists(), "Do not overwrite an actual receipt"
    results = []
    for row in entries:
        p = Path(row["remote_copy"]["path"])
        result = {"original_local_path": row["original_local_path"], "remote_path": str(p), "expected_sha256": row["sha256"], "expected_bytes": row["bytes"], "map_eligible": row["map_eligible"]}
        try:
            assert p.resolve(strict=True).is_relative_to(remote_root), "Outside destination repo"
            assert p.is_file(), "Not a file"
            actual = digest(p)
            result.update({"actual_sha256": actual, "actual_bytes": p.stat().st_size, "bytes_actually_read": True, "status": "MATCH" if actual == row["sha256"] and p.stat().st_size == row["bytes"] else "MISMATCH"})
        except Exception as ex:
            result.update({"status": "MISSING_OR_UNREADABLE", "bytes_actually_read": False, "error": type(ex).__name__ + ": " + str(ex)})
        results.append(result)
    passed = all(r["status"] == "MATCH" for r in results)
    receipt = {"schema": "N1024_REPORT_FIGURE_REMOTE_COPY_VERIFICATION_RECEIPT_V1", "status": "PASS" if passed else "FAIL", "actual_remote_files_read": True,
               "scope": SCOPE, "not_for_scientific_input_resolution": True, "request_sha256": args.request_sha,
               "verifier_code_sha256": digest(__file__), "remote_root": str(remote_root), "hostname": platform.node(),
               "platform": platform.platform(), "python_executable": sys.executable, "verified_utc": datetime.now(timezone.utc).isoformat(),
               "file_count": len(results), "matched_count": sum(r["status"] == "MATCH" for r in results), "results": results}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "matched_count": receipt["matched_count"], "file_count": len(results), "receipt": str(receipt_path), "receipt_sha256": digest(receipt_path)}, indent=2))
    if not passed:
        raise SystemExit(2)


def seal(args):
    assert digest(args.request) == args.request_sha
    assert digest(args.receipt) == args.receipt_sha
    request = read_json(args.request)
    entries = validate_request(request)
    receipt = read_json(args.receipt)
    assert receipt["schema"] == "N1024_REPORT_FIGURE_REMOTE_COPY_VERIFICATION_RECEIPT_V1"
    assert receipt["status"] == "PASS" and receipt["actual_remote_files_read"] is True
    assert receipt["request_sha256"] == args.request_sha
    assert receipt["verifier_code_sha256"] == request["verifier_code_sha256"]
    assert receipt["scope"] == SCOPE and receipt["not_for_scientific_input_resolution"] is True
    assert receipt["file_count"] == receipt["matched_count"] == len(entries)
    actual = {r["remote_path"]: r for r in receipt["results"]}
    assert len(actual) == len(entries)
    for row in entries:
        r = actual[row["remote_copy"]["path"]]
        assert r["status"] == "MATCH" and r["bytes_actually_read"] is True
        assert r["actual_sha256"] == r["expected_sha256"] == row["sha256"]
        assert r["actual_bytes"] == r["expected_bytes"] == row["bytes"]
        assert r["original_local_path"] == row["original_local_path"]
        assert r["map_eligible"] == row["map_eligible"]
    out = fresh(args.out)
    shutil.copyfile(args.request, out / "remote_verification_request.json")
    shutil.copyfile(args.receipt, out / "actual_remote_verification_receipt.json")
    mapping = {"schema": "N1024_LOCAL_AUTHORED_REPORT_FIGURE_PATH_MAP_V1", "status": "SEALED_AFTER_ACTUAL_REMOTE_SHA_VERIFICATION", "scope": SCOPE,
               "not_for_scientific_input_resolution": True, "rewrite_original_file_bytes": False,
               "request_sha256": args.request_sha, "actual_remote_receipt_sha256": args.receipt_sha,
               "verified_file_count": len(entries), "map_entry_count": request["map_eligible_count"],
               "entries": [r for r in entries if r["map_eligible"]]}
    write_json(out / "report_figure_path_map.json", mapping)
    write_json(out / "completion.json", {"status": mapping["status"], "T6_complete": False, "scope": SCOPE,
               "new_scientific_calls": 0, "request_sha256": args.request_sha, "receipt_sha256": args.receipt_sha,
               "outputs": {p.name: digest(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({"out": str(out), "status": mapping["status"], "map_entry_count": mapping["map_entry_count"], "completion_sha256": digest(out / "completion.json")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--root", type=Path, default=Path(".")); p.add_argument("--out", type=Path, required=True)
    p = sp.add_parser("verify")
    p.add_argument("--request", type=Path, required=True); p.add_argument("--request-sha", required=True); p.add_argument("--receipt", type=Path, required=True)
    p = sp.add_parser("seal")
    p.add_argument("--request", type=Path, required=True); p.add_argument("--request-sha", required=True); p.add_argument("--receipt", type=Path, required=True); p.add_argument("--receipt-sha", required=True); p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"prepare": prepare, "verify": verify, "seal": seal}[args.mode](args)


if __name__ == "__main__":
    main()
