"""Publish only the completed, separately registered metric extension.

No push is forced, no dirty file is discarded, and no running original study is
modified. Large CSVs are represented by lossless, independently usable shards.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs/UNIFIED-METRICS-20261002"
RESULT = ROOT / "results/unified_metrics_20261002"
PARENT = ROOT / "outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002"
LIMIT = 8_000_000
SOURCE_SUFFIXES = {".py", ".md"}
RESULT_SUFFIXES = {".csv", ".json", ".md", ".svg", ".txt"}
NEEDED_METRICS = {"clip_image_cosine", "dinov2_vitl14_cosine", "dists", "dreamsim", "ms_ssim", "resnet50_top1_label", "resnet50_top1_source_prediction"}
QUALIFIED_MODELS = ("clip", "dists", "resnet50", "dreamsim", "ms_ssim", "dinov2_vitl14")
REQUIRED_FRAME_COLUMNS = NEEDED_METRICS | {"dino_cosine"}
OLD_MARKERS = ("experiments/scale-causal-partial-residual-20261002/", "outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002/")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8_000_000), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def bindings(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(paths)}


def verify_bindings(mapping, label, *, required=True):
    if not isinstance(mapping, dict) or (required and not mapping):
        raise RuntimeError(f"Missing {label} bindings")
    for path, digest in mapping.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or not Path(path).is_file() or sha(path) != digest:
            raise RuntimeError(f"Changed {label}: {path}")


def command(args, *, log=None, cpu=False, capture=False):
    env = os.environ.copy()
    if cpu:
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    completed = subprocess.run([str(a) for a in args], cwd=ROOT, env=env,
        stdout=subprocess.PIPE if capture else log, stderr=subprocess.PIPE if capture else subprocess.STDOUT)
    if log:
        log.flush()
    if completed.returncode:
        tail = completed.stderr.decode(errors="replace")[-2000:] if capture else "see publication log"
        raise RuntimeError(f"Command failed ({completed.returncode}): {args!r}: {tail}")
    return completed.stdout if capture else b""


def git(*args):
    return command(["git", *args], capture=True).decode("utf-8").strip()


def git_names(*args):
    return set(filter(None, command(["git", *args, "-z"], capture=True).decode("utf-8").split("\0")))


def process_snapshot(proc_root=Path("/proc")):
    """Read Linux process identities without signalling or changing any process."""
    if not proc_root.is_dir():
        raise RuntimeError("Linux /proc is required to verify old process exit")
    result = {}
    for folder in proc_root.iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            raw = (folder / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            result[int(folder.name)] = {"state": fields[0], "start_ticks": fields[19],
                "command": (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")}
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Unreadable other users are not ours; the known owner PID is checked below.
            result[int(folder.name)] = {"unreadable": True}
    return result


def gpu_pids():
    output = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture=True).decode()
    result = set()
    for line in output.splitlines():
        line = line.strip()
        if line.isdecimal():
            result.add(int(line))
        elif line and line != "No running processes found":
            raise RuntimeError("Unexpected nvidia-smi process output")
    return result


def verify_push_record(record):
    if (record.get("status") != "PUSHED" or record.get("checks") != "PASS"
            or not re.fullmatch(r"[0-9a-f]{40}", str(record.get("commit", "")))
            or record.get("commit") != record.get("remote_commit")):
        raise RuntimeError("A verified normal-push record is required")


def verify_parent_gate(parent=PARENT, *, processes=None, compute_pids=None):
    paths = [parent / "completion.json", parent / "supervisor_status.json", parent / "supervisor_launch.json"]
    complete, supervisor, launch = map(read, paths)
    if (complete.get("status") != "AUTHORIZED_TWO_METHODS_COMPLETE" or complete.get("synthetic") is not False
            or complete.get("training_updates") != 0 or complete.get("stop") is not True):
        raise RuntimeError("The original two-method study is not fully delivered")
    verify_push_record(complete.get("publication", {}))
    if supervisor.get("status") != "COMPLETE" or supervisor.get("stopped_after_authorized_scope") is not True:
        raise RuntimeError("The original supervisor has not completed its scope")
    if int(launch["pid"]) != int(supervisor["pid"]):
        raise RuntimeError("Original supervisor identity differs from its launch")
    processes = process_snapshot() if processes is None else processes
    compute_pids = gpu_pids() if compute_pids is None else compute_pids
    own_pid = int(launch["pid"])
    identity = processes.get(own_pid)
    if identity and identity.get("unreadable"):
        raise RuntimeError("Cannot verify exit of the known original supervisor")
    if identity and identity.get("start_ticks") == str(launch["start_ticks"]) and identity.get("state") != "Z":
        raise RuntimeError("Original supervisor still exists after its completion receipt")
    old_live = {pid for pid, item in processes.items() if item.get("state") != "Z" and any(marker in item.get("command", "").replace("\\", "/") for marker in OLD_MARKERS)}
    if old_live:
        raise RuntimeError(f"Original experiment processes remain alive: {sorted(old_live)}")
    # A zombie cannot own live CUDA allocations. The original PID in the GPU
    # process list is unsafe even if /proc inspection raced with process exit.
    if own_pid in compute_pids and (identity is None or identity.get("start_ticks") == str(launch["start_ticks"])):
        raise RuntimeError("Original supervisor still owns a GPU context")
    return {"status": "PARENT_DELIVERED_AND_EXITED", "receipts": bindings(paths),
        "parent_commit": complete["publication"]["commit"], "verified_utc": timestamp(),
        "original_live_processes": [], "gpu_process_scan_passed": True}


def verify_results(out=OUT, result=RESULT, source=HERE):
    analysis_path = result / "metrics_analysis_completion.json"
    scoring_path = out / "scoring_completion.json"
    qualification_path = out / "models_qualification.json"
    analysis, scoring, qualification = map(read, (analysis_path, scoring_path, qualification_path))
    for name, rec in (("analysis", analysis), ("scoring", scoring)):
        if rec.get("status") != "COMPLETE" or rec.get("synthetic") is not False or rec.get("training_updates") != 0 or rec.get("policy_selection_updates") != 0:
            raise RuntimeError(f"Completed real zero-training/zero-selection {name} required")
        verify_bindings(rec.get("inputs"), name + " inputs")
        verify_bindings(rec.get("outputs"), name + " outputs")
    if (scoring.get("parity_passed") is not True or scoring.get("frames") != analysis.get("frames")
            or not isinstance(scoring.get("frames"), int) or scoring["frames"] <= 0):
        raise RuntimeError("Complete frame-wise replay parity has not passed")
    if (analysis.get("sources") != 100 or analysis.get("bootstrap_replicates") != 10000
            or analysis.get("statistical_unit") != "source_image"
            or not NEEDED_METRICS.issubset(analysis.get("evaluated_metrics", []))):
        raise RuntimeError("The registered source-paired metric analysis is incomplete")
    frame_path = result / "metrics_per_frame.csv"
    if (str(frame_path.resolve()) not in scoring["outputs"]
            or scoring["outputs"][str(frame_path.resolve())] != sha(frame_path)):
        raise RuntimeError("Scoring did not bind its original complete per-frame metric table")
    with frame_path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    missing = REQUIRED_FRAME_COLUMNS - set(header)
    if missing:
        raise RuntimeError("Required scored metric columns missing: " + ", ".join(sorted(missing)))
    if (qualification.get("status") != "REAL_MODEL_WEIGHTS_QUALIFICATION_PASS"
            or qualification.get("real_model_weights") is not True
            or qualification.get("scientific_result") is not False
            or qualification.get("synthetic_images") is not True):
        raise RuntimeError("All real metric weights must pass their separate qualification")
    verify_bindings(qualification.get("source_bindings"), "qualified evaluator source")
    if not {str((source / name).resolve()) for name in ("metric_models.py", "qualify_models.py")}.issubset(qualification["source_bindings"]):
        raise RuntimeError("Model qualification is not bound to the active evaluator and qualifier")
    for name in QUALIFIED_MODELS:
        if qualification.get("metadata", {}).get("metrics", {}).get(name, {}).get("status") != "READY":
            raise RuntimeError(f"Metric qualification did not load {name}")
    manifest_path = out / "modelmanifest.json"
    digest = sha(manifest_path)
    if qualification.get("modelmanifest_sha256") != digest or scoring.get("modelmanifest_sha256") != digest:
        raise RuntimeError("Scoring/qualification used a different metric asset manifest")
    verify_bindings(scoring.get("source_bindings"), "scoring source")
    registration_path = result / "metrics_registration.json"
    registration = read(registration_path)
    if registration.get("modelmanifest_sha256") != digest or registration.get("original_pipeline_complete") is not True:
        raise RuntimeError("Scoring registration does not bind the completed parent and metric manifest")
    if str(registration_path.resolve()) not in analysis["inputs"] or analysis["inputs"][str(registration_path.resolve())] != sha(registration_path):
        raise RuntimeError("Analysis did not bind the scoring registration")
    batch_path = verify_metric_batch(out, result, source, scoring, registration, digest)
    return {"analysis": analysis, "scoring": scoring, "qualification": qualification,
        "verified_inputs": bindings((analysis_path, scoring_path, qualification_path, manifest_path, registration_path, batch_path))}


def verify_metric_batch(out, result, source, scoring, registration, manifest_digest):
    """Publication must preserve the measured batch and its exact evidence."""
    from batch_speed import validate_receipt
    path = out / "metric_batch_qualification.json"
    copy = result / path.name
    receipt = read(path)
    validate_receipt(receipt, receipt.get("binding", {}))
    verify_bindings(receipt.get("source_bindings"), "batch qualification source")
    required = {str((source / name).resolve()) for name in ("batch_speed.py", "metric_models.py")}
    if not required.issubset(receipt["source_bindings"]):
        raise RuntimeError("Batch qualification did not bind its active implementation")
    digest = sha(path)
    if (receipt.get("modelmanifest_sha256") != manifest_digest
            or receipt.get("training_updates") != 0 or receipt.get("policy_selection_updates") != 0
            or sha(copy) != digest or scoring["inputs"].get(str(path.resolve())) != digest
            or scoring["outputs"].get(str(copy.resolve())) != digest):
        raise RuntimeError("Scoring did not preserve its exact metric batch evidence")
    for record in (scoring, registration):
        if (record.get("metric_batch_qualification_sha256") != digest
                or record.get("metric_batch_size") != receipt["chosen_batch_size"]
                or record.get("metric_qualified_batch_sizes") != receipt["qualified_batch_sizes"]
                or record.get("metric_evaluator_identity") != receipt["metric_evaluator_identity"]):
            raise RuntimeError("Scoring batch differs from the qualified metric batch")
    if registration.get("metric_batch_qualification_path") != str(path.resolve()):
        raise RuntimeError("Registration did not identify the metric batch evidence")
    return path


class _CapturedLines:
    def __init__(self, binary):
        self.binary, self.chunks = binary, []

    def __iter__(self):
        return self

    def __next__(self):
        data = self.binary.readline()
        if not data:
            raise StopIteration
        self.chunks.append(data)
        return data.decode("utf-8")


def csv_records(path):
    """Yield exact bytes per logical CSV record, including embedded newlines."""
    with Path(path).open("rb") as file:
        lines = _CapturedLines(file)
        reader = csv.reader(lines, strict=True)
        for fields in reader:
            raw = b"".join(lines.chunks)
            lines.chunks.clear()
            yield fields, raw
        if lines.chunks:
            raise RuntimeError("Trailing bytes outside a logical CSV record")


def safe_relative(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise RuntimeError(f"Archive path escapes result directory: {relative}")
    return candidate


def archive_tables(result=RESULT, limit=LIMIT):
    table_dir = result / "table_shards"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "exact_original_bytes": True, "part_byte_limit": limit, "tables": {}}
    for path in sorted(result.rglob("*.csv")):
        if table_dir in path.parents or path.stat().st_size <= limit:
            continue
        rel = path.relative_to(result).as_posix()
        records = iter(csv_records(path))
        try:
            header_fields, header = next(records)
        except StopIteration:
            raise RuntimeError(f"Cannot archive empty CSV: {rel}")
        if not header_fields or len(header) >= limit:
            raise RuntimeError(f"Invalid or oversized CSV header: {rel}")
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)[:80] + "_" + hashlib.sha256(rel.encode()).hexdigest()[:12]
        parts, buffer, rows, total = [], bytearray(header), 0, 0

        def flush():
            nonlocal buffer, rows, total
            target = table_dir / f"{stem}.part{len(parts):04d}.csv"
            data = bytes(buffer)
            if target.exists() and target.read_bytes() != data:
                raise RuntimeError("Existing archived CSV part differs; preserve the previous publication")
            if not target.exists():
                target.write_bytes(data)
            parts.append({"path": target.relative_to(result).as_posix(), "bytes": len(data), "rows": rows, "sha256": sha(target)})
            total += rows
            buffer, rows = bytearray(header), 0

        for fields, raw in records:
            if len(fields) != len(header_fields):
                raise RuntimeError(f"CSV column count differs: {rel}")
            if len(header) + len(raw) > limit:
                raise RuntimeError(f"A single CSV record exceeds the shard limit: {rel}")
            if len(buffer) + len(raw) > limit and rows:
                flush()
            buffer.extend(raw)
            rows += 1
        if rows:
            flush()
        entry = {"sha256": sha(path), "bytes": path.stat().st_size, "rows": total, "header_bytes": len(header),
            "header_sha256": hashlib.sha256(header).hexdigest(), "parts": parts, "index": f"table_shards/{stem}.index.md"}
        if not parts:
            raise RuntimeError(f"Oversized CSV has no data records: {rel}")
        verify_archive_entry(result, entry)
        index = result / entry["index"]
        index.write_text(f"# {rel}\n\n{total} logical CSV records; original SHA256 `{entry['sha256']}`.\n\n"
            "Each part is a standalone CSV with the same header. Restore the exact original bytes with `publish.py --restore-tables`. Quoted fields containing newlines stay in one part.\n\n"
            + "\n".join(f"- [{Path(p['path']).name}]({Path(p['path']).name}): {p['rows']} records, SHA256 `{p['sha256']}`" for p in parts) + "\n", encoding="utf-8")
        manifest["tables"][rel] = entry
    previous = table_dir / "manifest.json"
    if previous.exists() and read(previous) != manifest:
        raise RuntimeError("An earlier table archive manifest differs")
    write(previous, manifest)
    return manifest


def verify_archive_entry(result, entry, output=None):
    digest = hashlib.sha256()
    total_bytes = 0
    header = None
    total_rows = 0
    for index, part in enumerate(entry["parts"]):
        path = safe_relative(result, part["path"])
        if sha(path) != part["sha256"] or path.stat().st_size != part["bytes"]:
            raise RuntimeError("Archived CSV part changed")
        rows = sum(1 for _ in csv_records(path)) - 1
        if rows != part["rows"]:
            raise RuntimeError("Archived CSV logical row count changed")
        total_rows += rows
        with path.open("rb") as file:
            prefix = file.read(entry["header_bytes"])
            if hashlib.sha256(prefix).hexdigest() != entry["header_sha256"] or (header is not None and prefix != header):
                raise RuntimeError("Archived CSV header changed")
            header = prefix
            if index == 0:
                digest.update(prefix)
                total_bytes += len(prefix)
                if output:
                    output.write(prefix)
            for chunk in iter(lambda: file.read(LIMIT), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
                if output:
                    output.write(chunk)
    if digest.hexdigest() != entry["sha256"] or total_bytes != entry["bytes"] or total_rows != entry["rows"]:
        raise RuntimeError("Exact original CSV restoration failed")


def restore_tables(result=RESULT):
    manifest = read(result / "table_shards/manifest.json")
    for name, entry in manifest["tables"].items():
        path = safe_relative(result, name)
        if path.exists():
            if sha(path) != entry["sha256"]:
                raise RuntimeError("Refusing to overwrite a different existing original CSV")
            verify_archive_entry(result, entry)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".restore.{os.getpid()}")
        try:
            with temporary.open("wb") as file:
                verify_archive_entry(result, entry, file)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def prepare_result_publication(evidence, parent_gate, result=RESULT, out=OUT):
    provenance = result / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for name in ("models_qualification.json", "scoring_completion.json", "source_publication.json"):
        shutil.copyfile(out / name, provenance / name)
    write(provenance / "parent_completion_gate.json", parent_gate)
    write(provenance / "publication_inputs.json", evidence["verified_inputs"])
    archives = archive_tables(result)
    original_report = result / "METRICS_REPORT.md"
    text = original_report.read_text(encoding="utf-8")
    def relocate(match):
        name = match[1]
        if name in archives["tables"]:
            return "](" + archives["tables"][name]["index"] + ")"
        return match[0]
    rewritten = re.sub(r"\]\(([^)]+)\)", relocate, text)
    (result / "METRICS_PUBLISHED_REPORT.md").write_text(rewritten, encoding="utf-8")
    (result / "README.md").write_text("# Unified additional metrics\n\n"
        "[Complete report](METRICS_PUBLISHED_REPORT.md). The original analysis report and its hashes are retained.\n\n"
        "Large CSVs are stored as standalone parts under `table_shards`. Their manifest preserves the exact original bytes and SHA256. The experiment's `publish.py --restore-tables` restores these original CSVs. Full provenance-verifying analysis additionally requires its registered local input files. No model weights, caches, tensors or image datasets are included.\n\n"
        "The real-weight qualification uses synthetic pixels and is not a scientific result. The final tables use the registered 100 sources and paired source bootstrap.\n", encoding="utf-8")
    write(provenance / "report_transform.json", {"original_report_sha256": sha(original_report), "published_report_sha256": sha(result / "METRICS_PUBLISHED_REPORT.md"), "transform": "Only relative links to archived large CSVs are redirected to their tracked part indexes"})
    return archives


def source_files(source=HERE):
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise RuntimeError("Symlinks are not publishable source")
        if path.suffix in SOURCE_SUFFIXES:
            files.append(path)
        elif path.suffix not in {".pyc"}:
            raise RuntimeError(f"Unexpected source artifact requires explicit review: {path}")
    if not files:
        raise RuntimeError("No reviewed metric source files found")
    return files


def result_files(result=RESULT, archives=None):
    omitted = set((archives or {"tables": {}})["tables"])
    files = []
    for path in sorted(result.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(result).as_posix()
        if rel in omitted:
            continue
        if path.is_symlink() or path.suffix not in RESULT_SUFFIXES or path.stat().st_size > LIMIT:
            raise RuntimeError(f"Unexpected or oversized result artifact: {path}")
        files.append(path)
    return files


def assert_scope(names, allowed):
    if set(names) - set(allowed):
        raise RuntimeError(f"Unrelated staged or dirty tracked files: {sorted(set(names)-set(allowed))[:10]}")


def verify_previous_publication(record, log):
    verify_push_record(record)
    command(["git", "fetch", "origin"], log=log)
    command(["git", "cat-file", "-e", record["commit"] + "^{commit}"], log=log)
    command(["git", "merge-base", "--is-ancestor", record["commit"], "origin/main"], log=log)


def run_checks(log):
    command([sys.executable, "tools/update_repository_manifest.py"], log=log, cpu=True)
    command(["git", "add", "--", "release_manifest.json"], log=log)
    command([sys.executable, "tools/verify_repository.py"], log=log, cpu=True)
    command([sys.executable, "tools/run_cpu_checks.py"], log=log, cpu=True)
    for test in sorted(HERE.glob("test_*.py")):
        command([sys.executable, str(test)], log=log, cpu=True)


def push_record(record, record_path, log):
    """Finish a new commit or resume a verified commit after a push-only failure."""
    command(["git", "push", "origin", "main"], log=log)
    remote_lines = command(["git", "ls-remote", "origin", "refs/heads/main"], capture=True).decode().split()
    if len(remote_lines) != 2 or remote_lines[0] != record["commit"]:
        raise RuntimeError("Normal push did not verify the expected remote main SHA")
    record.update(status="PUSHED", remote_commit=remote_lines[0], verified_utc=timestamp())
    write(record_path, record)
    if record["phase"] == "results":
        write(OUT / "completion.json", {"status": "UNIFIED_METRICS_COMPLETE", "publication": record, "stop": True, "synthetic": False, "training_updates": 0, "policy_selection_updates": 0})
    return record


def main(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    # The gate runs before git add or any tracked file is changed.
    parent_gate = verify_parent_gate()
    sources = source_files()
    source_identity = bindings(sources)
    record_path = OUT / f"{phase}_publication.json"
    log_path = OUT / f"{phase}_publication_checks.log"
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT.resolve() or git("branch", "--show-current") != "main":
        raise RuntimeError("Publication requires the sole project root on branch main")
    if git("remote", "get-url", "origin") not in {"git@github.com:daiqizai/var-comm.git", "https://github.com/daiqizai/var-comm.git"}:
        raise RuntimeError("Unexpected publication repository")
    with log_path.open("a", encoding="utf-8") as log:
        previous = read(record_path) if record_path.exists() else None
        if previous and previous.get("status") == "PUSHED" and previous.get("source_bindings") == source_identity:
            verify_bindings(previous["published_files"], "already published files")
            verify_previous_publication(previous, log)
            if phase == "results":
                write(OUT / "completion.json", {"status": "UNIFIED_METRICS_COMPLETE", "publication": previous, "stop": True, "synthetic": False, "training_updates": 0, "policy_selection_updates": 0})
            return previous
        if previous and previous.get("status") == "COMMITTED" and previous.get("source_bindings") == source_identity:
            if previous.get("checks") != "PASS" or git("rev-parse", "HEAD") != previous.get("commit"):
                raise RuntimeError("Cannot safely resume a different committed publication")
            verify_bindings(previous["published_files"], "committed publication files")
            if git_names("diff", "--cached", "--name-only") or git_names("diff", "--name-only"):
                raise RuntimeError("Tracked files changed after the pending publication commit")
            for path, digest in previous["published_files"].items():
                committed_bytes = command(["git", "show", "HEAD:" + Path(path).relative_to(ROOT).as_posix()], capture=True)
                if hashlib.sha256(committed_bytes).hexdigest() != digest:
                    raise RuntimeError("Pending commit does not contain the verified publication bytes")
            command(["git", "fetch", "origin"], log=log)
            command(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], log=log)
            return push_record(previous, record_path, log)
        source_record = None
        if phase == "results":
            source_record = read(OUT / "source_publication.json")
            verify_previous_publication(source_record, log)
            if source_record.get("source_bindings") != source_identity:
                raise RuntimeError("Scoring source changed after its registered publication")
        # Reject unrelated tracked changes before creating a manifest from working bytes.
        allowed_prefixes = [HERE.relative_to(ROOT).as_posix() + "/"]
        if phase == "results":
            allowed_prefixes.append(RESULT.relative_to(ROOT).as_posix() + "/")
        dirty = git_names("diff", "--name-only", "--no-renames") | git_names("diff", "--cached", "--name-only", "--no-renames")
        if any(not any(name.startswith(prefix) for prefix in allowed_prefixes) for name in dirty):
            raise RuntimeError("Unrelated tracked changes must be preserved; publication stopped")
        evidence = verify_results() if phase == "results" else None
        archives = prepare_result_publication(evidence, parent_gate) if evidence else None
        files = sources + (result_files(archives=archives) if phase == "results" else [])
        intended = {p.relative_to(ROOT).as_posix() for p in files}
        tracked = git_names("ls-files")
        if archives and any((RESULT / p).relative_to(ROOT).as_posix() in tracked for p in archives["tables"]):
            raise RuntimeError("An oversized original CSV is already tracked; do not remove it automatically")
        staged = git_names("diff", "--cached", "--name-only", "--no-renames")
        assert_scope(staged, intended)
        # Explicit file paths prevent the large original CSVs or weights from being staged.
        for start in range(0, len(files), 100):
            command(["git", "add", "--", *[str(p.relative_to(ROOT)) for p in files[start:start+100]]], log=log)
        published_identity = bindings(files)
        run_checks(log)
        allowed = intended | {"release_manifest.json"}
        assert_scope(git_names("diff", "--cached", "--name-only", "--no-renames"), allowed)
        if git_names("diff", "--name-only", "--no-renames"):
            raise RuntimeError("Tracked files changed while publication checks were running")
        verify_bindings(published_identity, "publication input")
        if verify_parent_gate()["receipts"] != parent_gate["receipts"]:
            raise RuntimeError("Original completion evidence changed during publication checks")
        for path, digest in published_identity.items():
            staged_bytes = command(["git", "show", ":" + Path(path).relative_to(ROOT).as_posix()], capture=True)
            if hashlib.sha256(staged_bytes).hexdigest() != digest:
                raise RuntimeError("Staged bytes differ from verified publication input")
        command(["git", "fetch", "origin"], log=log)
        command(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], log=log)
        command(["git", "merge-base", "--is-ancestor", parent_gate["parent_commit"], "origin/main"], log=log)
        if git_names("diff", "--cached", "--name-only"):
            message = OUT / f"{phase}_commit_message.txt"
            message.write_text(("Register frozen unified image metric evaluation" if phase == "source" else "Publish paired additional metrics for frozen N512 N1024 and two-method outputs") + "\n", encoding="utf-8")
            command(["git", "commit", "-F", str(message)], log=log)
        commit = git("rev-parse", "HEAD")
        record = {"status": "COMMITTED", "phase": phase, "commit": commit, "checks": "PASS", "source_bindings": source_identity,
            "published_files": published_identity, "parent_gate": parent_gate, "time_utc": timestamp(), "log_path": str(log_path)}
        write(record_path, record)
        return push_record(record, record_path, log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--phase", choices=("source", "results"))
    actions.add_argument("--restore-tables", action="store_true")
    args = parser.parse_args()
    try:
        if args.restore_tables:
            restore_tables()
        else:
            main(args.phase)
    except Exception as error:
        write(OUT / f"publication_failure_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json", {"phase": args.phase, "error": str(error), "traceback": traceback.format_exc(), "time_utc": timestamp()})
        raise
