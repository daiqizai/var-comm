"""Publish sealed M1 results and explicitly track reviewed, inactive extensions.

This standalone recovery does not edit any registered experiment or extension
source. It reuses the original M1 preparation and completion functions, preserves
the original source_bindings field, and never qualifies or starts either newly
tracked extension. The remote location is tools/recover_m1_publication_20261002.py.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SELF = Path(__file__).resolve()
ROOT = SELF.parents[1]
PARENT_REL = Path("outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002")
SPEED_REL = Path("experiments/metric-speed-20261002")
METRICS_REL = Path("experiments/unified-metrics-20261002")
BASE_REL = Path("experiments/scale-causal-partial-residual-20261002")
CODE_STATUS = dict(source_tracking_only=True, qualified=False, activated=False)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seal(path, record):
    if Path(path).exists():
        if read(path) != record:
            raise RuntimeError("Recovery registration changed: " + str(path))
    else:
        write(path, record)


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def command(args, *, log=None, capture=False, cpu=False):
    environment = os.environ.copy()
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(arg) for arg in args], cwd=ROOT, env=environment,
        stdout=subprocess.PIPE if capture else log, stderr=subprocess.PIPE if capture else subprocess.STDOUT)
    if log:
        log.flush()
    if result.returncode:
        tail = result.stderr.decode(errors="replace")[-2000:] if capture else "see recovery_checks.log"
        raise RuntimeError(f"Recovery command failed ({result.returncode}): {args!r}: {tail}")
    return result.stdout if capture else b""


def git(*args):
    return command(["git", *args], capture=True).decode().strip()


def names(*args):
    return set(filter(None, command(["git", *args, "-z"], capture=True).decode().split("\0")))


def bindings(paths):
    return {str(Path(path).resolve()): sha(path) for path in sorted(paths)}


def verify(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError("Nonempty checked file bindings are required")
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise RuntimeError("Checked recovery input changed: " + path)


def additional_inventory(controller, metric_assets_sha256=None, metric_assets_test_sha256=None):
    speed = ROOT / SPEED_REL
    metrics = ROOT / METRICS_REL
    speed_bindings = controller.bound
    metric_bindings = controller_module.sources(metrics)
    registration_path = ROOT / "outputs/UNIFIED-METRICS-20261002/supervisor_registration.json"
    if not registration_path.exists() or read(registration_path).get("source_bindings") != metric_bindings:
        raise RuntimeError("Reviewed metric source inventory must match its existing supervisor registration")
    if controller_module.sources(speed) != speed_bindings:
        raise RuntimeError("Reviewed speed source inventory differs from the frozen controller")
    tools = [SELF]
    optional = ROOT / "tools/recover_metric_assets_20261002.py"
    optional_test = ROOT / "tools/test_recover_metric_assets_20261002.py"
    extra_test = None
    if optional.exists() or optional_test.exists() or metric_assets_sha256 or metric_assets_test_sha256:
        for path, approved in ((optional, metric_assets_sha256), (optional_test, metric_assets_test_sha256)):
            if not path.is_file() or path.is_symlink() or not approved or sha(path) != approved:
                raise RuntimeError("Optional asset recovery tool and test each require their explicit reviewed SHA-256: " + str(path))
        tools.extend([optional, optional_test])
        extra_test = optional_test
    additional = {**speed_bindings, **metric_bindings, **bindings(tools)}
    if len(additional) != len(speed_bindings) + len(metric_bindings) + len(tools):
        raise RuntimeError("Recovery source inventories overlap")
    return additional, registration_path, extra_test


def assert_scope(staged, allowed):
    unexpected = staged - allowed
    if unexpected:
        raise RuntimeError("Unrelated staged paths must be preserved: " + repr(sorted(unexpected)))


def snapshot(evidence):
    """Keep exact index bytes and staged blob IDs before touching the index."""
    if (evidence / "initial_snapshot.json").exists():
        record = read(evidence / "initial_snapshot.json")
        verify(record["evidence_files"])
        return record
    evidence.mkdir(parents=True, exist_ok=True)
    status = command(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], capture=True)
    index = Path(git("rev-parse", "--git-path", "index"))
    if not index.is_absolute():
        index = ROOT / index
    shutil.copyfile(index, evidence / "initial_index")
    (evidence / "initial_status.bin").write_bytes(status)
    (evidence / "initial_staged_blobs.bin").write_bytes(command(["git", "ls-files", "--stage", "-z"], capture=True))
    (evidence / "initial_staged_names.bin").write_bytes(command(["git", "diff", "--cached", "--name-only", "-z"], capture=True))
    record = dict(HEAD=git("rev-parse", "HEAD"), original_index_path=str(index),
        staged_paths=sorted(names("diff", "--cached", "--name-only")),
        evidence_files=bindings([evidence / name for name in ("initial_index", "initial_status.bin", "initial_staged_blobs.bin", "initial_staged_names.bin")]),
        preservation="Exact Git index plus staged blob identifiers; no reset, clean, deletion or unstaging", time=time.time())
    write(evidence / "initial_snapshot.json", record)
    return record


def committed_bytes(record):
    for path, digest in record["published_files"].items():
        relative = Path(path).relative_to(ROOT).as_posix()
        data = command(["git", "show", record["commit"] + ":" + relative], capture=True)
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Recorded publication bytes differ from the commit: " + relative)


def validate_record(record, original_sources, additional_sources):
    if (record.get("checks") != "PASS" or record.get("source_bindings") != original_sources
            or record.get("additional_source_bindings") != additional_sources or record.get("code_status") != CODE_STATUS
            or record.get("scope") != "M1_RESULTS_WITH_INACTIVE_SOURCE_RECONCILIATION"):
        raise RuntimeError("Existing recovery publication has a different checked scope")
    if record.get("status") not in ("COMMITTED", "PUSHED"):
        raise RuntimeError("Unknown recovery publication status")
    committed_bytes(record)


def push(record, receipt, original, log):
    command(["git", "push", "origin", "main"], log=log)
    remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
    if remote != record["commit"]:
        raise RuntimeError("Normal push did not verify the exact recovery commit")
    record.update(status="PUSHED", remote_commit=remote, push_verified_at=time.time())
    write(receipt, record)
    original.mark_complete("m1", record)
    return record


def main(metric_assets_sha256=None, metric_assets_test_sha256=None, metric_test_python=None):
    global controller_module
    parent = ROOT / PARENT_REL
    output = parent / "m1_recovery"
    output.mkdir(parents=True, exist_ok=True)
    controller_module = load("_m1_recovery_frozen_controller", ROOT / SPEED_REL / "controller.py")
    with ExitStack() as locks:
        for path in (output / "recovery.lock", ROOT / "outputs/METRIC-SPEED-20261002/controller.lock", parent / "supervisor.lock"):
            locks.enter_context(controller_module.acquire_lock(path))
        controller = controller_module.Controller(ROOT, ROOT / SPEED_REL)
        controller.initialize()  # Frozen member verification only; never takeover/run.
        launch = read(parent / "supervisor_launch.json")
        identity = controller_module.checked_identity(launch)
        if not controller_module.exited(identity, controller_module.process_identity(identity["pid"])):
            raise RuntimeError("Previous controller must have exited before publication recovery")
        controller.require_previous_worker_exit(read(parent / "supervisor_status.json"))
        if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT or git("branch", "--show-current") != "main":
            raise RuntimeError("Recovery requires the original repository on main")
        if git("remote", "get-url", "origin") not in ("git@github.com:daiqizai/var-comm.git", "https://github.com/daiqizai/var-comm.git"):
            raise RuntimeError("Unexpected recovery repository")
        original = load("_m1_recovery_original_publish", ROOT / BASE_REL / "publish.py")
        original_sources = original.source_bindings()
        if original_sources != controller.registration["original_source_bindings"]:
            raise RuntimeError("Original publication source_bindings changed")
        additional, metrics_registration, extra_test = additional_inventory(controller, metric_assets_sha256, metric_assets_test_sha256)
        for stage in controller.stages[:6]:
            if not controller.receipt(stage):
                raise RuntimeError("Every M1 scientific stage must already be sealed COMPLETE")
        analysis = read(original.RESULT / "m1_analysis_completion.json")
        original.verify_analysis(analysis)
        scientific_inputs = dict(analysis["inputs"], **analysis["outputs"])
        scientific_inputs.update(bindings([ROOT / stage["receipt"] for stage in controller.stages[:6]]))
        reviewed_inputs = dict(scientific_inputs, **additional)
        reviewed_inputs[str(metrics_registration)] = sha(metrics_registration)
        registration_path = output / "recovery_registration.json"
        registration = dict(source_bindings=original_sources, additional_source_bindings=additional,
            code_status=CODE_STATUS, reviewed_inputs=reviewed_inputs,
            controller_registration_sha256=sha(controller.out / "controller_registration.json"),
            scope="M1_RESULTS_WITH_INACTIVE_SOURCE_RECONCILIATION", training_updates=0, policy_selection_updates=0)
        seal(registration_path, registration)
        receipt = parent / "m1_publication.json"
        with (output / "recovery_checks.log").open("a", encoding="utf-8") as log:
            if receipt.exists():
                previous = read(receipt)
                validate_record(previous, original_sources, additional)
                controller.verify();verify(reviewed_inputs)
                command(["git", "fetch", "origin"], log=log)
                if previous["status"] == "PUSHED":
                    if previous.get("remote_commit") != previous["commit"]:
                        raise RuntimeError("Existing recovery push SHA differs")
                    command(["git", "merge-base", "--is-ancestor", previous["commit"], "origin/main"], log=log)
                    original.mark_complete("m1", previous)
                    return previous
                if git("rev-parse", "HEAD") != previous["commit"] or names("diff", "--cached", "--name-only") or names("diff", "--name-only"):
                    raise RuntimeError("Pending recovery commit or tracked bytes changed")
                verify(previous["published_files"])
                command(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], log=log)
                return push(previous, receipt, original, log)
            initial = snapshot(output / "evidence")
            allowed_prefixes = [BASE_REL.as_posix() + "/", original.RESULT.relative_to(ROOT).as_posix() + "/"]
            fixed = {".gitignore", "RESEARCH_STATUS.md", "PROGRESS.md", "EXPERIMENTS.md", "release_manifest.json",
                     "reports/scale_causal_m1_20261002.md"}
            added_paths = {Path(path).relative_to(ROOT).as_posix() for path in additional}
            def allowed(path):
                return path in fixed or path in added_paths or any(path.startswith(prefix) for prefix in allowed_prefixes)
            dirty = names("diff", "--name-only") | names("diff", "--cached", "--name-only")
            if any(not allowed(path) for path in dirty):
                raise RuntimeError("Unrelated tracked/staged changes must be preserved")
            if not set(initial["staged_paths"]).issubset(names("diff", "--cached", "--name-only")):
                raise RuntimeError("Initial M1 staged paths were removed after the preserved index snapshot")
            target = original.prepare("m1")
            original.verify_analysis(analysis);controller.verify();verify(reviewed_inputs)
            provenance = original.RESULT / "provenance"
            shutil.copyfile(registration_path, provenance / "m1_source_reconciliation.json")
            shutil.copyfile(output / "evidence/initial_snapshot.json", provenance / "m1_recovery_initial_index_identity.json")
            note = provenance / "m1_source_reconciliation.md"
            note.write_text("# M1 publication recovery\n\n"
                "The original M1 scientific stages were complete before this publication. "
                "The frozen M1 publisher stopped because the repository checker requires all self-written sources to be tracked. "
                "This independent recovery reuses its preparation and completion functions, preserves the original execution source bindings, "
                "and also tracks the reviewed receiver acceleration and unified metric source files. "
                "Those additional sources are unqualified and inactive at this commit. "
                "Receiver acceleration still requires its calibration-only parity benchmark and separate publication after M1; "
                "unified metric scoring still requires original M2 completion and its own qualification/publication. "
                "No training, scientific policy selection, rerun of M1 quality evaluation, old source edit, reset or clean is performed. "
                "The exact initial Git index and staged blob identifiers remain in the original output recovery evidence directory.\n", encoding="utf-8")
            published_map = provenance / "m1_published_files.json"
            original.write(published_map, {path.relative_to(original.RESULT).as_posix(): original.sha(path)
                for path in original.RESULT.rglob("*") if path.is_file() and path.stat().st_size <= 10_000_000 and path != published_map})
            paths = [BASE_REL.as_posix(), original.RESULT.relative_to(ROOT).as_posix(), target.relative_to(ROOT).as_posix(),
                     ".gitignore", "RESEARCH_STATUS.md", "PROGRESS.md", "EXPERIMENTS.md", *sorted(added_paths)]
            command(["git", "add", "--", *paths], log=log)
            command([sys.executable, "tools/update_repository_manifest.py"], log=log, cpu=True)
            command(["git", "add", "--", "release_manifest.json"], log=log)
            staged = names("diff", "--cached", "--name-only")
            if any(not allowed(path) for path in staged):
                raise RuntimeError("Unexpected staged scope after M1 source reconciliation")
            published_files = {str(ROOT / path): sha(ROOT / path) for path in staged if (ROOT / path).is_file()}
            # Bind unchanged original/new sources too, even when already tracked.
            published_files.update(original_sources);published_files.update(additional)
            command([sys.executable, "tools/verify_repository.py"], log=log, cpu=True)
            command([sys.executable, "tools/run_cpu_checks.py"], log=log, cpu=True)
            for directory, python in ((ROOT / BASE_REL, sys.executable), (ROOT / SPEED_REL, sys.executable),
                                      (ROOT / METRICS_REL, metric_test_python or sys.executable)):
                for test in sorted(directory.glob("test_*.py")):
                    command([python, test], log=log, cpu=True)
            if extra_test is not None:
                command([sys.executable, extra_test], log=log, cpu=True)
            controller.verify();verify(reviewed_inputs);original.verify_analysis(analysis);verify(published_files)
            if names("diff", "--name-only") or any(not allowed(path) for path in names("diff", "--cached", "--name-only")):
                raise RuntimeError("Publication worktree/scope changed during checks")
            for path, digest in published_files.items():
                data = command(["git", "show", ":" + Path(path).relative_to(ROOT).as_posix()], capture=True)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise RuntimeError("Staged bytes differ from reviewed recovery publication: " + path)
            command(["git", "fetch", "origin"], log=log)
            command(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], log=log)
            command(["git", "merge-base", "--is-ancestor", controller.registration["original_source_commit"], "HEAD"], log=log)
            message = output / "commit_message.txt"
            message.write_text("Publish sealed M1 results and reconcile reviewed inactive extension sources\n\n"
                "Preserve original experiment/source bindings and all completed calibration, development, timing and analysis evidence. "
                "Track the reviewed speed and unified metric sources to satisfy the unchanged repository checker. "
                "Additional code remains unqualified and inactive; original ordered qualification and activation gates still apply.\n", encoding="utf-8")
            if names("diff", "--cached", "--name-only"):
                command(["git", "commit", "-F", message], log=log)
            record = dict(status="COMMITTED", commit=git("rev-parse", "HEAD"), checks="PASS",
                source_bindings=original_sources, additional_source_bindings=additional, code_status=CODE_STATUS,
                published_files=published_files, scope=registration["scope"], training_updates=0, policy_selection_updates=0,
                recovery_registration_sha256=sha(registration_path), initial_index_evidence_sha256=sha(output / "evidence/initial_snapshot.json"),
                check_log=str(output / "recovery_checks.log"), time=time.time())
            write(receipt, record)
            committed_bytes(record)
            return push(record, receipt, original, log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-metric-assets-sha256")
    parser.add_argument("--reviewed-metric-assets-test-sha256")
    parser.add_argument("--metric-test-python")
    args = parser.parse_args()
    try:
        main(args.reviewed_metric_assets_sha256, args.reviewed_metric_assets_test_sha256, args.metric_test_python)
    except Exception as error:
        write(ROOT / PARENT_REL / "m1_recovery" / ("failure_" + str(time.time_ns()) + ".json"),
              dict(status="FAILED", error=str(error), time=time.time()))
        raise
