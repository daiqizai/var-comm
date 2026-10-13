"""Finite R0 body-only BLER supplement; registration required before science.

There is no registration/authorization command. Import, validation and unit
tests never construct the real PHY. A reviewed registration pins all 1,389
missing cells before the first of exactly 64 trials per cell. Historical points
and paid headers are not rerun. Unknown failures retain reservations, fail the
attempt, and cannot be replayed by resume.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback

import numpy as np

SCHEMA = "LAYERED_RAW_R0_BLER_SUPPLEMENT_V1"
NAMESPACE = "UEP_MISMATCH_RAW_R0_BLER_SUPPLEMENT_V1"
BODY_SESSION = NAMESPACE + "_PUBLIC_SCRAMBLER"
SNRS = (4, 7, 10)
TRIALS = 64
CELLS = 1389
CAP = CELLS * TRIALS
Q = {"QPSK": 2, "16QAM": 4, "64QAM": 6}
RAW_REQUEST_SHA = "9e587fcec3be465d8d6c43a7123400635441918befbbd45a6339f0f208af2faf"
SCREEN_PINS = dict(screening_source="a1968d3196c5ebb4a733b31c7de4bbc1b316e562756b23300723cf6f4b66d7c0",
    screening_contract="6380037674b15ba2a7346c1af6153832177a38837ab769a5d7558ad91c783d24",
    authorization="22b3bb973dc965b14c654c14446ba30a3e0b7fe80892ec38eec905e5e1500507",
    server_relocation="00b82102d73f615f0e9338fdb7f77570fe2b0d1b227f801a7264e42e87c9342d",
    missing="62b0e5959f0765f5684421decbea584c5ba31709ada69ae6befe4dd55d31ce4e",
    historical="3d340790a18d7c7c90b2c67dd60cfc21fb057c4cdc0b0b04d59241691f15115a")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin(path):
    return dict(path=str(Path(path).absolute()), sha256=sha(path))


def checkpin(value):
    p = Path(value["path"])
    require(p.is_absolute() and p.is_file() and not p.is_symlink() and sha(p) == value["sha256"], "Changed/missing pin: " + str(p))
    return p


def readpin(value):
    return json.loads(checkpin(value).read_text(encoding="utf-8-sig"))


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8", newline="\n") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return pin(p)


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def rows(value):
    if isinstance(value, list):
        return value
    return value["points"] if "points" in value else value["rows"]


def cell_key(row):
    return str(row["phy_key"]), int(row["snr_db"])


def validate_group(g):
    q = Q[g["modulation"]]
    k, n = g["information_bits"], g["transmitted_bits"]
    layout = g["layout"]
    require(type(k) is int and type(n) is int and g["source_bits"] > 0
        and g["source_bits"] % 12 == 0 and k == g["source_bits"] + 16
        and g["crc_bits"] == 16 and g["tail_bits"] == 0
        and n == q * g["symbols"] and g["modulation_padding_bits"] == 0, "Raw body accounting differs")
    require(layout["k"] == k and layout["n"] == n and layout["k_ldpc"] - layout["k_filler"] == k
        and layout["shortening_bits"] == layout["k_filler"] and layout["repetition_bits"] == 0
        and layout["layout_id"] == digest({a:b for a,b in layout.items() if a != "layout_id"}), "Layout seal/filler differs")
    require(layout["puncturing_first_2Z"] == 2 * layout["z"]
        and layout["puncturing_remaining"] == layout["n_cb_comp"] - n
        and layout["puncturing_bits"] == layout["mother_bits"] - layout["k_filler"] - n
        and layout["power_protocol"] == "fixed_constellation_average_Es2", "Rate matching/power changed")
    expected = dict(backend=layout["implementation"], source_bits=g["source_bits"], CRC=16,
        layout=layout, modulation=g["modulation"], modulation_padding_bits=0)
    require(g["phy_key"] == digest(expected), "Exact body physical key differs")
    return q, k, n


def validate_cells(groups, missing, historical):
    """Reconstruct exact missing selection grid, excluding every historical cell."""
    require(len(groups) == 687 and all(key == g["phy_key"] for key,g in groups.items()), "Frozen 687 body layouts required")
    for g in groups.values():
        validate_group(g)
    old = {cell_key(r) for r in historical}
    new = {cell_key(r) for r in missing}
    expected = {(key, s) for key in groups for s in SNRS}
    require(len(old) == len(historical) == 672 and len(new) == len(missing) == CELLS
        and not old & new and old | new == expected, "Historical/missing exact selection grid differs")
    for row in missing:
        g = groups[row["phy_key"]]
        q, k, n = validate_group(g)
        require(row["snr_db"] in SNRS and row["k"] == k and row["n"] == n and row["q"] == q
            and row["layout_id"] == g["layout"]["layout_id"], "Missing cell layout dimensions differ")
        if "source_bits" in row:
            require(row["source_bits"] == g["source_bits"], "Missing source length differs")
        require(row.get("trials", TRIALS) == TRIALS
            and row.get("layout", g["layout"]) == g["layout"]
            and row.get("layout_sha256", digest(g["layout"])) == digest(g["layout"]), "Full missing-cell layout differs")
    require(all(r["n_blocks"] > 0 and r["n_correct"] + r["n_reject"] + r["n_undetected"] == r["n_blocks"]
                for r in historical), "Historical outcome counts are not closed")
    return sorted(missing, key=cell_key)


def validate(registration):
    reg = readpin(registration)
    require(reg["schema"] == SCHEMA and reg["status"] == "REGISTERED_R0_BLER_SUPPLEMENT_V1", "Reviewed real registration required")
    require(reg["namespace"] == NAMESPACE and reg["body_session"] == BODY_SESSION
        and reg["snrs_db"] == list(SNRS) and reg["trials_per_point"] == TRIALS
        and reg["point_count"] == CELLS and reg["body_trial_cap"] == CAP
        and reg["header_trials"] == 0 and reg["training_updates"] == 0
        and reg["automatic_retry"] is False and reg["automatic_successor"] is False,
        "Finite authorized supplement changed")
    pins = reg["pins"]
    require({"groups", "missing", "historical", "raw_request", "r0_bler"} | set(SCREEN_PINS) <= set(pins), "Missing supplement pins")
    for p in pins.values():
        checkpin(p)
    require(all(pins[k]["sha256"] == value for k,value in SCREEN_PINS.items()), "Reviewed screening rule/authorization changed")
    contract = readpin(pins["screening_contract"])
    relocation = readpin(pins["server_relocation"])
    require(relocation["original_contract_sha256"] == SCREEN_PINS["screening_contract"], "Relocation differs from original contract")
    for row in relocation["mapping"]:
        require(contract["input_bindings"][row["role"]] == dict(path=row["registered_local_path"], sha256=row["sha256"]), "Unregistered relocated path")
        checkpin(dict(path=row["server_path"], sha256=row["sha256"]))
    require(Path(pins["r0_bler"]["path"]).resolve() == Path(__file__).resolve(), "Wrong executing entry")
    require(pins["raw_request"]["sha256"] == RAW_REQUEST_SHA, "Frozen raw backend authority changed")
    raw = readpin(pins["raw_request"])
    for p in raw["raw_runtime_binding"]["files"]:
        checkpin(p)
    groups = readpin(pins["groups"])
    cells = validate_cells(groups, rows(readpin(pins["missing"])), rows(readpin(pins["historical"])))
    require(digest(cells) == reg["ordered_cells_sha256"], "Frozen cell order changed")
    cfg = reg["runtime"]
    require(cfg["python"] == raw["CPU_python"] and cfg["threads"] == 2
        and len(set(cfg["cpu_affinity"])) == 2 and cfg["CUDA_VISIBLE_DEVICES"] == ""
        and cfg["max_seconds"] > 0 and cfg["deadline_unix"] > 0, "Original two-core CPU runtime required")
    root = Path(cfg["storage_root"]).resolve()
    out = Path(cfg["output_directory"]).resolve()
    require(str(root).endswith("/modelTeam/liberai-tmp/liulu") and out != root and root in out.parents,
        "New journals/logs must remain under personal temporary storage")
    timing_run = Path(cfg["timing_v6_run_directory"])
    require(timing_run.is_absolute() and root in timing_run.parents and timing_run.name == "six_method_timing_v6",
        "Bind the actual timing v6 runtime directory")
    return reg, raw, groups, cells


def random_inputs(group, snr, trial):
    q, _, n = validate_group(group)
    require(type(trial) is int and 0 <= trial < TRIALS and snr in SNRS, "Unregistered trial")
    base = [NAMESPACE, group["phy_key"], group["layout"]["layout_id"], int(snr), trial]
    def rng(label):
        seed = int.from_bytes(hashlib.sha256(canonical(base + [label]).encode()).digest(), "big")
        return np.random.Generator(np.random.PCG64(seed))
    payload = rng("payload").integers(0, 2, group["source_bits"], dtype=np.uint8)
    noise = rng("standard_noise").standard_normal((n // q, 2)).astype(np.float32)
    counter = int.from_bytes(hashlib.sha256(canonical(base + ["public_counter"]).encode()).digest()[:8], "little") & ((1 << 63) - 1)
    return payload, noise, counter


class Ledger:
    """Durable reservation before TX; exact complete reuse only; no failed replay."""
    def __init__(self, path, registration_sha, cap=CAP):
        require(0 < cap <= CAP, "Body trial cap expanded")
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, registration TEXT, cap INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS trials (id TEXT PRIMARY KEY, input TEXT, status TEXT, result TEXT)")
        old = self.db.execute("SELECT registration,cap FROM meta WHERE id=1").fetchone()
        if old is None:
            self.db.execute("INSERT INTO meta VALUES (1,?,?)", (registration_sha, cap)); self.db.commit()
        else:
            if old != (registration_sha, cap):
                self.db.close()
                raise ValueError("Different registration/cap cannot reopen ledger")
        self.cap = cap
        values = dict(self.db.execute("SELECT status,COUNT(*) FROM trials GROUP BY status"))
        self.completed, self.unresolved = values.get("COMPLETE", 0), values.get("RESERVED", 0)
        if self.unresolved:
            self.db.close()
            raise ValueError("Unresolved prior attempt requires diagnosis; no automatic retry")

    def reserve(self, identity):
        key, encoded = digest(identity), canonical(identity)
        row = self.db.execute("SELECT input,status,result FROM trials WHERE id=?", (key,)).fetchone()
        if row is not None:
            require(row[0] == encoded and row[1] == "COMPLETE", "Unresolved or changed trial cannot replay")
            result = json.loads(row[2]); require(result["trial_identity"] == identity, "Complete trial identity differs")
            return key, result
        count = self.snapshot()
        require(count["unresolved"] == 0 and count["reserved"] < self.cap, "Unresolved trial/cap prevents new call")
        with self.db:
            self.db.execute("INSERT INTO trials VALUES (?,?,?,NULL)", (key, encoded, "RESERVED"))
        self.unresolved += 1
        return key, None

    def complete(self, key, result):
        require(result["status"] == "COMPLETE" and digest(result["trial_identity"]) == key, "Invalid completion identity")
        with self.db:
            cursor = self.db.execute("UPDATE trials SET status='COMPLETE',result=? WHERE id=? AND status='RESERVED'",
                (canonical(result), key))
            require(cursor.rowcount == 1, "Trial completion lacks reservation")
        self.completed += 1; self.unresolved -= 1

    def snapshot(self):
        return dict(reserved=self.completed + self.unresolved, complete=self.completed, unresolved=self.unresolved, cap=self.cap)

    def close(self):
        self.db.close()


def assess(payload, hard, accepted, decoded, crc_check):
    hard, decoded = np.asarray(hard), np.asarray(decoded)
    require(hard.shape == payload.shape and decoded.shape == (len(payload) + 16,)
        and np.isin(hard, (0, 1)).all() and np.isin(decoded, (0, 1)).all()
        and np.array_equal(hard, decoded[:-16]) and type(accepted) is bool, "Actual decoder output invalid")
    require(bool(crc_check(decoded)) == accepted, "Actual CRC state contradicts hard information")
    equal = bool(np.array_equal(payload, hard))
    return dict(correct=accepted and equal, rejected=not accepted, undetected=accepted and not equal,
        payload_equal=equal, crc_accept=accepted, hard_payload_sha256=array_sha(hard), decoded_sha256=array_sha(decoded))


def execute_trial(group, snr, trial, runtime, ledger, runtime_identity):
    identity = dict(namespace=NAMESPACE, phy_key=group["phy_key"], layout_id=group["layout"]["layout_id"],
        snr_db=snr, trial=trial, session=BODY_SESSION, group=0, runtime_identity=runtime_identity)
    key, previous = ledger.reserve(identity)
    if previous is not None:
        return previous, True
    payload, noise, counter = random_inputs(group, snr, trial)
    wave, energy = runtime.transmit(payload, group, counter)
    wave = np.asarray(wave)
    require(wave.dtype == np.float32 and wave.shape == noise.shape and np.isfinite(wave).all()
        and np.isfinite(energy) and abs(float(np.square(wave, dtype=np.float64).sum()) - energy) <= max(1e-3, energy * 1e-6),
        "Actual unnormalized waveform/energy differs")
    received = (wave + noise * np.float32(10 ** (-snr / 20))).astype(np.float32)
    # The receive function has only received samples and public layout/SNR/counter.
    hard, accepted, decoded = runtime.receive(received, group, snr, counter)
    outcome = assess(payload, hard, accepted, decoded, runtime.crc_check)
    result = dict(status="COMPLETE", trial_identity=identity, public_counter=counter, **outcome,
        payload_sha256=array_sha(payload), standard_noise_sha256=array_sha(noise),
        waveform_sha256=array_sha(wave), received_sha256=array_sha(received), actual_E=energy,
        symbols=group["symbols"], observed_mean_Es=energy / group["symbols"],
        noise_variance_per_component=10 ** (-snr / 10), actual_packet_encodes=1, actual_packet_decodes=1,
        header_calls=0, neural_model_calls=0)
    ledger.complete(key, result)
    return result, False


def wilson(successes, count):
    require(0 <= successes <= count and count > 0, "Invalid binomial counts")
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return [max(0., center - radius), min(1., center + radius)]


def aggregate(group, snr, records):
    require(len(records) == TRIALS and [r["trial_identity"]["trial"] for r in records] == list(range(TRIALS))
        and all(r["status"] == "COMPLETE" and r["trial_identity"]["phy_key"] == group["phy_key"]
                and r["trial_identity"]["snr_db"] == snr for r in records), "Only closed exact 64-trial cells aggregate")
    counts = {"n_" + name:sum(r[name] for r in records) for name in ("correct", "rejected", "undetected")}
    counts["n_reject"] = counts.pop("n_rejected")
    require(sum(counts.values()) == TRIALS, "Software failures must not become scientific rejections")
    value = dict(kind="body", phy_key=group["phy_key"], snr_db=snr, n_blocks=TRIALS, **counts,
        block_errors=TRIALS - counts["n_correct"], physical_configuration=group,
        physical_configuration_sha256=digest(group), layout_id=group["layout"]["layout_id"],
        code_impl=group["layout"]["implementation"], power_protocol="fixed_constellation_average_Es2",
        source_id="*", applicability="iid_scrambled_payload_approximation",
        real_image_payload_probability_validation_complete=False,
        source_probability_mode="Independent random raw payload and public coding scrambler; image-payload validation incomplete",
        interval="Wilson two-sided95% binomial", zero_observed_errors_is_not_true_BLER_zero=True,
        random_namespace=NAMESPACE, trial_result_sha256=digest(records),
        actual_packet_encodes=TRIALS, actual_packet_decodes=TRIALS, header_calls=0,
        actual_energy_mean=float(np.mean([r["actual_E"] for r in records])),
        observed_average_energy_per_complex_symbol=float(np.mean([r["observed_mean_Es"] for r in records])),
        actual_energy_min=min(r["actual_E"] for r in records), actual_energy_max=max(r["actual_E"] for r in records),
        actual_energy_sd=float(np.std([r["actual_E"] for r in records])))
    for name in ("correct", "reject", "undetected"):
        value["p_" + name] = counts["n_" + name] / TRIALS
        value["p_" + name + "_ci"] = wilson(counts["n_" + name], TRIALS)
    return value


def load(path, name):
    require(name not in sys.modules, "Fresh module namespace required: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


class RealRuntime:
    def __init__(self, raw, groups):
        roles = raw["raw_runtime_binding"]["roles"]
        modules = {}
        for role, name in (("original_common", "uep_common"), ("original_planner", "profiles"),
                           ("original_backend", "_layered_r0_ldpc"), ("original_phy", "uep_phy"),
                           ("plan_module", "main_raw64_plan_only"), ("adapter_module", "main_raw64_packet_adapter")):
            modules[name] = load(roles[role], name)
        self.legacy, self.packet = modules["uep_phy"], modules["main_raw64_packet_adapter"]
        require(self.packet.Q == Q, "Explicit raw q2/q4/q6 adapter required")
        old = modules["_layered_r0_ldpc"].SionnaBackend("cpu", roles["legacy_qualification"])
        require(old.qualified, "Original qualified LDPC implementation required")
        self.packet.runtime_flags(old.torch)
        self.backend = self.packet.Backend(old)
        require(not old.torch.cuda.is_initialized() and old.torch.get_num_threads() == 2
            and old.torch.get_num_interop_threads() == 1, "Original CPU numerical settings differ")
        for group in groups.values():
            q, k, n = validate_group(group)
            require(self.backend.plan(k, n, q) == group["layout"], "Actual k/n/q/bg/z/filler/rate-match layout changed")
        self.identity = dict(backend=self.backend.identity, raw64_backend=self.backend.identity64,
            torch=old.torch.__version__, numpy=np.__version__, CPU_python=sys.executable, threads=2, interop_threads=1,
            TF32=False, deterministic=old.torch.are_deterministic_algorithms_enabled(), actual_host=os.uname().nodename)

    def transmit(self, payload, group, counter):
        q, _, n = validate_group(group)
        wave, meta = self.packet.transmit_packet(self.backend, self.legacy, payload[None], n, q, [counter], 0, BODY_SESSION)
        require(meta["k"] == group["information_bits"] and meta["n"] == n and meta["q"] == q
            and meta["crc_bits"] == 16, "Original actual TX metadata differs")
        return wave[0].detach().cpu().numpy().astype(np.float32), float(meta["actual_E"][0])

    def receive(self, received_samples, public_group, true_snr, public_counter):
        q, k, n = validate_group(public_group)
        received = self.backend.torch.as_tensor(received_samples[None], dtype=self.backend.torch.float32, device="cpu")
        hard, accept, decoded = self.packet.receive_packet(self.backend, self.legacy, received, k, n, q,
            true_snr, [public_counter], 0, BODY_SESSION)
        require(np.asarray(hard).shape == (1, k - 16) and np.asarray(decoded).shape == (1, k)
            and np.asarray(accept).shape == (1,) and np.asarray(accept).dtype == np.dtype(bool), "Actual receive shapes differ")
        return np.asarray(hard)[0], bool(accept[0]), np.asarray(decoded)[0]

    def crc_check(self, decoded):
        return self.legacy.crc_accept_batch(decoded)


def timing_idle(directory, run_directory):
    """Conservative read-only conflict check; no scheduling, killing or retry."""
    d = Path(directory)
    if (d / "launch.json").exists() or (d / "dispatch.json").exists() or Path(run_directory).exists():
        if not (d / "actual_wait.json").is_file():
            return False
        wait = json.loads((d / "actual_wait.json").read_text())
        if not wait.get("actual_child_waited", wait.get("actual_wait", False)):
            return False
        for name in ("launch.json", "dispatch.json", "worker_launch.json"):
            p = d / name
            if p.exists():
                value = json.loads(p.read_text())
                identities = [value] + [v for v in value.values() if isinstance(v, dict)]
                for identity in identities:
                    if "pid" in identity and Path(f"/proc/{identity['pid']}").exists():
                        if "start_ticks" not in identity or str(identity["start_ticks"]) == Path(f"/proc/{identity['pid']}/stat").read_text().split()[21]:
                            return False
        return True
    return True


def timing_shared_lock(directory, flock_module):
    """Use the existing v6 owner lock; its exclusive lock prevents overlap."""
    lock = (Path(directory) / "owner.lock").open("a+")
    try:
        flock_module.flock(lock, flock_module.LOCK_SH | flock_module.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def work(registration, owner_pid):
    reg, raw, groups, cells = validate(registration)
    cfg = reg["runtime"]; out = Path(cfg["output_directory"]); started = time.monotonic()
    require(sys.platform.startswith("linux") and str(Path(sys.executable).absolute()) == cfg["python"]
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "2" for k in THREAD_ENV), "Actual original CPU child required")
    require(set(cfg["cpu_affinity"]) == os.sched_getaffinity(0), "Exact two-core child affinity required")
    def guard():
        require(os.getppid() == owner_pid and not (out / "STOP").exists()
            and time.time() < cfg["deadline_unix"] and time.monotonic() - started < cfg["max_seconds"], "Parent/STOP/deadline boundary")
        require(timing_idle(cfg["timing_v6_directory"], cfg["timing_v6_run_directory"]), "Timing v6 launched/running; stop supplementary CPU work")
    ledger = Ledger(out / "trials.sqlite", registration["sha256"])
    try:
        guard()
        runtime = RealRuntime(raw, groups)
        identity = dict(registration=registration, actual_runtime=runtime.identity)
        if (out / "actual_runtime.json").exists():
            require(json.loads((out / "actual_runtime.json").read_text()) == identity, "Resume actual numerical environment changed")
        else:
            save(out / "actual_runtime.json", identity)
        table = []
        for cell in cells:
            guard(); group = groups[cell["phy_key"]]; records = []
            for trial in range(TRIALS):
                guard()
                row, _ = execute_trial(group, cell["snr_db"], trial, runtime, ledger, identity)
                records.append(row)
            value = aggregate(group, cell["snr_db"], records)
            point_path = out / "points" / (digest(cell_key(cell)) + ".json")
            if point_path.exists():
                require(json.loads(point_path.read_text()) == value, "Complete point reuse changed")
            else:
                save(point_path, value)
            table.append(dict(row=value, closure=pin(point_path)))
            with (out / "progress.jsonl").open("a") as f:
                elapsed = time.monotonic() - started
                f.write(canonical(dict(points_complete=len(table), ledger=ledger.snapshot(), elapsed_seconds=elapsed,
                    estimated_remaining_seconds=elapsed / (len(table) * TRIALS) * (CAP - len(table) * TRIALS),
                    estimate_from_registered_trials_only=True)) + "\n")
        guard(); validate(registration)
        require(ledger.snapshot() == dict(reserved=CAP, complete=CAP, unresolved=0, cap=CAP), "Full supplement ledger incomplete")
        result = dict(status="COMPLETE_R0_BLER_SUPPLEMENT", registration=registration,
            rows=[r["row"] for r in table], point_closures=[r["closure"] for r in table],
            ledger=ledger.snapshot(), ledger_sha256=sha(out / "trials.sqlite"), actual_runtime=pin(out / "actual_runtime.json"),
            historical_points_rerun=0, header_trials=0, model_calls=0, actual_elapsed_seconds=time.monotonic() - started,
            no_R1_automatic_successor=True, applicability="Pre-screen random-payload approximation, not image-level BLER or a strict quality bound")
        return save(out / "completion.json", result)
    except BaseException:
        save(out / ("failure_" + str(time.time_ns()) + ".json"), dict(status="FAILED_STOP_NO_AUTOMATIC_RETRY",
            traceback=traceback.format_exc(), ledger=ledger.snapshot(), actual_elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        ledger.close()


def run(registration, recovery=None):
    """One parent/child owner with actual wait. Never creates a persistent scheduler."""
    import fcntl
    reg, _, _, _ = validate(registration); cfg = reg["runtime"]
    out = Path(cfg["output_directory"])
    if recovery is None:
        require(not out.exists(), "Existing attempt requires explicit diagnostic recovery, not automatic relaunch")
    else:
        approval = readpin(recovery)
        require(approval["status"] == "REVIEWED_COMPLETE_ONLY_RESUME" and approval["registration"] == registration
            and approval["replay_unresolved"] is False and out.is_dir() and not (out / "completion.json").exists(),
            "Resume requires explicit reviewed unchanged registration; unresolved calls cannot replay")
        prior_wait = readpin(approval["prior_actual_wait"])
        require(prior_wait["actual_wait"] is True, "Prior actual child wait required")
    timing_lock = timing_shared_lock(cfg["timing_v6_directory"], fcntl)
    if timing_lock is None:
        return dict(status="WAIT_TIMING_V6_OWNER_LOCK", scientific_calls=0, dispatched=False)
    if not timing_idle(cfg["timing_v6_directory"], cfg["timing_v6_run_directory"]):
        timing_lock.close()
        return dict(status="WAIT_TIMING_V6", scientific_calls=0, dispatched=False)
    out.mkdir(parents=True, exist_ok=recovery is not None)
    attempt = out if recovery is None else out / ("reviewed_resume_" + str(time.time_ns()))
    if recovery is not None:
        attempt.mkdir()
        save(attempt / "reviewed_recovery.json", recovery)
    owner_lock = (out / "owner.lock").open("a+")
    fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic(); child = None
    try:
        # A resumed journal is inspected before even constructing a scientific child.
        if recovery is not None:
            checked = Ledger(out / "trials.sqlite", registration["sha256"]); checked.close()
        require(set(cfg["cpu_affinity"]) <= os.sched_getaffinity(0), "Assigned CPU cores unavailable")
        seconds = min(cfg["max_seconds"], cfg["deadline_unix"] - time.time())
        require(seconds > 0, "Expired supplement deadline")
        env = os.environ.copy(); env.update({key:"2" for key in THREAD_ENV})
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        argv = [cfg["python"], "-B", "-u", str(Path(__file__).absolute()), "_worker", "--registration", registration["path"],
            "--registration-sha256", registration["sha256"], "--owner-pid", str(os.getpid())]
        def affinity():
            os.sched_setaffinity(0, set(cfg["cpu_affinity"]))
        with (attempt / "child.log").open("x") as log:
            child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                preexec_fn=affinity, pass_fds=(timing_lock.fileno(),))
            save(attempt / "launch.json", dict(owner_pid=os.getpid(), child_pid=child.pid, child_start_ticks=Path(f"/proc/{child.pid}/stat").read_text().split()[21],
                registration=registration, argv=argv, started_unix=time.time(), cpu_affinity=cfg["cpu_affinity"]))
            try:
                code = child.wait(timeout=seconds); timeout = False
            except subprocess.TimeoutExpired:
                timeout = True; child.terminate()
                try: code = child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); code = child.wait()
        waited = dict(actual_wait=True, returncode=code, timeout=timeout, elapsed_seconds=time.monotonic() - started,
            scientific_automatic_retry=False)
        save(attempt / "actual_wait.json", waited)
        require(code == 0 and not timeout, "Supplement child failed; inspect immutable ledger before any recovery")
        require((out / "completion.json").is_file(), "Child exit0 lacks actual completion")
        return dict(status="COMPLETE", actual_wait=waited, completion=pin(out / "completion.json"))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        owner_lock.close()
        timing_lock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("validate", "run", "resume", "_worker"))
    p.add_argument("--registration", required=True); p.add_argument("--registration-sha256", required=True)
    p.add_argument("--owner-pid", type=int)
    p.add_argument("--recovery"); p.add_argument("--recovery-sha256")
    a = p.parse_args(); registration = dict(path=str(Path(a.registration).absolute()), sha256=a.registration_sha256)
    if a.command == "validate":
        reg, raw, groups, cells = validate(registration)
        print(canonical(dict(status="VALIDATED_METADATA_ONLY", points=len(cells), real_packet_calls=0)))
    elif a.command in ("run", "resume"):
        if a.command == "resume":
            require(a.recovery and a.recovery_sha256, "Explicit reviewed resume pin required")
        print(canonical(run(registration, dict(path=str(Path(a.recovery).absolute()), sha256=a.recovery_sha256)
            if a.command == "resume" else None)))
    else:
        require(a.owner_pid is not None, "Worker requires its actual owner PID")
        print(canonical(work(registration, a.owner_pid)))
