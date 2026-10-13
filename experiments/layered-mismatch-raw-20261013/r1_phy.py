"""Finite registered R1 physical runner; import/validate never constructs PHY.

There is deliberately no register or nominate command. A future reviewed
registration and its exact SHA are required to run. Current preparation tests
inject synthetic runtimes; no real packet qualification or R1 launch is implied.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import contract
import grouped_wire as wire

SCHEMA = "LAYERED_RAW_R1_PHY_REGISTRATION_V1"
RAW_REQUEST_SHA = "9e587fcec3be465d8d6c43a7123400635441918befbbd45a6339f0f208af2faf"
CAL_REQUEST_SHA = "39bc1c524450ecd981a5c2dafdeb5c898bde06b2723e24f08da3efdf711354ca"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
require = contract.require
digest = contract.digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(path):
    return dict(path=str(Path(path).absolute()), sha256=sha(path))


def checkpin(value):
    p = Path(value["path"])
    require(p.is_absolute() and p.is_file() and not p.is_symlink()
            and sha(p) == value["sha256"], "Changed/missing bound file: " + str(p))
    if "bytes" in value:
        require(p.stat().st_size == value["bytes"], "Bound byte count differs")
    return p


def readpin(value):
    return json.loads(checkpin(value).read_text(encoding="utf-8-sig"))


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8", newline="\n") as f:
        f.write(contract.canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return pin(p)


def source_scales(record):
    """Read authenticated original-calibration uint8 pixels/int64 680 tokens."""
    cp = readpin(record["checkpoint"])
    archive = checkpin(record["archive"])
    require(cp["status"] == "H_FULL1000_SOURCE_ASSET_READY"
        and cp["original_calibration_index"] == cp["source_index"] == record["source_index"]
        and all(cp[key] == record[key] for key in ("source_id", "tokens_sha256", "preprocessing_id"))
        and cp["outputs"][cp["archive"]] == record["archive"]["sha256"],
        "Original calibration source checkpoint identity differs")
    with np.load(archive, allow_pickle=False) as z:
        require(set(z.files) == {"tokens", "pixels"}, "Calibration source NPZ schema differs")
        tokens, pixels = z["tokens"], z["pixels"]
    require(tokens.dtype == np.dtype("int64") and tokens.shape == (680,)
            and ((tokens >= 0) & (tokens < 4096)).all(), "Actual original680 tokens required")
    require(pixels.dtype == np.dtype("uint8") and pixels.shape == (3, 256, 256), "Original uint8 CHW crop required")
    require(hashlib.sha256(b"int64:680\0" + tokens.astype("<i8", copy=False).tobytes()).hexdigest()
            == record["tokens_sha256"], "Actual calibration tokens hash differs")
    require(hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
            == record["preprocessing_id"], "Actual calibration pixels hash differs")
    return [tokens[wire.PREFIX[i]:wire.PREFIX[i + 1]].copy() for i in range(10)]


def nomination_groups(rows, profiles):
    """Dedupe physical transmission, retain every arm/receiver logical identity."""
    contract.validate_shortlist(rows)
    mapped = {p["profile_id"]: p for p in profiles}
    require(len(mapped) == len(profiles), "Ambiguous public IDs")
    by_physical = {}
    for index, row in enumerate(rows):
        p = mapped[row["profile_id"]]
        require(row["physical_key"] == p["wire_key"], "Nomination is not the exact public wire")
        require(row["candidate_id"] == p["candidate_id"], "Physical aliases must use the catalogue canonical candidate")
        require((p["G"] == 1) == (row["arm"] in ("S", "S_nom", "S_back3")), "Arm/wire family mismatch")
        require(p["G"] == 2 or row["profile_id"] < 433, "Single baselines retain only raw433")
        if p["G"] == 2:
            require(row["arm"] in p["family_memberships"], "U/EEP nomination is outside its registered family")
        key = (row["snr_hat"], row["physical_key"])
        value = by_physical.setdefault(key, dict(snr_hat=row["snr_hat"], physical_key=row["physical_key"],
            candidate_id=row["candidate_id"], profile_id=row["profile_id"], G=p["G"], logical_receivers=[]))
        require(value["profile_id"] == row["profile_id"], "Same physical key maps to different paid IDs")
        for rule in row["receiver_variants"]:
            existing = [v for v in value["logical_receivers"] if v["rule"] == rule]
            if existing:
                view = existing[0]
            else:
                view = dict(candidate_id=row["candidate_id"], rule=rule, arms=[], nomination_indices=[])
                value["logical_receivers"].append(view)
            view["arms"] = sorted(set(view["arms"] + [row["arm"]]))
            view["nomination_indices"].append(index)
    return [by_physical[key] for key in sorted(by_physical)]


def validate_registration(registration_pin):
    """Pure bound-file/metadata validation; no source arrays or PHY construction."""
    reg = readpin(registration_pin)
    require(reg["schema"] == SCHEMA and reg["status"] == "REGISTERED_R1_PHY_V1", "Real reviewed R1 registration required")
    require(reg["plan_id"] == contract.PLAN_ID and reg["caps"] == contract.R1_CAPS
            and reg["namespace"] == contract.NAMESPACE and reg["automatic_successor"] is False,
            "Frozen finite R1 scope differs")
    require(reg["training_updates"] == 0 and reg["historical_reuse_admitted"] is False,
            "No training or cross-namespace historical PHY reuse")
    pins = reg["pins"]
    required = {"nominations", "cal200_sources", "grouped_catalogue", "raw_runtime_request",
                "calibration_request", "mismatch_core", "mismatch_plan", "r1_phy", "grouped_wire", "contract"}
    require(required <= set(pins), "Missing explicit execution bindings")
    for value in pins.values():
        checkpin(value)
    for key, path in (("r1_phy", __file__), ("grouped_wire", wire.__file__), ("contract", contract.__file__)):
        require(Path(pins[key]["path"]).resolve() == Path(path).resolve(), "Imported bound entry differs: " + key)
    require(pins["raw_runtime_request"]["sha256"] == RAW_REQUEST_SHA
            and pins["calibration_request"]["sha256"] == CAL_REQUEST_SHA, "Historical authoritative request changed")
    raw = readpin(pins["raw_runtime_request"])
    for value in raw["raw_runtime_binding"]["files"]:
        checkpin(value)
    source_records = readpin(pins["cal200_sources"])["records"]
    original = readpin(pins["calibration_request"])["records"]
    expected = contract.select_cal200(original)
    require(len(source_records) == 200 and [r["selection_slot"] for r in source_records] == list(range(200)),
            "Exact ordered cal200 population required")
    for row, selected in zip(source_records, expected):
        require(all(row[key] == selected[key] for key in selected), "cal200 does not match frozen original1000 selection")
        checkpin(row["archive"])
        checkpin(row["checkpoint"])
    catalogue = readpin(pins["grouped_catalogue"])
    cat = wire.Catalogue(catalogue["profiles"])
    old = json.loads(Path(raw["raw_runtime_binding"]["roles"]["catalogue"]).read_bytes())["profiles"]
    require(catalogue["profiles"][:433] == old and len(old) == 433, "Original raw433 bytes/IDs changed")
    require(cat.digest == reg["grouped_catalogue_digest"]
            and wire.identity(old) == reg["single_catalogue_digest"], "Both explicit RX domains must match")
    nominations = readpin(pins["nominations"])
    require(nominations["status"] == "FROZEN_R1_NOMINATIONS" and nominations["selection_uses_holdout"] is False,
            "Real frozen R1 shortlist is missing")
    groups = nomination_groups(nominations["rows"], catalogue["profiles"])
    policy = readpin(raw["input_closure"]["original_policy"])
    require(policy["status"] == "POLICIES_FROZEN_ON_CALIBRATION1000", "Original single policy authority changed")
    winners = {int(w["snr_db"]): w["candidate_id"] for w in policy["winners"] if w["family"] == "PARTIAL"}
    for row in nominations["rows"]:
        if row["arm"] in ("S_nom", "S_back3"):
            selected_snr = row["snr_hat"] - (3 if row["arm"] == "S_back3" else 0)
            require(winners[selected_snr] in old[row["profile_id"]]["alias_candidate_ids"], "Frozen nominal/backoff anchor changed")
    expected_physical = len(groups) * 200 * 3 * 3
    expected_logical = sum(len(g["logical_receivers"]) for g in groups) * 200 * 3 * 3
    require(expected_physical <= 36000 and expected_logical <= 46800
            and reg["expected_physical_frames"] == expected_physical
            and reg["expected_logical_var_images"] == expected_logical, "Nominated grid/caps differ")
    runtime = reg["runtime"]
    require(runtime["python"] == raw["CPU_python"] and runtime["threads"] == 2
            and len(set(runtime["cpu_affinity"])) == 2 and runtime["CUDA_VISIBLE_DEVICES"] == ""
            and runtime["max_seconds"] > 0, "Pinned CPU-only two-thread runtime required")
    return reg, source_records, catalogue, groups, raw


class Journal:
    """Immutable reservations and results; unknown failed attempts cannot replay."""
    def __init__(self, path, frame_cap=36000, packet_cap=108000):
        self.path = Path(path)
        require(not self.path.exists(), "Fresh execution journal required; failure/relaunch needs separate review")
        self.path.mkdir(parents=True)
        require(0 < frame_cap <= 36000 and 0 < packet_cap <= 108000, "Journal cap exceeds R1")
        self.frame_cap, self.packet_cap = frame_cap, packet_cap
        self.frames, self.packets, self.active = {}, {}, None
        self._frame_packet_keys = {}
        self._packet_decodes_reserved = 0

    def reserve_frame(self, identity):
        key = digest(identity)
        if key in self.frames:
            saved = self.frames[key]
            require("result" in saved, "Unresolved physical reservation cannot replay")
            result = readpin(saved["result"])
            require(result["input_identity"] == identity, "Complete reuse identity differs")
            return key, result
        require(self.active is None, "Unresolved physical reservation blocks new calls")
        require(len(self.frames) < self.frame_cap, "Physical frame allowance exhausted")
        reservation = save(self.path / "frames" / (key + ".reserved.json"), dict(input_identity=identity, state="RESERVED"))
        self.frames[key] = dict(reservation=reservation)
        self._frame_packet_keys[key] = []
        self.active = key
        return key, None

    def complete_frame(self, key, result):
        require(key == self.active and key in self.frames and "result" not in self.frames[key], "Unknown/already-closed frame")
        require(result["status"] == "COMPLETE" and digest(result["input_identity"]) == key,
                "Completed frame identity/status differs from reservation")
        require(all("result" in self.packets[k] for k in self._frame_packet_keys[key]),
                "Unresolved packet cannot become a complete physical frame")
        result_pin = save(self.path / "frames" / (key + ".complete.json"), result)
        self.frames[key]["result"] = result_pin
        self.active = None
        return result_pin

    def _reserve_packet(self, key, request, count_decode):
        require(self.active is not None and key not in self.packets, "Packet lacks unique active frame")
        if count_decode:
            require(self._packet_decodes_reserved < self.packet_cap, "Packet decode cap exhausted")
        self.packets[key] = dict(counts_decode=count_decode, frame=self.active,
            reservation=save(self.path / "packets" / (key + ".reserved.json"), dict(frame=self.active, request=request)))
        self._frame_packet_keys[self.active].append(key)
        self._packet_decodes_reserved += int(count_decode)

    def call(self, event, request, callback):
        """Original raw433 Receiver charge signature; one real RX callback only."""
        key = digest([self.active, event])
        self._reserve_packet(key, dict(event=event, request=request), True)
        result = callback()  # A failure leaves the immutable reservation unresolved.
        self.packets[key]["result"] = save(self.path / "packets" / (key + ".complete.json"), result)
        return result

    def grouped_hook(self, event):
        stable = {k: v for k, v in event.items() if k not in ("stage", "exception_type")}
        key = digest([self.active, "grouped", stable])
        if event["stage"] == "begin":
            self._reserve_packet(key, stable, event["direction"] == "rx")
        elif event["stage"] == "done":
            require(key in self.packets and "result" not in self.packets[key], "Group packet completion lacks reservation")
            self.packets[key]["result"] = save(self.path / "packets" / (key + ".complete.json"), event)
        elif event["stage"] == "failed":
            require(key in self.packets, "Failure lacks group packet reservation")
            save(self.path / "packets" / (key + ".failure.json"), event)
        else:
            raise ValueError("Unknown grouped packet ledger stage")

    def snapshot(self):
        return dict(physical_reserved=len(self.frames), physical_complete=sum("result" in x for x in self.frames.values()),
            packet_decodes_reserved=sum(x["counts_decode"] for x in self.packets.values()),
            packet_decodes_complete=sum(x["counts_decode"] and "result" in x for x in self.packets.values()),
            unresolved_packets=sum("result" not in x for x in self.packets.values()),
            physical_cap=self.frame_cap, packet_decode_cap=self.packet_cap)


def receiver_views(actual, kind, variants):
    """Derive DROP from one unchanged single KEEP event, never decode again."""
    if kind == "single":
        base = copy.deepcopy(actual["receiver_state"])
        accept = bool(actual["header_ok"] and actual["body_crc_accept"])
        return {rule: (copy.deepcopy(base) if rule == "KEEP" or accept else wire.gray_state()) for rule in variants}
    require(kind == "grouped" and set(variants) == {"DROP"}, "Group first-round receiver is always DROP")
    return {"DROP": copy.deepcopy(actual["state"])}


def input_identity(source, nomination, true_snr, seed, runtime_identity):
    slot, hat = source["selection_slot"], nomination["snr_hat"]
    return dict(schema="LAYERED_RAW_CAL200_PHY_INPUT_V1", source=copy.deepcopy(source),
        snr_hat=hat, snr_true=true_snr, noise_seed=seed, noise_key=contract.channel_key(source["source_id"], hat, seed, "cal200"),
        public_counter=contract.public_counter(slot, hat, seed, "cal200"), physical_key=nomination["physical_key"],
        profile_id=nomination["profile_id"], G=nomination["G"], runtime_identity=runtime_identity,
        body_session="actual-body" if nomination["G"] == 1 else wire.BODY_SESSION)


def execute_frame(source, scales, nomination, true_snr, seed, runtime, journal, runtime_identity):
    require(true_snr in contract.true_snrs(nomination["snr_hat"]), "Outside six-condition grid")
    identity = input_identity(source, nomination, true_snr, seed, runtime_identity)
    key, reused = journal.reserve_frame(identity)
    if reused is not None:
        return journal.frames[key]["result"], reused, True
    counter = identity["public_counter"]
    wave, transmission, payloads = runtime.transmit(nomination, scales, counter)
    wave = np.asarray(wave)
    require(wave.dtype == np.dtype("float64") and wave.shape == (1024, 2) and np.isfinite(wave).all(), "Actual paid waveform differs")
    require(transmission["waveform_sha256"] == wire.array_sha(wave)
            and transmission.get("actual_E", transmission.get("E")) == float(np.square(wave).sum()),
            "Actual TX waveform/energy receipt differs")
    noise = contract.standard_noise(source["source_id"], nomination["snr_hat"], seed, "cal200")
    received = wave + noise * 10 ** (-float(true_snr) / 20)
    # Only received samples, true SNR and public counter enter the actual RX.
    actual = runtime.receive(nomination["G"], received, float(true_snr), counter, key)
    require(actual["received_sha256"] == wire.array_sha(received), "Actual RX consumed different observations")
    variants = sorted({v["rule"] for v in nomination["logical_receivers"]})
    views = receiver_views(actual, "single" if nomination["G"] == 1 else "grouped", variants)
    result = dict(schema="LAYERED_RAW_CAL200_ACTUAL_PHY_V1", status="COMPLETE", input_identity=identity,
        observation_identity=dict(waveform_sha256=wire.array_sha(wave), standard_noise_sha256=wire.array_sha(noise),
            received_sha256=wire.array_sha(received), payload_sha256=[wire.array_sha(p) for p in payloads]),
        transmission=transmission, actual_RX=actual, receiver_views=views,
        view_sha256={name: digest(state) for name, state in views.items()},
        actual_frame_energy=float(np.square(wave).sum()), actual_frame_mean_Es=float(np.square(wave).sum()) / 1024,
        channel_noise_variance_per_component=10 ** (-float(true_snr) / 10),
        quality_scoring="NOT_RUN", reconstruction="NOT_RUN")
    result["post_receiver_truth"] = runtime.audit(actual, nomination, payloads)
    return journal.complete_frame(key, result), result, False


def load_module(value, name):
    path = checkpin(value)
    require(name not in sys.modules, "Fresh science process/module name required: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RealRuntime:
    """Lazily constructed only inside explicit run with real reviewed pins."""
    def __init__(self, reg, raw_request, catalogue, journal, guard):
        load_module(reg["pins"]["mismatch_plan"], "mismatch_plan")
        core = load_module(reg["pins"]["mismatch_core"], "_layered_r1_original_raw_core")
        self.raw = core.create_runtime(raw_request["raw_runtime_binding"], journal, guard)
        self.cat = wire.Catalogue(catalogue["profiles"])
        self.grouped = wire.ActualPhyBackend(self.raw.backend, self.raw.legacy,
            self.raw.packet, self.raw.header, journal.grouped_hook)
        self.grouped.verify_catalogue(self.cat)
        require(self.raw.cat.digest == reg["single_catalogue_digest"] and self.cat.digest == reg["grouped_catalogue_digest"],
                "Distinct full single/grouped RX domains differ")
        self.raw.cat.verify()

    def transmit(self, nomination, scales, counter):
        pid = nomination["profile_id"]
        if nomination["G"] == 1:
            profile = self.raw.cat.entry(pid)
            payload = self.raw.original.serialize_raw(scales, profile)
            wave, ledger = self.raw.rx.transmit_frame(self.raw.cat, self.raw.backend, self.raw.legacy,
                self.raw.header, pid, payload, counter)
            return wave, ledger, [payload]
        profile = self.cat.row(pid)
        payloads = wire.serialize_groups(scales, profile)
        wave, ledger = wire.transmit(pid, payloads, counter, self.cat, self.grouped)
        return wave, ledger, payloads

    def receive(self, group_count, received, true_snr, counter, audit_tag):
        if group_count == 1:
            actual = self.raw.receiver.receive(received, true_snr, counter,
                phase="raw_holdout", event_prefix="LAYERED_RAW_CAL200_V1/" + audit_tag)
            require(all(actual[k] == v for k, v in self.raw.original.present_actual(
                actual["header"], actual["body"], self.raw.cat).items()), "Raw actual token presentation differs")
            return actual
        return wire.receive(received, true_snr, counter, self.cat, self.grouped)

    def audit(self, actual, nomination, payloads):
        if nomination["G"] == 2:
            return wire.audit_after_receiver(actual, nomination["profile_id"], payloads)
        header_correct = actual["header_ok"] and actual["rx_profile_id"] == nomination["profile_id"]
        return dict(header_correct=bool(header_correct),
            header_false_accept=bool(actual["header_ok"] and not header_correct),
            body_comparison_available=bool(header_correct),
            body_false_accept=(bool(actual["body_crc_accept"] and not np.array_equal(
                actual["body"]["hard_payload"], payloads[0])) if header_correct else None))


def run(registration_pin):
    reg, records, catalogue, nominations, raw = validate_registration(registration_pin)
    cfg = reg["runtime"]
    require(sys.platform.startswith("linux") and str(Path(sys.executable).absolute()) == cfg["python"], "Exact original LDPC Python required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "2" for k in THREAD_ENV),
            "Launch must set CUDA hidden and frozen two-thread environment")
    require(set(cfg["cpu_affinity"]) <= os.sched_getaffinity(0), "Requested CPU affinity unavailable")
    os.sched_setaffinity(0, set(cfg["cpu_affinity"]))
    start = time.monotonic()
    out = Path(cfg["output_directory"])
    require(out.is_absolute() and not out.exists(), "Fresh absolute finite output directory required")
    out.mkdir(parents=True)
    journal = Journal(out / "journal", reg["expected_physical_frames"], 108000)
    def guard():
        require(time.time() < cfg["deadline_unix"] and time.monotonic() - start < cfg["max_seconds"]
            and not Path(cfg["stop_file"]).exists(), "Registered STOP/deadline reached")
    try:
        guard()
        runtime = RealRuntime(reg, raw, catalogue, journal, guard)
        runtime_identity = dict(registration=registration_pin, source_code_pins=reg["pins"],
            raw_backend=runtime.raw.backend.identity, raw64_backend=runtime.raw.backend.identity64,
            single_catalogue=runtime.raw.cat.digest, aliases=runtime.raw.cat.aliases_digest,
            grouped_catalogue=runtime.cat.digest, phy_threads=2, CPU_python=sys.executable)
        logical, physical = [], []
        for record in records:
            guard()
            scales = source_scales(record)
            for nomination in nominations:
                for true_snr in contract.true_snrs(nomination["snr_hat"]):
                    for seed in contract.CAL_SEEDS:
                        guard()
                        frame_pin, frame, reused = execute_frame(record, scales, nomination, true_snr,
                            seed, runtime, journal, runtime_identity)
                        physical.append(frame_pin)
                        for view in nomination["logical_receivers"]:
                            logical.append(dict(source_id=record["source_id"], source_index=record["source_index"],
                                selection_slot=record["selection_slot"], snr_hat=nomination["snr_hat"], snr_true=true_snr,
                                noise_seed=seed, physical_key=nomination["physical_key"], profile_id=nomination["profile_id"],
                                public_counter=frame["input_identity"]["public_counter"], **view,
                                actual_RX=frame_pin, receiver_state_sha256=frame["view_sha256"][view["rule"]],
                                reused_complete_identity=reused))
        guard()
        snapshot = journal.snapshot()
        require(len(physical) == snapshot["physical_complete"] == reg["expected_physical_frames"]
            and len(logical) == reg["expected_logical_var_images"] and snapshot["unresolved_packets"] == 0
            and snapshot["packet_decodes_complete"] == snapshot["packet_decodes_reserved"], "R1 physical grid/ledger did not close")
        validate_registration(registration_pin)
        return save(out / "completion.json", dict(status="COMPLETE_R1_PHY_ONLY", registration=registration_pin,
            physical_frames=save(out / "physical_frames.json", physical), logical_frames=save(out / "logical_frames.json", logical),
            ledger=snapshot, actual_elapsed_seconds=time.monotonic() - start, reconstruction="NOT_RUN",
            metrics="NOT_RUN", policy_selection="NOT_RUN", R2="NOT_RUN", automatic_successor=False))
    except BaseException:
        save(out / "failure.json", dict(status="FAILED_PRESERVE_NO_RETRY", traceback=traceback.format_exc(),
            journal=journal.snapshot(), actual_elapsed_seconds=time.monotonic() - start))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--registration", required=True)
    parser.add_argument("--registration-sha256", required=True)
    args = parser.parse_args()
    p = dict(path=str(Path(args.registration).absolute()), sha256=args.registration_sha256)
    if args.command == "validate":
        reg, records, catalogue, rows, raw = validate_registration(p)
        print(json.dumps(dict(status="VALIDATED_METADATA_ONLY_NOT_LAUNCHED", sources=len(records),
            physical_candidates=len(rows), actual_packet_calls=0, visual_calls=0)))
    else:
        print(json.dumps(run(p)))
