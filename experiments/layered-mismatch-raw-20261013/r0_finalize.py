"""Read-only R0 closure audit and frozen nomination arithmetic; never runs R1.

Call only after the two registered supplementary owners have actually exited.
There are no model, channel, coding, bootstrap, scheduler or recovery calls.
Scientific ledgers and all archived scientific files are opened read-only.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
import r0_bler as bler
import r0_clean as clean
import r0_screening as screening

require = bler.require
sha = bler.sha
pin = bler.pin
checkpin = bler.checkpin
readpin = bler.readpin
digest = bler.digest
SCHEMA = "LAYERED_RAW_R0_AUTHENTICATED_NOMINATION_V1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(value):
    with checkpin(value).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def actual_success(value):
    require(value.get("actual_wait") is True and type(value.get("returncode")) is int
            and value["returncode"] == 0 and not any(value.get(k, False)
            for k in ("timeout", "timed_out", "terminated", "killed")),
            "Actual successful child wait required; a launch or completion alone is insufficient")
    require(math.isfinite(value["elapsed_seconds"]) and value["elapsed_seconds"] >= 0,
            "Actual elapsed time missing")


def same_pin(a, b):
    return a["path"] == b["path"] and a["sha256"] == b["sha256"]


def readonly_db(path):
    p = Path(path).resolve()
    require(p.is_file() and not p.is_symlink(), "Missing actual ledger")
    db = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    require(db.execute("PRAGMA quick_check").fetchone() == ("ok",), "Corrupt ledger")
    return db


def ensure_dead(pid, start_ticks=None):
    if not sys.platform.startswith("linux"):
        return
    p = Path("/proc") / str(pid)
    if p.exists():
        stat = (p / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        require(fields[0] == "Z" or (start_ticks is not None and str(start_ticks) != fields[19]),
                "Recorded scientific child remains alive")


def checked_trial(identity, result, group, snr, trial, runtime_identity):
    expected = dict(namespace=bler.NAMESPACE, phy_key=group["phy_key"], layout_id=group["layout"]["layout_id"],
                    snr_db=snr, trial=trial, session=bler.BODY_SESSION, group=0, runtime_identity=runtime_identity)
    require(identity == expected and result["trial_identity"] == expected and result["status"] == "COMPLETE",
            "Actual trial identity changed")
    for key in ("correct", "rejected", "undetected", "payload_equal", "crc_accept"):
        require(type(result[key]) is bool, "Outcome is not a recorded boolean")
    equal, accepted = result["payload_equal"], result["crc_accept"]
    require(result["correct"] == (equal and accepted) and result["rejected"] == (not accepted)
            and result["undetected"] == (accepted and not equal), "Scientific outcomes not partitioned")
    require(result["actual_packet_encodes"] == result["actual_packet_decodes"] == 1
            and result["header_calls"] == result["neural_model_calls"] == 0, "Unexpected scientific calls")
    for key in ("payload_sha256", "standard_noise_sha256", "waveform_sha256", "received_sha256",
                "hard_payload_sha256", "decoded_sha256"):
        value = result[key]
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                "Missing actual array fingerprint")
    base = [bler.NAMESPACE, group["phy_key"], group["layout"]["layout_id"], snr, trial, "public_counter"]
    counter = int.from_bytes(hashlib.sha256(bler.canonical(base).encode()).digest()[:8], "little") & ((1 << 63) - 1)
    require(result["public_counter"] == counter and result["symbols"] == group["symbols"]
            and math.isfinite(result["actual_E"]) and result["actual_E"] > 0
            and result["observed_mean_Es"] == result["actual_E"] / group["symbols"]
            and result["noise_variance_per_component"] == 10 ** (-snr / 10), "Counter/energy/noise identity changed")


def audit_bler(registration):
    reg, _, groups, cells = bler.validate(registration)
    out = Path(reg["runtime"]["output_directory"])
    wait = read(out / "actual_wait.json"); actual_success(wait)
    launch = read(out / "launch.json")
    require(same_pin(launch["registration"], registration), "BLER launch registration differs")
    ensure_dead(launch["child_pid"], launch["child_start_ticks"])
    require(not list(out.glob("failure_*.json")), "Failure evidence requires separate reviewed closure")
    completion = read(out / "completion.json")
    require(completion["status"] == "COMPLETE_R0_BLER_SUPPLEMENT"
            and same_pin(completion["registration"], registration)
            and completion["ledger"] == dict(reserved=bler.CAP, complete=bler.CAP, unresolved=0, cap=bler.CAP)
            and completion["historical_points_rerun"] == completion["header_trials"] == completion["model_calls"] == 0,
            "Incomplete finite BLER closure")
    require(len(completion["rows"]) == len(completion["point_closures"]) == len(cells), "Missing body cells")
    runtime = readpin(completion["actual_runtime"])
    require(same_pin(runtime["registration"], registration), "Actual runtime belongs to another registration")
    dbpath = out / "trials.sqlite"
    require(sha(dbpath) == completion["ledger_sha256"], "Actual ledger SHA differs")
    db = readonly_db(dbpath)
    try:
        require(db.execute("SELECT registration,cap FROM meta WHERE id=1").fetchone() == (registration["sha256"], bler.CAP),
                "Ledger reservation scope differs")
        require(dict(db.execute("SELECT status,count(*) FROM trials GROUP BY status")) == {"COMPLETE": bler.CAP},
                "Ledger incomplete, duplicated, failed or reserved")
        cursor = db.execute("SELECT id,input,status,result FROM trials ORDER BY json_extract(input,'$.phy_key'),"
                            "json_extract(input,'$.snr_db'),json_extract(input,'$.trial')")
        for cell, recorded, closed in zip(cells, completion["rows"], completion["point_closures"]):
            group = groups[cell["phy_key"]]; records = []
            for trial in range(bler.TRIALS):
                entry = next(cursor, None); require(entry is not None, "Missing actual trial")
                key, encoded, status, data = entry
                identity, result = json.loads(encoded), json.loads(data)
                require(key == digest(identity) and encoded == bler.canonical(identity) and status == "COMPLETE", "Unsealed trial row")
                checked_trial(identity, result, group, cell["snr_db"], trial, runtime)
                records.append(result)
            # Count aggregation and the registered Wilson formula only. No RNG,
            # signal, model, or inference is reconstructed during this audit.
            rebuilt = bler.aggregate(group, cell["snr_db"], records)
            require(rebuilt == recorded == readpin(closed), "Point result disagrees with its 64 actual ledger records")
        require(next(cursor, None) is None, "Unregistered extra trial")
    finally:
        db.close()
    require(sha(dbpath) == completion["ledger_sha256"], "Ledger changed during read-only audit")
    return completion["rows"], dict(registration=registration, completion=pin(out / "completion.json"),
        actual_wait=pin(out / "actual_wait.json"), launch=pin(out / "launch.json"), ledger=pin(dbpath),
        cells=bler.CELLS, actual_body_trials=bler.CAP, header_calls=0,
        array_evidence="Recorded actual array hashes authenticated; waveforms are not regenerated")


def audit_clean_ledger(path, stage, caps, schedule, sources):
    db = readonly_db(path)
    bykind = collections.defaultdict(list)
    try:
        for identifier, kind, metadata, state in db.execute("SELECT id,kind,metadata,state FROM calls ORDER BY id"):
            require(kind in caps and state == "COMPLETE", "Unknown, failed or unresolved clean scientific call")
            value = json.loads(metadata); require(isinstance(value, dict), "Invalid call metadata")
            bykind[kind].append(value)
        require({k: len(v) for k, v in bykind.items()} == caps, "Actual clean ledger does not close registered caps")
    finally:
        db.close()
    expected = [r["cell_index"] for r in schedule]
    for kind in (("var_render",) if stage == "render" else ("reconstruction_dino", "image_score")):
        require([r["cell"] for r in bykind[kind]] == expected, "Missing/duplicated or reordered clean cell call")
    if stage == "render" and "prior_scale" in caps:
        expected_scales = [dict(source=cell["source_index"], scale=scale) for cell in schedule for scale in range(10)]
        require(bykind["prior_scale"] == expected_scales, "Actual prior source/scale sequence differs")
        require(bykind["decoder_forward"] == [dict(source=cell["source_index"]) for cell in schedule],
                "Actual decoder source sequence differs")
    if stage == "score":
        require([r["source_id"] for r in bykind["source_embedding"]] == [s["source_id"] for s in sources],
                "Source feature calls do not match fixed cal200")
        require([r["source_id"] for r in bykind["reconstruction_dino"]] == [s["source_id"] for s in schedule],
                "Reconstruction feature source mismatch")
    return dict(ledger=pin(path), counts=caps, unresolved=0)


def same_csv_row(parsed, recorded):
    require(set(parsed) == set(recorded), "CSV/score receipt columns differ")
    for key, value in recorded.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            require(float(parsed[key]) == value, "CSV value differs from actual per-cell score: " + key)
        else:
            require(parsed[key] == str(value), "CSV value differs from actual per-cell identity: " + key)


def audit_clean(registration):
    reg = clean.registration(checkpin(registration), registration["sha256"])
    out, tmp = Path(reg["output_root"]), Path(reg["runtime_root"])
    completion = read(out / "completion.json")
    require(completion["status"] == "COMPLETE_CLEAN5400" and same_pin(completion["registration"], registration)
            and completion["all_actual_exit_zero"] is True
            and completion["clean_VAR_actual"] == completion["reconstruction_DINO_actual"] == 5400
            and completion["source_DINO_actual"] == 200 and completion["complete_joint_terminal_domain"] == 8800
            and completion["selection_or_R1_started"] is False, "Clean supplement incomplete")
    require(all(completion[k] == 0 for k in ("extra_qualification_reconstructions", "training_updates", "channel_calls", "packet_calls", "tokenizer_calls")),
            "Additional unregistered scientific calls")
    require([s["stage"] for s in completion["stages"]] == ["render", "score"], "Incomplete or repeated clean stages")
    require(not (tmp / "owner_failure.json").exists(), "Clean owner has unresolved failure history")
    stages, evidence = {}, []
    for s in completion["stages"]:
        stage = s["stage"]; d = tmp / stage
        require(not (d / "failure.json").exists(), "Failed clean stage cannot be accepted")
        waited = read(d / "actual_wait.json"); actual_success(waited)
        require(waited == s["actual_wait"] == read(tmp / (stage + "_actual_wait.json")), "Actual wait receipts differ")
        launch = read(d / "launch.json")
        require(same_pin(launch["registration"], registration) and launch["stage"] == stage
                and launch["process"]["pid"] == waited["pid"], "Clean launch identity differs")
        ensure_dead(waited["pid"], launch["process"]["start_ticks"])
        done = readpin(s["completion"]); stages[stage] = done
        require(done["status"] == "COMPLETE" and done["stage"] == stage and done["cell_count"] == 5400,
                "Stage closure does not cover all 5400 cells")
        expected = dict(caps=reg["caps"][stage], reserved=reg["caps"][stage], completed=reg["caps"][stage], unresolved=0)
        require(done["counts"] == expected, "Clean stage count receipt incomplete")
        checkpin(done["resource_history"])
        result = audit_clean_ledger(d / "calls.sqlite", stage, reg["caps"][stage], reg["schedule"], reg["sources"])
        result.update(stage=stage, completion=s["completion"], actual_wait=pin(d / "actual_wait.json"), launch=pin(d / "launch.json"))
        evidence.append(result)
    require(stages["render"]["models"] == reg["models"] and stages["render"]["extra_qualification_calls"] == 0,
            "Actual render model identity differs")
    score = stages["score"]
    require(score["bootstrap_calls"] == 0 and score["source_embeddings_actual"] == 200
            and score["clean_quality"] == completion["clean_quality"], "Score boundary differs")
    metric = readpin(score["metric_identity"])
    require(metric["original_source"] == reg["metric_config"]["sources"]["metric_models.py"]
            and metric["numeric_flags"] == reg["numeric_flags"] and metric["CPU_slots"] == reg["CPU_slots"]
            and metric["source_feature_preparations_cap"] == 200 and metric["extra_metric_models"] == 0
            and metric["batch_size"] == 1 and metric["source_and_reconstruction_preprocessing_identical"] is True
            and metric["original_reference_and_score_code_unchanged"] is True, "Actual metric configuration differs")
    checkpin(metric["original_source"])
    frames, refs, quality = readpin(stages["render"]["frames"]), readpin(score["source_embeddings"]), csv_rows(score["clean_quality"])
    require(len(frames) == len(quality) == 5400 and len(refs) == 200, "Incomplete actual frame or feature grid")
    source_arrays = {}; reference = {}
    for s, ref in zip(reg["sources"], refs):
        pixels, tokens = clean.validate_source(s)
        source_arrays[s["source_id"]] = tokens
        require(ref["source_id"] == s["source_id"] and ref["preprocessing_id"] == s["preprocessing_id"]
                and ref["input_array_sha256"] == clean.array_sha(pixels.astype(np.float32) / 255)
                and ref["metric_identity"] == metric["metric_identity"]
                and ref["cache_key"] == clean.digest([s["preprocessing_id"], metric["metric_identity"]]), "Reference feature identity differs")
        with np.load(checkpin(ref["feature_archive"]), allow_pickle=False) as z:
            feature = z["feature"].copy()
        require(feature.dtype == np.float32 and feature.shape == (1, 1024) and np.isfinite(feature).all()
                and clean.array_sha(feature) == ref["feature_sha256"], "Actual source feature bytes differ")
        reference[s["source_id"]] = ref
    for cell, frame, row in zip(reg["schedule"], frames, quality):
        require({k: frame[k] for k in cell} == cell, "Actual clean grid differs from frozen schedule")
        source = reg["sources"][cell["selection_slot"]]
        require(frame["source_archive"] == source["archive"] and frame["source_preprocessing_id"] == source["preprocessing_id"]
                and frame["source_tokens_sha256"] == source["tokens_sha256"] and frame["model_identity"] == clean.digest(reg["models"])
                and frame["prefix_array_sha256"] == clean.array_sha(source_arrays[cell["source_id"]][:cell["T"]])
                and frame["conditioning"] == "unconditional_class1000" and frame["order"] == "raw_raster", "Clean prefix/model/source differs")
        with np.load(checkpin(frame["image_archive"]), allow_pickle=False) as z:
            image = z["image"].copy()
        clean.validate_image(image)
        require(clean.array_sha(image) == frame["image_array_sha256"], "Actual image bytes differ")
        require(read(tmp / "render/rows" / (f"{cell['cell_index']:04d}.json")) == frame, "Render per-cell receipt changed")
        actual = read(tmp / "score/rows" / (f"{cell['cell_index']:04d}.json"))
        same_csv_row(row, actual)
        require(all(actual[k] == frame[k] for k in ("cell_index", "selection_slot", "source_index", "source_id", "m", "K", "T",
                "source_preprocessing_id", "source_tokens_sha256", "prefix_array_sha256", "image_array_sha256", "model_identity"))
                and actual["metric_identity"] == metric["metric_identity"]
                and actual["source_feature_sha256"] == reference[cell["source_id"]]["feature_sha256"]
                and actual["image_path"] == frame["image_archive"]["path"] and actual["origin"] == "H800_NEW_CLEAN_RASTER",
                "Actual scoring provenance does not match the render")
        value = actual[screening.METRIC]
        require(math.isfinite(value) and -1.000001 <= value <= 1.000001, "Invalid DINO score")
    return quality, dict(registration=registration, completion=pin(out / "completion.json"), stages=evidence,
        frames=stages["render"]["frames"], source_embeddings=score["source_embeddings"], metric_identity=score["metric_identity"],
        clean_quality=score["clean_quality"], actual_clean_VAR=5400, actual_reconstruction_DINO=5400, actual_source_DINO=200,
        actual_archive_pixels_and_features_SHA_verified=True)


def finalize(bler_registration, clean_registration, output):
    output = Path(output)
    require(not output.exists(), "New independent finalization directory required")
    # Validate original arithmetic code before any ranking. Importing these
    # modules is metadata-only and does not construct their scientific classes.
    reg, _, _, _ = bler.validate(bler_registration)
    contract = readpin(reg["pins"]["screening_contract"])
    require(sha(screening.__file__) == contract["source_code_sha256"]
            and sha(screening.contract.__file__) == contract["contract_sha256"], "Frozen nomination code changed")
    relocated = readpin(reg["pins"]["server_relocation"])
    inputs = {r["role"]: dict(path=r["server_path"], sha256=r["sha256"]) for r in relocated["mapping"]}
    for value in inputs.values():
        checkpin(value)
    new_bler, bler_proof = audit_bler(bler_registration)
    new_clean, clean_proof = audit_clean(clean_registration)
    cr = readpin(clean_registration)
    require(same_pin(cr["source_manifest"], inputs["source"]) and same_pin(cr["historical_clean"], inputs["old_clean"])
            and same_pin(cr["missing_pairs"], inputs["clean_missing"]), "Supplement populations differ")
    catalogue = readpin(inputs["catalogue"]); sources = readpin(inputs["source"])["records"]
    historical = readpin(reg["pins"]["historical"]); historical_clean = csv_rows(inputs["old_clean"])
    reuse = readpin(inputs["reuse_audit"])
    require(reuse["historical_raw_intact_reuse"]["source_endpoint_pairs"] == 3400
            and reuse["original_cal1000_actual_arrays_verified"] is True
            and all(row["actual_token_equality"] == row["body_crc_accept"] == "True" for row in historical_clean),
            "Historical exact-clean reuse authority differs")
    header_pin = pin(Path(reg["pins"]["screening_contract"]["path"]).with_name("screening_historical_headers.json"))
    header_rows = readpin(header_pin)
    expected_headers = [x["row"] for x in readpin(inputs["coverage_report"])["matched_header_rows"]
                        if x["row"]["snr_db"] in screening.SELECTION_SNRS]
    require(header_rows == expected_headers and len(header_rows) == 3, "Historical headers differ; no header supplement authorized")
    gray = [r for r in csv_rows(inputs["gray"]) if r["state_id"] == "gray" and r["receiver"] == "gray"]
    require(len(catalogue["profiles"]) == 689 and len(sources) == len(gray) == 200
            and len(historical) == 672 and len(new_bler) == 1389 and len(historical_clean) == 3400 and len(new_clean) == 5400,
            "Authenticated finite closure is incomplete")
    nomination = screening.rank_and_nominate(catalogue["profiles"], [s["source_id"] for s in sources],
        historical + new_bler, header_rows, historical_clean, new_clean, gray, readpin(inputs["anchors"]))
    require(nomination["R1"] == nomination["R2"] == nomination["GO_decision"] == "NOT_RUN", "R0 cannot authorize further science")
    evidence = dict(schema=SCHEMA, status="R0_COMPLETE_AUTHENTICATED_R1_NOT_REGISTERED", inputs=inputs,
        screening_contract=reg["pins"]["screening_contract"], screening_source=pin(screening.__file__), headers=header_pin,
        BLER=bler_proof, clean=clean_proof, historical_clean_rows=3400, old_BLER_selection_points=672,
        full_body_selection_grid=2061, clean_source_state_grid=8800, gray_source_count=200,
        historical_clean_hardware="Historical 4090D exact-clean screening evidence, not current H800 reruns",
        proxy_scope="IID scrambled payload correctness plus conditional group independence; false acceptance quality is approximated, not a strict lower bound",
        no_new_science=True, bootstrap_calls=0, R1="NOT_REGISTERED_NOT_RUN", R2="NOT_REGISTERED_NOT_RUN")
    # No output is emitted until every scientific closure and arithmetic check
    # has passed. Existing scientific files are never changed.
    output.mkdir(parents=True)
    nomination_pin = bler.save(output / "nominations.json", nomination)
    bler.save(output / "closure_audit.json", dict(evidence, nominations=nomination_pin, finalizer=pin(__file__)))
    return dict(status=evidence["status"], nominations=nomination_pin, audit=pin(output / "closure_audit.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bler", "clean"):
        parser.add_argument("--" + name + "-registration", required=True)
        parser.add_argument("--" + name + "-registration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(bler.canonical(finalize(dict(path=args.bler_registration, sha256=args.bler_registration_sha256),
        dict(path=args.clean_registration, sha256=args.clean_registration_sha256), args.output)))
