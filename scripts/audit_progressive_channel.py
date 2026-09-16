#!/usr/bin/env python3
"""Audit trusted-state traces, complete budgets, all image scores and strong controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import lpips
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as functional
import yaml

from audit_single_scale_channel import crc_independent, require
from var_comm.entropy import arithmetic_decode, probability_cdf
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_artifacts, write_json

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
RAW_BITS = (1092, 768, 1200, 2028)


def number(bits):
    return int(np.asarray(bits, dtype=np.int64) @ (1 << np.arange(len(bits) - 1, -1, -1)))


def token_bits(tokens):
    return ((np.asarray(tokens, dtype=np.int64)[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.uint8).ravel()


def token_indices(bits):
    return np.asarray(bits).reshape(-1, 12) @ (1 << np.arange(11, -1, -1))


def divide_prefix(tokens, count):
    cuts = np.cumsum([size ** 2 for size in SIZES[:count]])
    require(len(tokens) == cuts[-1], "incorrect prefix scale count")
    return [value.tolist() for value in np.split(np.asarray(tokens), cuts[:-1])]


def context_key(prefix, label):
    digest = hashlib.sha256(f"{label}|{len(prefix)}|".encode())
    for scale in prefix:
        digest.update(np.asarray(scale, dtype="<u2").tobytes())
    return digest.hexdigest()


def verify_context(run, key, prefix, label, cache):
    require(key == context_key(prefix, label), "prior was not keyed by the accepted prefix and decoded class")
    if key not in cache:
        with np.load(run / "probabilities" / f"{key}.npz", allow_pickle=False) as table:
            require(int(table["label"]) == label and int(table["prefix_scales"]) == len(prefix), "probability context metadata mismatch")
            require(np.array_equal(table["prefix"], np.concatenate(prefix)), "cached probability contains a different prefix")
            probabilities = table["log_probs"].copy()
        require(probabilities.shape == (SIZES[len(prefix)] ** 2, 4096) and probabilities.dtype == np.float32, "wrong complete probability table")
        require(np.isfinite(probabilities).all(), "truncated or invalid source prior")
        cache[key] = probabilities
    return cache[key]


def audit_receiver(record, row, source, label, expected_mode, run, table_cache, expected_uses):
    grouped = row["arm"].startswith("group_")
    entropy = row["arm"] == "group_entropy"
    payload_length = 48 if entropy else 12
    header = record["header"]
    bits = np.asarray(header["decoded_bits"], dtype=np.uint8)
    require(len(bits) == payload_length + 22 and not bits[-6:].any(), "invalid header length/termination")
    received_label, received_mode = number(bits[:10]), number(bits[10:12]) + 6
    valid = crc_independent(bits[:payload_length]) == number(bits[payload_length:payload_length + 16])
    valid = valid and received_label < 1000 and received_mode in (7, 8, 9)
    require(valid == header["accepted"] and received_label == header["label"] and received_mode == header["mode"], "header verification mismatch")
    lengths = [number(bits[start:start + 12]) for start in (12, 24, 36)] if entropy else []
    require(lengths == header["lengths"], "entropy stream lengths were not received in header")
    header_false = valid and (received_label != label or received_mode != expected_mode)
    prefix, false_blocks, last = [], 0, 0
    usable_header = valid and (not grouped or received_mode == 9)
    if not usable_header:
        require(record["label"] is None and not record["events"], "failed header leaked class or decoded payload")
    elif grouped:
        require(record["label"] == received_label, "oracle class substituted for decoded class")
        offset = 140 if entropy else 68
        for group, event in enumerate(record["events"]):
            require(event["group"] == group and group < 4, "groups were decoded out of protocol order")
            if event.get("reason") == "invalid_length_header":
                require(entropy and lengths[group - 1] > RAW_BITS[group], "spurious malformed-length rejection")
                break
            wire_length = (lengths[group - 1] if entropy and group > 0 else 0) or RAW_BITS[group]
            decoded = np.asarray(event["decoded_bits"], dtype=np.uint8)
            require(event["source_wire_bits"] == wire_length and len(decoded) == wire_length + 22 and not decoded[-6:].any(), "data bit ledger/termination mismatch")
            require(event["prior_prefix_scales"] == len(prefix), "generated scales entered the trusted decoding context")
            require(event["position"] == offset and event["complex_uses"] == expected_uses[group], "received packet boundary changed")
            offset += expected_uses[group]
            if group > 0 and (row["arm"] == "group_var" or (entropy and lengths[group - 1])):
                table = verify_context(run, event["prior_key"], prefix, received_label, table_cache)
            else:
                require(event["prior_key"] is None, "unexpected source-image prior in a control")
                table = None
            if entropy and group > 0 and lengths[group - 1]:
                recovered = arithmetic_decode(decoded[:wire_length], probability_cdf(table))
                payload = token_bits(recovered)
            else:
                payload = decoded[:wire_length]
                recovered = token_indices(payload)
            require(np.array_equal(recovered, event["recovered_tokens"]), "recovered source was not derived from received bits")
            accepts = crc_independent(payload) == number(decoded[wire_length:wire_length + 16])
            require(accepts == event["accepted"], "source CRC decision mismatch")
            if not accepts:
                require(group + 1 == len(record["events"]), "decoding continued after a rejected source group")
                break
            reference = np.concatenate(source[:6]) if group == 0 else source[group + 5]
            false_blocks += int(not np.array_equal(reference, recovered))
            prefix = divide_prefix(recovered, 6) if group == 0 else prefix + [recovered.tolist()]
            last = len(prefix)
    else:
        require(len(record["events"]) == 1, "whole-frame baseline did not keep its decoded ML payload")
        event = record["events"][0]
        decoded = np.asarray(event["decoded_bits"], dtype=np.uint8)
        source_length = 12 * sum(size ** 2 for size in SIZES[:received_mode])
        require(len(decoded) == source_length + 22 and not decoded[-6:].any(), "whole-frame length/tail mismatch")
        accepts = crc_independent(decoded[:source_length]) == number(decoded[source_length:source_length + 16])
        require(accepts == event["accepted"] and event["output_uses_unverified_ML_prefix_if_CRC_fails"], "strong whole-frame baseline was weakened")
        prefix = divide_prefix(token_indices(decoded[:source_length]), received_mode)
        last = received_mode if accepts else 0
        false_blocks = int(accepts and not all(np.array_equal(scale, source[index]) for index, scale in enumerate(prefix)))
    require(prefix == record["output_prefix"] and last == record["last_accepted_scale"], "output used unaccepted or hidden generated source scales")
    prefix_correct = all(np.array_equal(scale, source[index]) for index, scale in enumerate(prefix)) and record["label"] == label
    accepted_tokens = sum(len(scale) for scale in prefix[:last]) if prefix_correct and not header_false else 0
    metrics = {"header_accepted": int(valid), "header_false_acceptance": int(header_false),
               "source_crc_false_acceptance_count": false_blocks, "last_accepted_scale": last,
               "correct_accepted_tokens": accepted_tokens, "correct_m9_accepted": int(last == 9 and prefix_correct and not header_false),
               "output_prefix_scales": len(prefix), "output_prefix_correct": int(prefix_correct)}
    require(all(int(row[name]) == value for name, value in metrics.items()), "receiver reliability metrics mismatch")
    key = "header_erasure" if record["label"] is None else context_key(prefix, record["label"])
    require(key == row["reconstruction_key"], "image selection depended on information beyond receiver output")


@torch.no_grad()
def score_images(source, images, perceptual, dino, device):
    source_tensor = torch.tensor(source, device=device)[None]
    all_images = np.concatenate((source[None], images), axis=0)
    embeddings, perceptual_values = [], []
    for start in range(0, len(all_images), 8):
        batch = torch.tensor(all_images[start:start + 8], device=device)
        resized = functional.interpolate(batch, size=(224, 224), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        embeddings.extend(dino.forward_features((resized - mean) / std)["x_norm_clstoken"].float().cpu().numpy())
    for start in range(0, len(images), 8):
        batch = torch.tensor(images[start:start + 8], device=device)
        perceptual_values.extend(perceptual(batch * 2 - 1, source_tensor.expand_as(batch) * 2 - 1).reshape(-1).cpu().numpy())
    embeddings = np.asarray(embeddings, dtype=np.float64)
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    psnr = -10 * np.log10(np.mean((images.astype(np.float64) - source) ** 2, axis=(1, 2, 3)))
    return psnr, np.asarray(perceptual_values), normalized[1:] @ normalized[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-AUDIT-001")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    receipt = verify_artifacts(run, "completion.json")
    require(receipt["mode"] == "full" and receipt["rows"] == 12000, "not a complete full protocol run")
    config = yaml.safe_load((run / "snapshots/configs/progressive_channel.yaml").read_text())
    output = create_output(args.output_dir)
    snapshot(output, [Path(__file__), ROOT / "scripts/audit_single_scale_channel.py"])
    try:
        populations = json.loads((run / "populations.json").read_text())
        targets, calibration = populations["target"], populations["resource_calibration"]
        require(len(targets) == 100 and len(calibration) == 20, "population size mismatch")
        for field in ("image_id", "path", "file_sha256", "preprocessed_rgb_sha256"):
            require(not {row[field] for row in targets} & {row[field] for row in calibration}, "resource-selection population leaked target data")
        with (run / "calibration.csv").open() as handle:
            calibration_rows = list(csv.DictReader(handle))
        require(len(calibration_rows) == 1800, "incomplete registered calibration grid")
        policies = json.loads((run / "selected_policies.json").read_text())
        require(not policies["target_results_observed"] and sha256(run / "selected_policies.json") == receipt["selected_policies_sha256"], "policy changed after target evaluation")
        for family, selected in policies["choices"].items():
            for snr, chosen in selected.items():
                alternatives = []
                for base in config["calibration"]["base_uses_candidates"]:
                    values = [int(row["correct_accepted_tokens"]) for row in calibration_rows if row["family"] == family and float(row["snr_db"]) == float(snr) and int(row["base_uses"]) == base]
                    require(len(values) == 60, "calibration used unequal resources or sample counts")
                    alternatives.append((np.mean(values), -base))
                require(chosen == -max(alternatives)[1], "resource selection does not match the frozen rule")
        with (run / "per_frame.csv").open() as handle:
            recorded = list(csv.DictReader(handle))
        lookup = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]): row for row in recorded}
        require(len(lookup) == len(recorded) == 12000, "incomplete or duplicate target frame grid")
        with np.load(ROOT / config["prior_run"] / "source_tokens.npz", allow_pickle=False) as cache:
            token_lookup = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
        ledgers = json.loads((run / "wire_ledgers.json").read_text())
        require(len(ledgers) == 1000, "incomplete grouped budget ledgers")
        ledger_lookup = {(row["image_index"], row["snr_db"], row["family"]): row for row in ledgers}
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        torch.hub.set_dir(str(Path(config["quality"]["alexnet_checkpoint"]).parent.parent))
        with mock.patch("torch.hub.download_url_to_file", side_effect=RuntimeError("network weight downloads forbidden")):
            perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
        dino = torch.hub.load(config["quality"]["dino_source"], "dinov2_vits14", source="local", pretrained=False)
        dino.load_state_dict(torch.load(config["quality"]["dino_checkpoint"], map_location="cpu", weights_only=True), strict=True)
        dino = dino.to(device).eval().requires_grad_(False)
        require(state_sha256(perceptual) == receipt["frozen_before"]["lpips"], "manually loaded LPIPS was not the canonical pretrained model")
        require(state_sha256(dino) == receipt["frozen_before"]["dino"], "DINO checkpoint mismatch")
        table_cache, maximum_errors, quality_rows, legacy_examples = {}, np.zeros(3), [], []
        image_count, trace_count = 0, 0
        for image_index, target in enumerate(targets):
            directory = run / "images" / f"{image_index:03d}"
            records = json.loads((directory / "receivers.json").read_text())
            require(len(records) == 120, "missing receiver traces")
            flat = token_lookup[target["image_id"]]
            source = divide_prefix(flat, 10)
            with np.load(directory / "reconstructions.npz", allow_pickle=False) as cache:
                images, original = cache["images"], cache["source"]
            require(images.dtype == np.float32 and np.isfinite(images).all() and images.min() >= 0 and images.max() <= 1, "invalid image cache")
            require(sha256(target["path"]) == target["file_sha256"], "target image changed")
            with Image.open(target["path"]) as opened:
                image = opened.convert("RGB")
                factor = 256 / min(image.size)
                image = image.resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.BICUBIC)
                left, top = round((image.width - 256) / 2), round((image.height - 256) / 2)
                pixels = np.asarray(image.crop((left, top, left + 256, top + 256))).transpose(2, 0, 1).copy()
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == target["preprocessed_rgb_sha256"], "preprocessed source identity changed")
            require(np.max(np.abs(original - pixels / 255.0)) < 1e-7, "source metrics do not use the original image")
            recomputed_scores = score_images(original, images, perceptual, dino, device)
            image_count += len(images)
            with np.load(directory / "waveforms.npz", allow_pickle=False) as signals:
                for record in records:
                    arm, snr, seed = record["arm"], record["snr_db"], record["seed"]
                    row = lookup[(image_index, snr, seed, arm)]
                    expected_mode = 9 if arm.startswith("group_") else (config["adaptive_modes"][str(snr)] if arm == "whole_adaptive" else int(arm[-1]))
                    family = ("group_entropy" if arm == "group_entropy" else "group_raw") if arm.startswith("group_") else f"whole_m{expected_mode}"
                    signal = signals[f"{family}_{config['snr_db'].index(snr)}"]
                    require(signal.shape == (3060, 2) and np.isin(signal, (-1, 1)).all() and int(row["complex_uses"]) == 3060, "actual power/use budget mismatch")
                    seed_integer = int.from_bytes(hashlib.sha256(f"{target['image_id']}|{seed}".encode()).digest()[:8], "big")
                    received = signal + np.random.default_rng(seed_integer).standard_normal((3060, 2)) / np.sqrt(10 ** (snr / 10))
                    require(hashlib.sha256(received.tobytes()).hexdigest() == row["received_sha256"] == record["received_sha256"], "receiver observation provenance mismatch")
                    uses = None
                    if arm.startswith("group_"):
                        ledger = ledger_lookup[(image_index, snr, family)]
                        base = policies["choices"]["group_entropy" if arm == "group_entropy" else "group_var"][str(snr)]
                        header_uses = 140 if arm == "group_entropy" else 68
                        remaining = 3060 - header_uses - base
                        fine = remaining * np.asarray([790, 1222, 2050]) // 4062
                        fine[-1] += remaining - int(fine.sum())
                        uses = [base, *fine.tolist()]
                        require(ledger["block_uses"] == uses and ledger["header_uses"] == header_uses, "resource ledger mismatch")
                        require(sum(uses) + header_uses == ledger["total_complex_uses"] == 3060 and ledger["data_crc_bits"] == 64 and ledger["data_tail_bits"] == 24, "CRC/tail omitted from grouped budget")
                        for event in record["events"]:
                            if "position" in event:
                                piece = received[event["position"]:event["position"] + event["complex_uses"]]
                                require(hashlib.sha256(piece.tobytes()).hexdigest() == event["received_sha256"], "wrong scale waveform supplied to decoder")
                    audit_receiver(record, row, source, target["class_index"], expected_mode, run, table_cache, uses)
                    image_slot = int(row["reconstruction_index"])
                    require(image_slot == record["reconstruction_index"] and 0 <= image_slot < len(images), "wrong cached image selected")
                    values = {name: float(score[image_slot]) for name, score in zip(("psnr_db", "lpips_alex", "dino_cosine"), recomputed_scores)}
                    errors = np.array([abs(values[name] - float(row[name])) for name in values])
                    maximum_errors = np.maximum(maximum_errors, errors)
                    require(np.max(errors) < 2e-5, "independent image metric mismatch")
                    quality_rows.append({"image_index": image_index, "snr_db": snr, "arm": arm, **values})
                    if image_index < 2 and arm == "whole_m9" and seed == 2001 and snr in (4.0, 13.0) and record["label"] is not None:
                        legacy_examples.append((record["output_prefix"], record["label"], images[image_slot].copy()))
                    trace_count += 1
            for snr in config["snr_db"]:
                for seed in config["noise_seeds"]:
                    require(len({lookup[(image_index, snr, seed, arm)]["received_sha256"] for arm in ("group_ml", "group_static", "group_var")}) == 1, "matched receivers were given different information")
            if (image_index + 1) % 10 == 0:
                print(f"protocol audit {image_index + 1}/100, rescored_images={image_count}", flush=True)
        with (run / "summary.csv").open() as handle:
            for row in csv.DictReader(handle):
                group = [entry for entry in quality_rows if entry["arm"] == row["arm"] and entry["snr_db"] == float(row["snr_db"])]
                require(len(group) == 300, "unequal full-system comparison sample counts")
                for metric in ("psnr_db", "lpips_alex", "dino_cosine"):
                    require(abs(np.mean([entry[metric] for entry in group]) - float(row[metric])) < 2e-5, "image summary mismatch")
        max_interval_error = 0.0
        with (run / "paired_quality.csv").open() as handle:
            for row in csv.DictReader(handle):
                selected_snr = [float(value) for value in row["snr_db"].split("+")]
                grouped = defaultdict(list)
                for entry in quality_rows:
                    if entry["snr_db"] in selected_snr:
                        grouped[(entry["image_index"], entry["arm"])].append(entry[row["metric"]])
                differences = np.array([np.mean(grouped[(index, "group_var")]) - np.mean(grouped[(index, row["control"])]) for index in range(100)])
                draws = np.random.default_rng(config["bootstrap"]["seed"]).integers(100, size=(10000, 100))
                lower, upper = np.percentile(differences[draws].mean(axis=1), [2.5, 97.5])
                for name, value in (("VAR_minus_control", differences.mean()), ("ci_low", lower), ("ci_high", upper)):
                    max_interval_error = max(max_interval_error, abs(value - float(row[name])))
        require(max_interval_error < 2e-5, "paired source-image quality interval mismatch")
        model_config = yaml.safe_load((ROOT / config["model_config"]).read_text())
        vae, var = load_models(model_config["paths"], device)
        legacy_source = Path('/workspace/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc/src')
        sys.path.insert(0, str(legacy_source))
        from cadsd_jscc.var_prefix_consistency import complete_received_prefix
        max_legacy_error = 0.0
        with torch.no_grad():
            for prefix, label, cached_image in legacy_examples:
                received = [torch.tensor(scale, device=device, dtype=torch.long)[None] for scale in prefix]
                latent = complete_received_prefix(var, vae, received, torch.tensor([label], device=device))
                reference = vae.fhat_to_img(latent).clamp(-1, 1).add(1).mul(0.5)[0].cpu().numpy()
                max_legacy_error = max(max_legacy_error, float(np.max(np.abs(reference - cached_image))))
        require(max_legacy_error < 1e-5, "historical argmax receiver was not preserved")
        require(receipt["frozen_before"] == receipt["frozen_after"], "primary model freeze record failed")
        for name, model in (("vae", vae), ("var", var), ("lpips", perceptual), ("dino", dino)):
            require(state_sha256(model) == receipt["frozen_before"][name], f"independent frozen model fingerprint differs: {name}")
        result = {"status": "AUDIT_PASS", "run": str(run), "run_completion_sha256": sha256(run / "completion.json"),
                  "receiver_traces_recomputed": trace_count, "unique_images_neurally_rescored": image_count,
                  "canonical_pretrained_LPIPS_fingerprint_matches": True, "max_metric_errors_PSNR_LPIPS_DINO": maximum_errors.tolist(),
                  "max_paired_interval_error": max_interval_error, "accepted_context_probability_tables_checked": len(table_cache),
                  "legacy_argmax_replay_cases": len(legacy_examples), "legacy_argmax_max_pixel_error": max_legacy_error,
                  "limitations": ["arithmetic packet replay uses the registered codec; CRC and trusted-state checks are independent",
                                  "development-only statistics conditional on selected policies and fitted frequency table"],
                  "output_hashes": artifact_hashes(output)}
        write_json(output / "audit.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "AUDIT_FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
