from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from latent_enhancement.latent import (
    complete_latent, enhancement_noise, normalize_enhancement, original_rgb,
)
from latent_enhancement.runtime import CONFIG as TRAIN_CONFIG
from latent_enhancement.runtime import digest, image_losses, model_paths, settings, write_json
from latent_enhancement_b.common import load_decoder, scale_statistics
from latent_enhancement_b.model import build_arms, render_received
from latent_enhancement_eval.deployment import channel_apply, rx_base, rx_continuous, tx_encode
from var_comm.next_scale_prior import load_models, preprocess, state_sha256
from var_comm.progressive import (
    complete_image, header_bits, prefix_key, receive_whole, split_prefix, transmit_whole,
)
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.scale_channel import (
    bits_to_indices, channel_evidence, crc_accepts, decode_map, encode_packet,
    indices_to_bits, rate_match_indices,
)
from var_comm.study import paired_interval, seeded_noise
from var_comm.whole_entropy import decode_source, encode_prefixes, payload_key


PROJECT = Path(__file__).resolve().parents[5]
EXPERIMENT = PROJECT / "experiments/var-latent-enhancement-20260917"
EVAL_CONFIG = EXPERIMENT / "evaluation/config.json"
DIGITAL_ROOT = PROJECT / "outputs/COMMUNICATION-CONVERGENCE-20260915/DEVELOPMENT_001"
SOURCE_RECON_ROOT = PROJECT / "outputs/VAR-PROGRESSIVE-CHANNEL-001"
TOKENS_PATH = PROJECT / "outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/source_tokens.npz"
OUTPUT_ROOT = PROJECT / "outputs/VAR-LATENT-ENHANCEMENT-20260917"


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_rows(path, rows):
    if not rows:
        raise RuntimeError("refusing to write empty evaluation rows")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_targets():
    targets = read_json(DIGITAL_ROOT / "population.json")
    if len(targets) != 100:
        raise RuntimeError("development target population changed")
    with np.load(TOKENS_PATH, allow_pickle=False) as archive:
        image_ids = archive["image_ids"].tolist()
        token_values = archive["tokens"].copy()
        roles = archive["roles"].tolist()
        classes = archive["classes"].copy()
    if roles[:100] != ["target_development"] * 100:
        raise RuntimeError("source token target ordering changed")
    records = []
    for index, target in enumerate(targets):
        if target["image_id"] != image_ids[index] or int(target["class_index"]) != int(classes[index]):
            raise RuntimeError("development target/token binding changed")
        source_path = SOURCE_RECON_ROOT / f"images/{index:03d}/reconstructions.npz"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with np.load(source_path, allow_pickle=False) as archive:
            pixels = np.rint(archive["source"] * 255).astype(np.uint8)
        normalized, rgb_sha = preprocess(Path(target["path"]))
        expected_pixels = normalized.add(1).mul(127.5).round().to(torch.uint8).numpy()
        if not np.array_equal(pixels, expected_pixels):
            raise RuntimeError(f"development source pixels changed at {index}")
        records.append({"index": index, "target": target, "pixels": pixels,
                        "tokens": token_values[index].astype(np.int64), "rgb_sha256": rgb_sha,
                        "source_npz_sha256": digest(source_path)})
    return records


def raw_transmit_budget(source, label, mode, total_uses):
    header_uses = 68
    data_uses = int(total_uses) - header_uses
    raw_bits = indices_to_bits(np.concatenate(source[:mode]))
    header = encode_packet(header_bits(label, mode), header_uses)
    body = encode_packet(raw_bits, data_uses)
    signal = np.concatenate((header["symbols"], body["symbols"]))
    if signal.shape != (total_uses, 2) or not np.all(np.abs(signal) == 1):
        raise RuntimeError("raw budget waveform changed its declared length or energy")
    return signal, {"header_uses": header_uses, "data_uses": data_uses, "payload_bits": len(raw_bits),
                    "actual_payload_bits": len(raw_bits), "length_field": 0}


def raw_receive_budget(received, snr, total_uses):
    from var_comm.progressive import decode_header

    header_uses = 68
    data_uses = int(total_uses) - header_uses
    header = decode_header(received[:header_uses], snr)
    result = {"header": header, "label": header["label"] if header["accepted"] else None,
              "mode": header["mode"] if header["accepted"] else None, "prefix": [],
              "body_crc_accepted": False, "source_complete": False}
    if not header["accepted"]:
        return result
    mode = header["mode"]
    source_length = 12 * sum(size ** 2 for size in (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)[:mode])
    information_length = source_length + 22
    mapping = rate_match_indices(2 * information_length, 2 * data_uses)
    evidence = channel_evidence(received[header_uses:], mapping, information_length, snr)
    decoded, score = decode_map(evidence)
    result["prefix"] = split_prefix(bits_to_indices(decoded[:source_length]), mode)
    result["body_crc_accepted"] = bool(crc_accepts(decoded[:-6]))
    result["source_complete"] = True
    result["body_score"] = score
    result["decoded_bits"] = decoded
    return result


def arithmetic_transmit_budget(payload, label, mode, total_uses):
    header_uses = 94
    data_uses = int(total_uses) - header_uses
    length = int(payload["length_field"])
    bits = np.asarray(payload["payload"], dtype=np.uint8)
    header_bits_value = np.concatenate((indices_to_bits([label], 10), indices_to_bits([mode - 6], 2), indices_to_bits([length], 13)))
    header = encode_packet(header_bits_value, header_uses)
    body = encode_packet(bits, data_uses)
    signal = np.concatenate((header["symbols"], body["symbols"]))
    if signal.shape != (total_uses, 2) or not np.all(np.abs(signal) == 1):
        raise RuntimeError("arithmetic budget waveform changed its declared length or energy")
    return signal, {"header_uses": header_uses, "data_uses": data_uses, "payload_bits": payload["raw_payload_bits"],
                    "actual_payload_bits": len(bits), "length_field": length,
                    "attempted_arithmetic_bits": payload["attempted_arithmetic_bits"],
                    "raw_fallback": payload["raw_fallback"]}


def arithmetic_receive_budget(received, snr, total_uses):
    header_uses = 94
    data_uses = int(total_uses) - header_uses
    header_mapping = rate_match_indices(94, 188)
    header_bits_value, header_score = decode_map(channel_evidence(received[:header_uses], header_mapping, 47, snr))
    label = int(bits_to_indices(header_bits_value[:10], 10)[0])
    mode = 6 + int(bits_to_indices(header_bits_value[10:12], 2)[0])
    length = int(bits_to_indices(header_bits_value[12:25], 13)[0])
    checksum = bool(crc_accepts(header_bits_value[:-6]))
    raw_length = 12 * sum(size ** 2 for size in (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)[:mode]) if mode in (7, 8, 9) else 0
    legal = label < 1000 and mode in (7, 8, 9) and (length == 0 or 2 <= length < raw_length)
    header = {"accepted": bool(checksum and legal), "phy_crc_accepted": checksum, "fields_legal": legal,
              "label": label, "mode": mode, "length_field": length, "decoded_bits": header_bits_value,
              "score": header_score, "complex_uses": total_uses}
    result = {"header": header, "label": label if header["accepted"] else None,
              "mode": mode if header["accepted"] else None, "payload": None,
              "body_crc_accepted": False, "source_complete": False}
    if not header["accepted"]:
        return result
    payload_length = length or raw_length
    information_length = payload_length + 22
    mapping = rate_match_indices(2 * information_length, 2 * data_uses)
    decoded, score = decode_map(channel_evidence(received[header_uses:], mapping, information_length, snr))
    result["payload"] = decoded[:payload_length].copy()
    result["body_decoded_bits"] = decoded
    result["body_score"] = score
    result["body_crc_accepted"] = bool(crc_accepts(decoded[:-6]))
    return result


def raw_candidate_key(phy):
    if phy["label"] is None:
        return "header_erasure"
    return prefix_key(phy["prefix"], phy["label"])


def render_digital_candidate(phy, family, renderer, vae, var, decoder, device, cache):
    key = (family, raw_candidate_key(phy) if family == "raw" else payload_key(phy))
    if key not in cache:
        if phy["label"] is None:
            cache[key] = {"latent": None, "image_D0": np.full((3, 256, 256), 0.5, dtype=np.float32),
                          "prefix": [], "source_complete": False}
        elif family == "raw":
            latent = complete_latent(vae, var, phy["prefix"], phy["label"], device)
            cache[key] = {"latent": latent, "image_D0": original_rgb(vae, latent)[0].cpu().numpy(),
                          "prefix": phy["prefix"], "source_complete": True}
        else:
            decoded = decode_source(phy, vae, var, device, render=True, return_latent=True)
            latent = decoded.get("latent")
            if latent is None:
                latent = complete_latent(vae, var, decoded["prefix"], phy["label"], device)
            cache[key] = {"latent": latent, "image_D0": decoded["image"], "prefix": decoded["prefix"],
                          "source_complete": decoded["source_complete"], "source_error": decoded["source_error"]}
    item = cache[key]
    if renderer == "D0":
        image = item["image_D0"]
    elif item["latent"] is None:
        image = np.full((3, 256, 256), 0.5, dtype=np.float32)
    else:
        image = decoder(item["latent"])[0].cpu().numpy()
    return image, item


def render_base(latent, status, renderer, vae, decoder):
    if status[0] < 0.5 or latent is None:
        return np.full((3, 256, 256), 0.5, dtype=np.float32)
    if renderer == "D0":
        return original_rgb(vae, latent)[0].cpu().numpy()
    return decoder(latent)[0].cpu().numpy()


def method_specs(config):
    methods = [dict(item) for item in config["methods"]["latent"]]
    for family in config["methods"]["digital"]["families"]:
        for budget in config["methods"]["digital"]["budgets"]:
            for mode in config["methods"]["digital"]["modes"]:
                for renderer in config["methods"]["digital"]["renderers"]:
                    methods.append({"name": f"{family}_N{budget}_m{mode}_{renderer}", "family": family,
                                    "N": budget, "E": 2 * budget, "sent_mode": mode, "renderer": renderer})
    return methods


def execution_scope(method, *, model_sha256, decoder_sha256, population_role="target_development"):
    """Return the immutable identity carried by every evaluation row.

    The old table only carried ``N`` and ``family``.  Those fields are not
    sufficient to prevent a follow-up Decoder or protocol from being merged
    into the same summary, so the complete execution scope is emitted with
    each sample row and checked on resume by the validator.
    """
    family = str(method["family"])
    budget = int(method["N"])
    mode = method.get("sent_mode", 8)
    renderer = str(method.get("renderer", "Dc"))
    return {
        "population_role": str(population_role),
        "family": family,
        "N": budget,
        "renderer": renderer,
        "model_sha256": str(model_sha256),
        "decoder_sha256": str(decoder_sha256),
        "protocol_id": f"{family}:N{budget}:m{mode}",
        "noise_model": "seeded_noise_v1:variance=1/SNR",
        "budget_scope": f"N{budget}:E{int(method['E'])}",
    }


def method_model_sha(method, model_state):
    """Bind each row to the exact frozen model/arm used by that method."""
    family = str(method["family"])
    if family == "continuous_enhancement":
        arm_name = "enhancement512" if int(method["N"]) == 3572 else "enhancement1024"
        parts = (model_state["vae"], model_state["var"], model_state[arm_name])
    elif family == "receiver_only_refiner":
        parts = (model_state["vae"], model_state["var"], model_state["receiver_only_refiner"])
    else:
        parts = (model_state["vae"], model_state["var"])
    return hashlib.sha256(":".join(parts).encode("ascii")).hexdigest()


def checkpoint_bindings():
    files = [EVAL_CONFIG, TRAIN_CONFIG, TOKENS_PATH, DIGITAL_ROOT / "population.json"]
    files += sorted((DIGITAL_ROOT.parent / "images").glob("*/reconstructions.npz"))
    for relative in ("stage_A_v1/selected.json", "stage_B_v1/training/selected_enhancement512.json",
                     "stage_B_v1/training/selected_enhancement1024.json", "stage_B_v1/training/selected_receiver_only_refiner.json"):
        files.append(OUTPUT_ROOT / relative)
    return {str(path): digest(path) for path in files}


def source_frame_rows(record, method_order, outputs, metadata, quality_rows):
    frame_count = len(metadata[method_order[0]])
    rows = []
    offset = 0
    for method in method_order:
        for frame in range(frame_count):
            row = dict(metadata[method][frame])
            row.update({"method": method, **quality_rows[offset + frame]})
            rows.append(row)
        offset += frame_count
    return rows


def summarize(rows):
    by_source_snr = {}
    for row in rows:
        key = (row["method"], int(row["source_index"]), float(row["snr_db"]))
        values = by_source_snr.setdefault(key, {"psnr_db": [], "lpips_alex": [], "dino_cosine": []})
        for name in values:
            values[name].append(float(row[name]))
    source_means = {}
    for (method, source_index, snr), values in by_source_snr.items():
        source_means[(method, source_index, snr)] = {name: float(np.mean(data)) for name, data in values.items()}
    all_snr = {}
    for (method, source_index, _snr), values in source_means.items():
        current = all_snr.setdefault((method, source_index), {name: [] for name in values})
        for name in values:
            current[name].append(values[name])
    summary = {}
    for (method, source_index), values in all_snr.items():
        current = summary.setdefault(method, {name: [] for name in values})
        for name in values:
            current[name].append(float(np.mean(values[name])))
    method_rows = []
    for method, values in summary.items():
        method_rows.append({"method": method, "sources": len(values["psnr_db"]),
                            **{name: float(np.mean(data)) for name, data in values.items()}})
    method_rows.sort(key=lambda row: row["method"])
    comparisons = []
    source_arrays = {method: {name: np.asarray(data[name]) for name in data} for method, data in summary.items()}
    if "m8_D0" not in source_arrays:
        raise RuntimeError("m8_D0 reference is missing")
    for method in sorted(source_arrays):
        if method == "m8_D0":
            continue
        comparisons.append({"method": method, "reference": "m8_D0", "delta_psnr_db": paired_interval(
            source_arrays[method]["psnr_db"] - source_arrays["m8_D0"]["psnr_db"], 2026091801, 10000),
            "delta_lpips": paired_interval(source_arrays[method]["lpips_alex"] - source_arrays["m8_D0"]["lpips_alex"], 2026091802, 10000),
            "delta_dino": paired_interval(source_arrays[method]["dino_cosine"] - source_arrays["m8_D0"]["dino_cosine"], 2026091803, 10000)})
    return method_rows, comparisons


def run(args):
    config = read_json(EVAL_CONFIG)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configure_environment()
    targets = load_targets()
    limit = min(args.max_sources or len(targets), len(targets))
    snrs = config["snrs_db"][:args.max_snrs] if args.max_snrs else config["snrs_db"]
    seeds = config["noise_seeds"][:args.max_seeds] if args.max_seeds else config["noise_seeds"]
    methods = method_specs(config)
    method_order = [method["name"] for method in methods]
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("development evaluation requires the authorized GPU0")
    paths = model_paths()
    vae, var = load_models(paths, device)
    decoder = load_decoder(vae, device)
    scale = scale_statistics(device)
    recipe = settings()["stage_B"]
    selected_arms = {}
    for name in ("enhancement512", "enhancement1024", "receiver_only_refiner"):
        selected = read_json(OUTPUT_ROOT / f"stage_B_v1/training/selected_{name}.json")
        checkpoint = Path(selected["checkpoint"])
        if digest(checkpoint) != selected["checkpoint_sha256"]:
            raise RuntimeError(f"selected checkpoint changed: {name}")
        arms = build_arms((32, 16, 16), scale, recipe).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        arms.load_state_dict(state["arms"], strict=True)
        selected_arms[name] = arms[name].eval().requires_grad_(False)
        del arms, state
    perceptual, dino, _linear = load_quality_models(yaml.safe_load((PROJECT / "configs/progressive_channel.yaml").read_text())["quality"], device)
    before = {"vae": state_sha256(vae), "var": state_sha256(var), "decoder": state_sha256(decoder),
              **{name: state_sha256(model) for name, model in selected_arms.items()}}
    bindings = checkpoint_bindings()
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata["bindings"] != bindings:
            raise RuntimeError("evaluation input bindings changed on resume")
        if metadata.get("config_sha256") != digest(EVAL_CONFIG) or metadata.get("snrs_db") != snrs or metadata.get("noise_seeds") != seeds:
            raise RuntimeError("evaluation protocol/noise grid changed on resume")
        if metadata.get("model_state_sha256") != before:
            raise RuntimeError("evaluation model or decoder identity changed on resume")
        if metadata.get("scope_fields") != ["population_role", "family", "N", "renderer", "model_sha256", "decoder_sha256",
                                             "protocol_id", "noise_model", "budget_scope"]:
            raise RuntimeError("evaluation scope schema changed on resume")
    else:
        metadata = {"status": "DEVELOPMENT_EVALUATION_RUNNING", "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "config_sha256": digest(EVAL_CONFIG), "bindings": bindings, "method_order": method_order,
                    "snrs_db": snrs, "noise_seeds": seeds, "source_count": limit, "new_holdout_used": False,
                    "quality_model_state": {"perceptual": state_sha256(perceptual), "dino": state_sha256(dino)},
                    "model_state_sha256": before, "scope_fields": ["population_role", "family", "N", "renderer",
                    "model_sha256", "decoder_sha256", "protocol_id", "noise_model", "budget_scope"],
                    "no_development_mode_selection": True}
        atomic_json(metadata_path, metadata)
    global_rows = []
    completed = 0
    for record in targets[:limit]:
        source_index = record["index"]
        directory = output / "images" / f"{source_index:03d}"
        receipt_path = directory / "receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["image_id"] != record["target"]["image_id"]:
                raise RuntimeError("resumed source identity changed")
            rows_path = directory / "per_frame.csv"
            reconstructions_path = directory / "reconstructions.npz"
            if digest(rows_path) != receipt.get("output_hashes", {}).get("per_frame.csv"):
                raise RuntimeError("resumed per-frame result hash changed")
            if digest(reconstructions_path) != receipt.get("output_hashes", {}).get("reconstructions.npz"):
                raise RuntimeError("resumed reconstruction cache hash changed")
            with rows_path.open(newline="", encoding="utf-8") as handle:
                resumed_rows = list(csv.DictReader(handle))
            expected_rows = len(method_order) * len(snrs) * len(seeds)
            if len(resumed_rows) != expected_rows:
                raise RuntimeError(f"resumed source row count changed: {len(resumed_rows)} != {expected_rows}")
            for resumed in resumed_rows:
                method = next((item for item in methods if item["name"] == resumed.get("method")), None)
                if method is None:
                    raise RuntimeError("resumed result contains an unknown method")
                expected_scope = execution_scope(method, model_sha256=method_model_sha(method, before),
                                                 decoder_sha256=before["decoder"])
                for field, expected in expected_scope.items():
                    if str(resumed.get(field, "")) != str(expected):
                        raise RuntimeError(f"resumed execution scope changed: {field}")
            global_rows.extend(resumed_rows)
            completed += 1
            continue
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        pixels = record["pixels"]
        source_rgb = pixels.astype(np.float32) / 255.0
        source = split_prefix(record["tokens"], 10)
        label = int(record["target"]["class_index"])
        image = torch.from_numpy(source_rgb[None]).to(device).float().mul(2).sub(1)
        with torch.no_grad():
            continuous = vae.quant_conv(vae.encoder(image))
            base_tx = complete_latent(vae, var, source[:8], label, device)
        base_signal = transmit_whole(source, label, 8)
        base_encoded = tx_encode(base_signal)
        if base_signal.shape != (3060, 2) or np.square(base_signal).sum() != 6120:
            raise RuntimeError("base m8 waveform ledger changed")
        arithmetic_payloads = encode_prefixes(vae, var, source, label, device, modes=(7, 8, 9))
        prepared = {}
        for method in methods:
            if method["family"] == "raw":
                key = (method["family"], method["N"], method["sent_mode"])
                if key not in prepared:
                    prepared[key] = raw_transmit_budget(source, label, method["sent_mode"], method["N"])
            elif method["family"] == "arithmetic":
                key = (method["family"], method["N"], method["sent_mode"])
                if key not in prepared:
                    prepared[key] = arithmetic_transmit_budget(arithmetic_payloads[method["sent_mode"]], label,
                                                               method["sent_mode"], method["N"])
        images_by_method = {name: [] for name in method_order}
        metadata_by_method = {name: [] for name in method_order}
        candidate_cache = {}
        base_cache = {}
        for snr in snrs:
            for noise_seed in seeds:
                base_channel = channel_apply(base_encoded, record["target"]["image_id"], noise_seed, snr)
                def complete_base(prefix, received_label):
                    base_key = prefix_key(prefix, received_label)
                    if base_key not in base_cache:
                        base_cache[base_key] = complete_latent(vae, var, prefix, received_label, device)
                    return base_cache[base_key]
                base_result = rx_base(base_channel["base_received"], snr, complete_fn=complete_base,
                                      device=device, latent_shape=(1, 32, 16, 16))
                base_reception = base_result["reception"]
                base_latent = base_result["latent"]
                base_status = base_result["status"]
                base_header_ok = base_result["header_ok"]
                base_body_ok = base_result["body_ok"]
                for method in methods:
                    name = method["name"]
                    family = method["family"]
                    if family in ("digital_base", "decoder_only", "receiver_only_refiner", "continuous_enhancement"):
                        if family == "continuous_enhancement":
                            arm_name = "enhancement512" if method["N"] == 3572 else "enhancement1024"
                            arm = selected_arms[arm_name]
                            with torch.no_grad():
                                waveform = arm.encoder(continuous - base_tx, base_tx)
                                encoded = tx_encode(base_signal, waveform[0].cpu().numpy())
                                channel = channel_apply(encoded, record["target"]["image_id"], noise_seed, snr,
                                                         base_received=base_channel["base_received"])
                                latent = rx_continuous(channel["enhancement_received"], base_result, arm, snr,
                                                        device=device, latent_shape=(1, 32, 16, 16))
                                status_tensor = torch.as_tensor(base_status[None], device=device)
                                output_image = render_received(decoder, latent, status_tensor)[0].cpu().numpy()
                            header_ok, body_ok, received_mode, source_complete = base_header_ok, base_body_ok, int(base_status[2]), base_header_ok
                            payload_bits, source_error = int(255 * 12), "base_m8"
                        elif family == "receiver_only_refiner":
                            arm = selected_arms["receiver_only_refiner"]
                            with torch.no_grad():
                                snr_tensor = torch.tensor([snr], device=device)
                                status_tensor = torch.as_tensor(base_status[None], device=device)
                                latent = arm.receiver(None, base_latent if base_latent is not None else torch.zeros((1, 32, 16, 16), device=device), snr_tensor, status_tensor)
                                output_image = render_received(decoder, latent, status_tensor)[0].cpu().numpy()
                            header_ok, body_ok, received_mode, source_complete = base_header_ok, base_body_ok, int(base_status[2]), base_header_ok
                            payload_bits, source_error = int(255 * 12), "base_m8"
                        else:
                            renderer = "D0" if family == "digital_base" else "Dc"
                            output_image = render_base(base_latent, base_status, renderer, vae, decoder)
                            header_ok, body_ok, received_mode, source_complete = base_header_ok, base_body_ok, int(base_status[2]), base_header_ok
                            payload_bits, source_error = int(255 * 12), "base_m8"
                        images_by_method[name].append(output_image)
                        metadata_by_method[name].append({"source_index": source_index, "image_id": record["target"]["image_id"],
                            "snr_db": float(snr), "noise_seed": int(noise_seed), "N": method["N"], "E": method["E"],
                            "family": family, "sent_mode": "8", "received_mode": received_mode, "header_ok": int(header_ok),
                            "body_crc_ok": int(body_ok), "source_complete": int(source_complete), "payload_bits": payload_bits,
                            "source_error": source_error,
                            **execution_scope(method, model_sha256=method_model_sha(method, before),
                                              decoder_sha256=before["decoder"])})
                    else:
                        key = (family, method["N"], method["sent_mode"])
                        signal, ledger = prepared[key]
                        encoded = tx_encode(signal)
                        received_signal = channel_apply(encoded, record["target"]["image_id"], noise_seed, snr)["received"]
                        phy = raw_receive_budget(received_signal, snr, method["N"]) if family == "raw" else arithmetic_receive_budget(received_signal, snr, method["N"])
                        image_key = (family, raw_candidate_key(phy) if family == "raw" else payload_key(phy))
                        output_image, candidate = render_digital_candidate(phy, family, method["renderer"], vae, var, decoder, device, candidate_cache)
                        images_by_method[name].append(output_image)
                        metadata_by_method[name].append({"source_index": source_index, "image_id": record["target"]["image_id"],
                            "snr_db": float(snr), "noise_seed": int(noise_seed), "N": method["N"], "E": method["E"],
                            "family": family, "sent_mode": int(method["sent_mode"]), "received_mode": phy["mode"] if phy["mode"] is not None else "",
                            "header_ok": int(phy["header"]["accepted"]), "body_crc_ok": int(phy["body_crc_accepted"]),
                            "source_complete": int(candidate["source_complete"]), "payload_bits": ledger["actual_payload_bits"],
                            "arithmetic_length_field": ledger.get("length_field", ""), "source_error": candidate.get("source_error", ""),
                            "candidate_key": image_key,
                            **execution_scope(method, model_sha256=method_model_sha(method, before),
                                              decoder_sha256=before["decoder"])})
        concatenated = []
        for name in method_order:
            concatenated.extend(images_by_method[name])
        metric_rows, _source_features, _embeddings = quality_metrics(source_rgb, concatenated, perceptual, dino, device)
        rows = source_frame_rows(record, method_order, images_by_method, metadata_by_method, metric_rows)
        atomic_npz(directory / "reconstructions.npz", method_order=np.asarray(method_order),
                   images=np.asarray([images_by_method[name] for name in method_order], dtype=np.float32),
                   source=pixels, source_rgb=source_rgb)
        write_rows(directory / "per_frame.csv", rows)
        receipt = {"status": "SOURCE_EVALUATED", "source_index": source_index, "image_id": record["target"]["image_id"],
                   "rows": len(rows), "method_count": len(method_order), "output_hashes": {
                       "per_frame.csv": digest(directory / "per_frame.csv"), "reconstructions.npz": digest(directory / "reconstructions.npz")},
                   "source_npz_sha256": record["source_npz_sha256"], "elapsed_seconds": time.perf_counter() - started,
                   "candidate_cache_entries": len(candidate_cache), "all_failures_retained": True}
        atomic_json(receipt_path, receipt)
        global_rows.extend(rows)
        completed += 1
        atomic_json(output / "status.json", {"status": "DEVELOPMENT_EVALUATING", "completed_sources": completed,
            "total_sources": limit, "last_source": source_index, "last_source_seconds": receipt["elapsed_seconds"],
            "rows": len(global_rows), "timestamp": time.time()})
        print(f"development source {completed}/{limit} index={source_index} rows={len(rows)} seconds={receipt['elapsed_seconds']:.1f}", flush=True)
    if len(global_rows) != limit * len(method_order) * len(snrs) * len(seeds):
        raise RuntimeError(f"incomplete development rows: {len(global_rows)}")
    global_rows.sort(key=lambda row: (int(row["source_index"]), row["method"], float(row["snr_db"]), int(row["noise_seed"])))
    write_rows(output / "per_frame.csv", global_rows)
    summary, comparisons = summarize(global_rows)
    with (output / "method_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    atomic_json(output / "paired_comparisons.json", comparisons)
    after = {"vae": state_sha256(vae), "var": state_sha256(var), "decoder": state_sha256(decoder),
             **{name: state_sha256(model) for name, model in selected_arms.items()}}
    if before != after:
        raise RuntimeError("frozen model state changed during evaluation")
    atomic_json(output / "completion.json", {"status": "DEVELOPMENT_EVALUATION_COMPLETE", "sources": limit,
        "rows": len(global_rows), "methods": len(method_order), "snrs_db": snrs, "noise_seeds": seeds,
        "new_holdout_used": False, "all_failures_included": True, "frozen_models_unchanged": True,
        "method_summary": "method_summary.csv", "paired_comparisons": "paired_comparisons.json",
        "source_image_aggregation": "three_noise_then_source_image_then_mean_over_fixed_SNRs",
        "timestamp": time.time()})
    atomic_json(output / "status.json", {"status": "DEVELOPMENT_EVALUATION_COMPLETE", "sources": limit, "rows": len(global_rows), "timestamp": time.time()})


def configure_environment():
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "development_eval_v1"))
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--max-snrs", type=int)
    parser.add_argument("--max-seeds", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / f"failure_{time.time_ns()}.json", {"type": type(error).__name__, "error": repr(error),
            "traceback": traceback.format_exc(), "timestamp": time.time()})
        raise


if __name__ == "__main__":
    main()
