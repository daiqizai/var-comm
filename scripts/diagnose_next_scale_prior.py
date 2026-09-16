#!/usr/bin/env python3
"""Measure next-scale source-prior value before building a channel decoder."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import csv

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from var_comm.next_scale_prior import (
    METRICS, PATCH_NUMS, VOCAB_SIZE, file_sha256, frequency_log_probs, load_models,
    next_scale_log_probs, numerical_selfcheck, paired_mean_interval, preprocess,
    score_target_tokens, state_sha256,
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty results are not a completed experiment")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_contract(path: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    config = yaml.safe_load(path.read_text())
    if config["status"] != "preregistered_before_probability_results" or config["stage"] != "prior_predictiveness_only":
        raise ValueError("not a preregistered prior-only diagnostic")
    model = config["model"]
    if tuple(model["patch_nums"]) != PATCH_NUMS or model["target_scales"] != [8, 9]:
        raise ValueError("full scale schedule or target stages changed")
    if model["cfg"] != 0 or model["temperature"] != 1 or model["top_k"] != "disabled" or model["top_p"] != "disabled":
        raise ValueError("this gate requires untruncated, untempered, no-CFG source probabilities")
    paths = config["paths"]
    for name in ("development_manifest", "split_manifest"):
        if file_sha256(Path(paths[name])) != paths[f"{name}_sha256"]:
            raise RuntimeError(f"changed source manifest: {name}")
    targets = json.loads(Path(paths["development_manifest"]).read_text())
    entries = json.loads(Path(paths["split_manifest"]).read_text())["entries"]
    lookup = {row["image_id"]: row for row in entries}
    if len(lookup) != len(entries) or len(targets) != config["population"]["target_count"]:
        raise ValueError("source population count or unique IDs changed")
    donors = [lookup[row["same_class_donor_id"]] for row in targets]
    for target, donor in zip(targets, donors):
        if target["image_id"] == donor["image_id"] or target["class_index"] != donor["class_index"]:
            raise ValueError("donor must be a different image of the same class")
    target_ids = {row["image_id"] for row in targets}
    donor_ids = {row["image_id"] for row in donors}
    if len(target_ids) != len(targets) or len(donor_ids) != len(donors) or target_ids & donor_ids:
        raise ValueError("target/donor identities overlap or duplicate")
    fit = [row for row in entries if row["split"] == "calibration"]
    if len(fit) != config["population"]["frequency_fit_candidates"]:
        raise ValueError("frequency fit population changed")
    return config, targets, donors, fit


def snapshot_sources(output: Path, config_path: Path, config: dict) -> dict:
    project_paths = [Path(__file__), ROOT / "src/var_comm/__init__.py",
                     ROOT / "src/var_comm/next_scale_prior.py", config_path,
                     ROOT / "reports/next_scale_prior_preregistration_2026-09-07.md"]
    source = Path(config["paths"]["var_source"])
    paths = [(path, "project/" + str(path.relative_to(ROOT))) for path in project_paths]
    paths.extend((path, "upstream/" + str(path.relative_to(source))) for path in sorted(source.rglob("*.py")))
    records = {}
    for path, relative in paths:
        destination = output / "snapshots" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records[str(path)] = {"sha256": file_sha256(path), "snapshot": str(destination.relative_to(output))}
    return records


def encode_populations(vae, targets, donors, fit, config, output, device, mode):
    tokens, manifest, excluded = {}, [], []
    protected_ids, protected_paths, protected_file_hashes, protected_rgb_hashes = set(), set(), set(), set()
    batch_size = config["evaluation"]["encode_batch_size"]
    for role, entries in (("target_development", targets), ("same_class_donor_development", donors), ("frequency_fit", fit)):
        pending = []

        def flush():
            if not pending:
                return
            images = torch.stack([tensor for _record, tensor in pending]).to(device)
            with torch.no_grad():
                continuous = vae.quant_conv(vae.encoder(images))
                encoded = vae.quantize.f_to_idxBl_or_fhat(continuous, to_fhat=False, v_patch_nums=PATCH_NUMS)
            for batch_index, (record, _tensor) in enumerate(pending):
                tokens[record["image_id"]] = [scale[batch_index].detach().cpu() for scale in encoded]
            pending.clear()

        for entry_index, entry in enumerate(entries):
            path = Path(entry["path"]).resolve()
            image, rgb_sha = preprocess(path)
            file_sha = file_sha256(path)
            overlaps = (entry["image_id"] in protected_ids or str(path) in protected_paths
                        or file_sha in protected_file_hashes or rgb_sha in protected_rgb_hashes)
            if overlaps:
                if role == "frequency_fit":
                    excluded.append({"image_id": entry["image_id"], "path": str(path), "reason": "overlap_with_target_or_donor"})
                    continue
                raise RuntimeError("duplicate target/donor image content invalidates the matched-source control")
            record = {**entry, "original_split": entry["split"], "role": role,
                      "path": str(path), "file_sha256": file_sha, "preprocessed_rgb_sha256": rgb_sha}
            manifest.append(record)
            if role != "frequency_fit":
                protected_ids.add(entry["image_id"])
                protected_paths.add(str(path))
                protected_file_hashes.add(file_sha)
                protected_rgb_hashes.add(rgb_sha)
            pending.append((record, image))
            if len(pending) == batch_size:
                flush()
            if (entry_index + 1) % 100 == 0:
                print(f"encode {role} {entry_index + 1}/{len(entries)}", flush=True)
        flush()
    fit_records = [record for record in manifest if record["role"] == "frequency_fit"]
    minimum = config["population"]["minimum_frequency_fit_after_exclusions"] if mode == "full" else 1
    write_json(output / "population_manifest.json", manifest)
    write_json(output / "excluded_frequency_fit.json", excluded)
    if len(fit_records) < minimum:
        raise RuntimeError("insufficient disjoint frequency-fit population")
    identifiers = [record["image_id"] for record in manifest]
    flat_tokens = np.stack([torch.cat(tokens[image_id]).numpy().astype(np.uint16) for image_id in identifiers])
    np.savez(output / "source_tokens.npz", image_ids=np.asarray(identifiers), tokens=flat_tokens,
             roles=np.asarray([record["role"] for record in manifest]),
             classes=np.asarray([record["class_index"] for record in manifest]))
    counts = {}
    for scale in config["model"]["target_scales"]:
        fit_tokens = torch.cat([tokens[record["image_id"]][scale - 1] for record in fit_records])
        counts[scale] = torch.bincount(fit_tokens, minlength=VOCAB_SIZE)
        if int(counts[scale].sum()) != len(fit_records) * PATCH_NUMS[scale - 1] ** 2:
            raise RuntimeError("static frequency token count mismatch")
    np.savez(output / "frequency_counts.npz", **{f"scale_{scale}": value.numpy() for scale, value in counts.items()})
    return tokens, counts, manifest, excluded


@torch.no_grad()
def causality_selfcheck(vae, var, tokens, targets, config, device) -> list[dict]:
    checks = []
    for target in targets[:config["evaluation"]["causality_selfcheck_images"]]:
        source = [item[None].to(device) for item in tokens[target["image_id"]]]
        labels = torch.tensor([target["class_index"]], device=device)
        canonical = torch.log_softmax(var(labels, vae.quantize.idxBl_to_var_input(source)).float(), -1)
        for scale in config["model"]["target_scales"]:
            prior = next_scale_log_probs(var, vae, source[:scale - 1], labels)
            offset = sum(size ** 2 for size in PATCH_NUMS[:scale - 1])
            length = PATCH_NUMS[scale - 1] ** 2
            canonical_scale = canonical[:, offset:offset + length]
            erased = source[:scale - 1] + [torch.zeros_like(item) for item in source[scale - 1:]]
            altered = torch.log_softmax(var(labels, vae.quantize.idxBl_to_var_input(erased)).float(), -1)
            future_effect = float((canonical_scale - altered[:, offset:offset + length]).abs().max())
            forward_error = float((canonical_scale - prior).abs().max())
            if max(future_effect, forward_error) > config["evaluation"]["official_forward_absolute_tolerance"]:
                raise RuntimeError(f"causality/official-forward mismatch at scale {scale}")
            checks.append({"image_id": target["image_id"], "scale": scale,
                           "receiver_vs_official_max_abs": forward_error,
                           "changed_target_and_future_max_abs": future_effect})
    return checks


def evaluate_priors(vae, var, tokens, counts, targets, donors, config, output, device):
    static = {scale: frequency_log_probs(count, config["static_frequency"]["smoothing_pseudocount"]).to(device)
              for scale, count in counts.items()}
    rows = []
    batch_size = config["evaluation"]["prior_batch_size"]
    for start in range(0, len(targets), batch_size):
        target_batch, donor_batch = targets[start:start + batch_size], donors[start:start + batch_size]
        labels = torch.tensor([row["class_index"] for row in target_batch], device=device)
        source_tokens = [torch.stack([tokens[row["image_id"]][index] for row in target_batch]).to(device)
                         for index in range(len(PATCH_NUMS))]
        donor_tokens = [torch.stack([tokens[row["image_id"]][index] for row in donor_batch]).to(device)
                        for index in range(len(PATCH_NUMS))]
        cache_arrays = [{} for _target in target_batch]
        for scale in config["model"]["target_scales"]:
            ground_truth = source_tokens[scale - 1]
            shape = (*ground_truth.shape, VOCAB_SIZE)
            for prior_name in config["controls"]:
                synchronize(device)
                tick = time.perf_counter()
                if prior_name == "uniform":
                    log_probs = torch.full((1, 1, VOCAB_SIZE), -math.log(VOCAB_SIZE), device=device, dtype=torch.float64).expand(shape)
                elif prior_name == "static_scale_frequency":
                    log_probs = static[scale][None, None].expand(shape)
                else:
                    context = source_tokens if prior_name == "true_source_prefix" else donor_tokens
                    log_probs = next_scale_log_probs(var, vae, context[:scale - 1], labels)
                    for batch_index, arrays in enumerate(cache_arrays):
                        arrays[f"{prior_name}_scale_{scale}"] = log_probs[batch_index].cpu().numpy()
                synchronize(device)
                elapsed = (time.perf_counter() - tick) / len(target_batch)
                metrics, token_nll = score_target_tokens(log_probs, ground_truth)
                for batch_index, target in enumerate(target_batch):
                    rows.append({"image_index": start + batch_index, "image_id": target["image_id"],
                                 "class_index": target["class_index"], "donor_id": donor_batch[batch_index]["image_id"],
                                 "scale": scale, "tokens": ground_truth.shape[1], "prior": prior_name,
                                 **{name: float(values[batch_index]) for name, values in metrics.items()},
                                 "token_nll_p95_bits": float(torch.quantile(token_nll[batch_index], 0.95)),
                                 "token_nll_max_bits": float(token_nll[batch_index].max()),
                                 "prior_table_seconds_per_image": elapsed})
        for batch_index, arrays in enumerate(cache_arrays):
            np.savez(output / "probabilities" / f"{start + batch_index:03d}.npz", **arrays)
        write_csv(output / "per_image_prior_metrics.csv", rows)
        print(f"prior tables {min(start + batch_size, len(targets))}/{len(targets)} rows={len(rows)}", flush=True)
    return rows


def summarize(rows, config):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["scale"], row["prior"])
        if row["image_id"] in grouped[key]:
            raise RuntimeError("duplicate source/scale/prior entry")
        grouped[key][row["image_id"]] = row
    summary, comparisons, verdict = [], [], {}
    for scale in config["model"]["target_scales"]:
        reference = grouped[(scale, "true_source_prefix")]
        image_ids = sorted(reference)
        for prior in config["controls"]:
            group = grouped[(scale, prior)]
            if set(group) != set(reference):
                raise RuntimeError("unpaired source populations")
            row = {"scale": scale, "prior": prior, "images": len(group)}
            row.update({metric: float(np.mean([group[image_id][metric] for image_id in image_ids]))
                        for metric in (*METRICS, "prior_table_seconds_per_image")})
            summary.append(row)
            if prior == "true_source_prefix":
                continue
            for metric in ("nll_bits_per_token", "bit_marginal_nll_bits_per_token"):
                differences = np.asarray([group[image_id][metric] - reference[image_id][metric] for image_id in image_ids])
                mean, lower, upper = paired_mean_interval(
                    differences, seed=config["evaluation"]["bootstrap_seed"], resamples=config["evaluation"]["bootstrap_resamples"]
                )
                comparisons.append({"scale": scale, "control": prior, "metric": metric,
                                    "control_minus_true_gain": mean, "ci_low": lower, "ci_high": upper,
                                    "true_prefix_win_fraction": float((differences > 0).mean())})
        primary = [row for row in comparisons if row["scale"] == scale and row["metric"] == "nll_bits_per_token"]
        checks = {row["control"]: row["control_minus_true_gain"] >= config["gate"]["minimum_mean_gain_bits_per_token_over_each_control"]
                  and row["ci_low"] > 0 for row in primary}
        verdict[str(scale)] = {"passed": all(checks.values()), "checks": checks}
    return summary, comparisons, verdict


def write_report(output, summary, comparisons, verdict, mode):
    lines = ["# Next-scale source-prior diagnostic", "", f"Mode: {mode}. Only prior predictiveness; no FEC/CRC or image-quality claim.", "",
             "| Scale | Prior | NLL bits/token | Ideal bits/scale | Entropy | Top1 | Bit-marginal NLL |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['scale']} | {row['prior']} | {row['nll_bits_per_token']:.5f} | "
                     f"{row['ideal_cross_entropy_bits_per_scale']:.2f} | {row['predictive_entropy_bits_per_token']:.5f} | "
                     f"{row['top1_match_rate']:.5f} | {row['bit_marginal_nll_bits_per_token']:.5f} |")
    lines.extend(["", "## Paired information gain", "", "Positive = true source prefix has lower NLL. Image bootstrap; fixed fit table and donor assignment.", "",
                  "| Scale | Control | Metric | Control minus true | 95% CI |", "|---|---|---|---:|---|"])
    for row in comparisons:
        lines.append(f"| {row['scale']} | {row['control']} | {row['metric']} | {row['control_minus_true_gain']:.5f} | "
                     f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}] |")
    lines.extend(["", "```json", json.dumps(verdict, indent=2), "```", "",
                  "Correct prefixes are oracle capability conditions, not free received information.",
                  "Top1 uses deterministic lowest-index tie-breaking; the uniform row is not a meaningful argmax predictor.",
                  "Ideal cross entropy is not an actual entropy-coded bitstream length or a channel coding gain.",
                  "Bit-marginal priors discard within-token dependence; do not call their decoder exact token-MAP.",
                  "Prior table timing includes prefix traversal and FP32 host-cache transfer, excludes source encoding and scoring.",
                  "Tables are computed once per context; online reuse between accepted scales is not implemented in this diagnostic."])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/next_scale_prior_diagnostic.yaml")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if args.self_check_only:
        print(json.dumps(numerical_selfcheck(), indent=2))
        return
    config_path = args.config.resolve()
    config, targets, donors, fit = load_contract(config_path)
    if args.mode == "smoke":
        targets, donors, fit = targets[:2], donors[:2], fit[:16]
    output = (args.output_dir or ROOT / config["outputs"][args.mode]).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("all new outputs must physically stay under VAR_COMM/outputs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "probabilities").mkdir()
    records = snapshot_sources(output, config_path, config)
    metadata = {"local_started": datetime.now().astimezone().isoformat(), "command": sys.argv,
                "mode": args.mode, "source_files": records, "config_sha256": file_sha256(config_path),
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                "checkpoints": {name: {"path": config['paths'][name], "sha256": config['paths'][f'{name}_sha256']}
                                for name in ("vae_checkpoint", "var_checkpoint")}}
    write_json(output / "run_metadata.json", metadata)
    started = time.perf_counter()
    try:
        checks = numerical_selfcheck()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device(args.device)
        for name in ("vae_checkpoint", "var_checkpoint"):
            if file_sha256(Path(config["paths"][name])) != config["paths"][f"{name}_sha256"]:
                raise RuntimeError(f"checkpoint SHA mismatch: {name}")
        vae, var = load_models(config["paths"], device)
        before = {"vae": state_sha256(vae), "var": state_sha256(var)}
        tokens, counts, manifest, excluded = encode_populations(vae, targets, donors, fit, config, output, device, args.mode)
        causal = causality_selfcheck(vae, var, tokens, targets, config, device)
        write_json(output / "selfcheck.json", {"numerical": checks, "causality": causal})
        rows = evaluate_priors(vae, var, tokens, counts, targets, donors, config, output, device)
        expected_rows = len(targets) * len(config["model"]["target_scales"]) * len(config["controls"])
        if len(rows) != expected_rows:
            raise RuntimeError("incomplete prior evaluation")
        summary, comparisons, verdict = summarize(rows, config)
        write_csv(output / "summary.csv", summary)
        write_csv(output / "paired_gains.csv", comparisons)
        write_report(output, summary, comparisons, verdict, args.mode)
        after = {"vae": state_sha256(vae), "var": state_sha256(var)}
        if before != after or any(parameter.requires_grad or parameter.grad is not None for model in (vae, var) for parameter in model.parameters()):
            raise RuntimeError("model freeze boundary violated")
        for path, entry in records.items():
            if file_sha256(Path(path)) != entry["sha256"]:
                raise RuntimeError("code/config changed during execution")
        status = "PASS_PRIOR_GATE_ONLY" if all(row["passed"] for row in verdict.values()) else "STOP_UNCALIBRATED_PRIOR_PROTOTYPE"
        if args.mode == "smoke":
            status = "SMOKE_COMPLETE_NOT_SCIENTIFIC_RESULT"
        result = {**metadata, "status": status, "target_count": len(targets), "rows": len(rows),
                  "population_roles": dict(Counter(row["role"] for row in manifest)), "exclusions": len(excluded),
                  "gate": verdict, "frozen_before": before, "frozen_after": after,
                  "elapsed_seconds": time.perf_counter() - started,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  "output_hashes": {str(path.relative_to(output)): file_sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}}
        write_json(output / "completion.json", result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("source_files", "output_hashes")}, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED_CLOSED", "error": repr(error),
                                              "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
