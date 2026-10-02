"""Select a metric batch on synthetic inputs after all replay models are loaded.

This is an engineering throughput check, never a scientific evaluation or a
policy search. Existing receipts are reused only under identical bound inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


STATUS = "METRIC_BATCH_QUALIFICATION_PASS"
RECEIPT_NAME = "metric_batch_qualification.json"
FLOAT_FIELDS = ("clip_image_cosine", "dists", "dreamsim", "ms_ssim", "dinov2_vitl14_cosine")
EXACT_FIELDS = ("resnet50_prediction", "resnet50_source_prediction", "resnet50_top1_label",
                "resnet50_top1_source_prediction", "resnet50_source_top1_label", "label_conditioned")
PROTOCOL = dict(version=1, candidate_batch_sizes=[1, 2, 4, 8, 16], fixture_count=16,
                absolute_tolerance=2e-5, classification_requires_exact_agreement=True,
                warmup_full_passes=1, timed_full_passes=3, max_reserved_memory_fraction=.88,
                dtype="float32", automatic_mixed_precision=False, metric_formula_changed=False,
                selection="minimum_median_seconds_per_image_then_smaller_batch")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def compare_rows(rows, scalar, tolerance=PROTOCOL["absolute_tolerance"]):
    """Compare every registered score, including both classification meanings."""
    if len(rows) != len(scalar) or not rows:
        raise ValueError("Batch comparison row count differs")
    maximum = {name: 0. for name in FLOAT_FIELDS}
    exact = True
    expected_fields = set(FLOAT_FIELDS + EXACT_FIELDS)
    for actual, expected in zip(rows, scalar):
        if set(actual) != expected_fields or set(expected) != expected_fields:
            raise ValueError("Batch comparison metric fields differ")
        for name in FLOAT_FIELDS:
            a, b = actual[name], expected[name]
            if (isinstance(a, bool) or isinstance(b, bool) or
                    not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or
                    not math.isfinite(a) or not math.isfinite(b)):
                raise ValueError("Nonfinite or invalid batch metric: " + name)
            maximum[name] = max(maximum[name], abs(a - b))
        for name in EXACT_FIELDS:
            if type(actual[name]) is not type(expected[name]) or actual[name] != expected[name]:
                exact = False
    return dict(max_absolute_deltas=maximum, classification_exact=exact,
                passed=exact and all(value <= tolerance for value in maximum.values()))


def select_candidate(candidates):
    eligible = [item for item in candidates if item.get("eligible") is True]
    if not any(item["batch_size"] == 1 for item in eligible):
        raise RuntimeError("Scalar batch did not pass numerical and memory qualification")
    return min(eligible, key=lambda item: (item["median_seconds_per_image"], item["batch_size"]))["batch_size"]


def qualified_chunks(count, chosen, qualified):
    """Partition a tail using only successfully checked powers of two."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Invalid pending count")
    valid = set(PROTOCOL["candidate_batch_sizes"])
    if (not isinstance(chosen, int) or isinstance(chosen, bool) or chosen not in valid or
            any(type(item) is not int or item not in valid for item in qualified) or
            chosen not in qualified or 1 not in qualified):
        raise ValueError("Invalid qualified batch sizes")
    sizes = sorted({item for item in qualified if item <= chosen}, reverse=True)
    result = []
    while count:
        batch = next(item for item in sizes if item <= count)
        result.append(batch)
        count -= batch
    return result


def validate_receipt(receipt, binding):
    if (receipt.get("status") != STATUS or receipt.get("binding") != binding or
            receipt.get("protocol") != PROTOCOL or receipt.get("synthetic_images") is not True or
            receipt.get("scientific_result") is not False or receipt.get("real_model_weights") is not True or
            receipt.get("source_or_development_images_used") is not False or
            receipt.get("rng_state_preserved") is not True or
            receipt.get("metric_evaluator_identity") != binding["metric_evaluator_identity"] or
            receipt.get("modelmanifest_sha256") != binding["modelmanifest_sha256"] or
            receipt.get("source_bindings") != binding["source_bindings"]):
        raise RuntimeError("Metric batch qualification binding differs on resume")
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    if receipt.get("payload_sha256") != digest(payload):
        raise RuntimeError("Metric batch qualification checksum differs")
    candidates = receipt.get("candidates", [])
    if [item.get("batch_size") for item in candidates] != PROTOCOL["candidate_batch_sizes"]:
        raise RuntimeError("Metric batch candidate coverage differs")
    for item in candidates:
        if item.get("eligible") is True:
            comparison = item.get("comparison", {})
            if (item.get("status") != "PASS" or comparison.get("passed") is not True or
                    comparison.get("classification_exact") is not True or
                    set(comparison.get("max_absolute_deltas", {})) != set(FLOAT_FIELDS) or
                    any(not math.isfinite(v) or v < 0 or v > PROTOCOL["absolute_tolerance"]
                        for v in comparison["max_absolute_deltas"].values()) or
                    item.get("peak_reserved_bytes", math.inf) > binding["device"]["total_memory_bytes"] * PROTOCOL["max_reserved_memory_fraction"]):
                raise RuntimeError("Invalid eligible metric batch")
            times = item.get("repeat_seconds_per_image", [])
            if (len(times) != PROTOCOL["timed_full_passes"] or
                    any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in times) or
                    item.get("median_seconds_per_image") != statistics.median(times)):
                raise RuntimeError("Invalid metric batch timing")
    chosen = select_candidate(candidates)
    qualified = [item["batch_size"] for item in candidates if item.get("eligible") is True]
    if receipt.get("chosen_batch_size") != chosen or receipt.get("qualified_batch_sizes") != qualified:
        raise RuntimeError("Metric batch selection differs")
    qualified_chunks(16, chosen, qualified)
    return receipt


def synthetic_fixtures(torch):
    """One analytic RGB source and 16 distinct corruptions; no random generator."""
    axis = torch.linspace(0., 1., 256, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    source = torch.stack((.12 + .73 * xx, .18 + .66 * yy,
                          .5 + .32 * torch.sin(13. * xx + 7. * yy)), dim=0).unsqueeze(0)
    images = []
    for index in range(16):
        shifted = torch.roll(source, shifts=(index - 7, 2 * index - 15), dims=(-2, -1))
        color = torch.tensor([index % 3 - 1, (index + 1) % 3 - 1, (index + 2) % 3 - 1],
                             dtype=torch.float32).reshape(1, 3, 1, 1) * .012
        images.append((shifted * (1 - .005 * index) + color + .002 * index).clamp(0, 1))
    return source.contiguous(), torch.cat(images, dim=0).contiguous()


def qualify_and_select_batch(evaluator, reference_device, manifest_path, out_dir, *, runtime_context):
    """Return a bound PASS receipt; never silently requalify a stale receipt.

    Call after loading the frozen replay engine and all metric models. CUDA's
    peak reservation includes these live models. No runtime flag is modified.
    """
    import torch
    import metric_models

    device = torch.device(reference_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal metric batch qualification requires the scoring CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    hardware = dict(type="cuda", index=device.index, name=properties.name,
                    total_memory_bytes=properties.total_memory,
                    compute_capability=[properties.major, properties.minor])
    if getattr(properties, "uuid", None) is not None:
        hardware["uuid"] = str(properties.uuid)
    paths = (Path(__file__).resolve(), Path(metric_models.__file__).resolve())
    sources = {str(path): sha(path) for path in paths}
    manifest_path = Path(manifest_path).resolve()
    binding = dict(metric_evaluator_identity=evaluator.identity(), modelmanifest_path=str(manifest_path),
                   modelmanifest_sha256=sha(manifest_path), source_bindings=sources,
                   device=hardware, runtime_context=runtime_context)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RECEIPT_NAME
    if path.exists():
        return validate_receipt(json.loads(path.read_text()), binding)

    started = time.monotonic()
    receipt = dict(status="RUNNING", binding=binding, protocol=PROTOCOL, synthetic_images=True,
                   real_model_weights=True, scientific_result=False, source_or_development_images_used=False,
                   training_updates=0, policy_selection_updates=0, source_bindings=sources,
                   metric_evaluator_identity=binding["metric_evaluator_identity"],
                   modelmanifest_sha256=binding["modelmanifest_sha256"], candidates=[],
                   live_context_memory_allocated_bytes=torch.cuda.memory_allocated(device),
                   live_context_memory_reserved_bytes=torch.cuda.memory_reserved(device))
    try:
        # Keep the frozen generator's RNG stream unchanged even if an imported
        # evaluator implementation unexpectedly consumes random numbers.
        with torch.random.fork_rng(devices=[device.index]), metric_models.no_network():
            source, images = synthetic_fixtures(torch)
            receipt["fixture_source_sha256"] = metric_models.image_digest(source)
            receipt["fixture_reconstructions_sha256"] = metric_models.image_digest(images)
            prepared = evaluator.prepare_reference(source)
            truth = int(prepared["resnet50"][0])
            scalar = []
            for index in range(16):
                scalar.extend(evaluator.score(source, images[index:index + 1],
                              torch.tensor([truth], dtype=torch.int64), prepared=prepared, label_conditioned=False))
            compare_rows(scalar, scalar)
            receipt["scalar_reference_scores"] = scalar

            def full_pass(batch):
                result = []
                for offset in range(0, len(images), batch):
                    reference, cached = evaluator.expand_reference(source, prepared, batch)
                    result.extend(evaluator.score(reference, images[offset:offset + batch],
                                  torch.tensor([truth] * batch, dtype=torch.int64),
                                  prepared=cached, label_conditioned=False))
                return result

            for batch in PROTOCOL["candidate_batch_sizes"]:
                item = dict(batch_size=batch, eligible=False, status="RUNNING")
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                try:
                    warm = full_pass(batch)
                    comparison = compare_rows(warm, scalar)
                    times = []
                    for _ in range(PROTOCOL["timed_full_passes"]):
                        torch.cuda.synchronize(device)
                        tick = time.perf_counter()
                        values = full_pass(batch)
                        torch.cuda.synchronize(device)
                        times.append((time.perf_counter() - tick) / len(images))
                        check = compare_rows(values, scalar)
                        comparison["classification_exact"] &= check["classification_exact"]
                        comparison["passed"] &= check["passed"]
                        for metric in FLOAT_FIELDS:
                            comparison["max_absolute_deltas"][metric] = max(
                                comparison["max_absolute_deltas"][metric], check["max_absolute_deltas"][metric])
                    item.update(comparison=comparison, repeat_seconds_per_image=times,
                                median_seconds_per_image=statistics.median(times),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
                    if item["peak_reserved_bytes"] > properties.total_memory * PROTOCOL["max_reserved_memory_fraction"]:
                        item["status"] = "REJECT_MEMORY_RESERVATION"
                    elif not comparison["passed"]:
                        item["status"] = "REJECT_NUMERICAL_AGREEMENT"
                    else:
                        item.update(status="PASS", eligible=True)
                except torch.cuda.OutOfMemoryError as error:
                    item.update(status="REJECT_CUDA_OOM", error=str(error),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
                receipt["candidates"].append(item)
                torch.cuda.empty_cache()
        receipt["rng_state_preserved"] = True
        receipt["chosen_batch_size"] = select_candidate(receipt["candidates"])
        receipt["qualified_batch_sizes"] = [x["batch_size"] for x in receipt["candidates"] if x["eligible"]]
        if sha(manifest_path) != binding["modelmanifest_sha256"] or any(sha(p) != v for p, v in sources.items()):
            raise RuntimeError("Metric qualification input changed during measurement")
        if evaluator.identity() != binding["metric_evaluator_identity"]:
            raise RuntimeError("Metric evaluator changed during batch qualification")
        receipt.update(status=STATUS, elapsed_seconds=time.monotonic() - started)
        receipt["payload_sha256"] = digest(receipt)
        validate_receipt(receipt, binding)
        write(path, receipt)
        return receipt
    except Exception as error:
        receipt.update(status="FAILED", error=str(error), elapsed_seconds=time.monotonic() - started)
        write(out_dir / "metric_batch_qualification_failed.json", receipt)
        raise
