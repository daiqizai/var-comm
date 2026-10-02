"""Real-weight contract checks on synthetic pixels, not scientific evaluation.

All six registered evaluators must load from local, verified assets. The default
CPU-only run uses two threads and processes one 256x256 image at a time. No source
image, communication model, development reconstruction, or policy is consulted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import time
import traceback

import metric_models as mm

STATUS_PASS = "REAL_MODEL_WEIGHTS_QUALIFICATION_PASS"
EXPECTED = ("clip", "dists", "resnet50", "dinov2_vitl14", "dreamsim", "ms_ssim")
METRIC_FIELDS = ("clip_image_cosine", "dists", "dinov2_vitl14_cosine", "dreamsim", "ms_ssim")
BATCH_COMPARISON_ATOL = 2e-5


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_bindings():
    result, stats = {}, {}
    for path in (Path(__file__).resolve(), Path(mm.__file__).resolve()):
        result[str(path)] = mm.sha256_file(path)
        st = path.stat()
        stats[str(path)] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    return result, stats


def fixtures():
    """Two analytical, nonconstant RGB images, independent of model outputs."""
    import torch
    axis = torch.linspace(0.0, 1.0, 256, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    checker = ((torch.floor(xx * 12) + torch.floor(yy * 9)) % 2).float()
    radial = (torch.sin(22 * torch.sqrt((xx-.42).square() + (yy-.57).square())) + 1) / 2
    stripes = (torch.cos(25 * xx + 7 * yy) + 1) / 2
    first = torch.stack((.15 + .6*xx + .2*checker, .1 + .6*yy + .25*radial, .1 + .6*stripes + .25*xx)).clamp(0, 1)[None]
    second = torch.stack((.15 + .5*radial + .3*yy, .1 + .6*checker + .25*xx, .2 + .45*yy + .25*stripes)).clamp(0, 1)[None]
    return [first.contiguous(), second.contiguous()]


def perturb(x):
    """Fixed spatial and color corruption; no stochastic search for a score."""
    import torch
    result = x[:, [2, 0, 1]].flip(-1).clone()
    result[:, :, 64:192, 80:176] = torch.tensor([0.05, 0.9, 0.1], dtype=result.dtype).view(1, 3, 1, 1)
    return result.contiguous()


def run(manifest_path, out_dir, device="cpu"):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    manifest_path = Path(manifest_path).resolve(strict=True)
    out_dir = Path(out_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = set(EXPECTED) - set(manifest)
    if missing:
        raise ValueError(f"Qualification requires all six metric registrations: {sorted(missing)}")
    bindings, file_stats = source_bindings()
    receipt = {
        "status": "RUNNING",
        "started_utc": now(),
        "synthetic_images": True,
        "real_model_weights": True,
        "scientific_result": False,
        "training": False,
        "policy_selection": False,
        "manifest_path": str(manifest_path),
        "modelmanifest_sha256": mm.sha256_file(manifest_path),
        "source_bindings": bindings,
        "source_file_stats": file_stats,
        "device": device,
        "cpu_threads": 2,
        "cpu_interop_threads": 2,
        "fixture_batch_size": 1,
        "fixture_count": 2,
        "batch_qualification": {"batch_size": 2, "reference_baseline_batch_size": 1,
            "absolute_tolerance": BATCH_COMPARISON_ATOL, "classification_requires_exact_agreement": True,
            "qualifies_larger_gpu_batch_sizes": False, "metric_formula_changed": False},
        "passed_checks": [],
        "model_stats": {},
        "fixture_results": [],
    }
    start = time.monotonic()
    checks = receipt["passed_checks"]

    def check(name, condition, detail=None):
        if not bool(condition):
            raise AssertionError(f"Qualification failed: {name}; detail={detail}")
        checks.append({"name": name, "detail": detail})

    destination = out_dir / "models_qualification.json"
    atomic_json(destination, receipt)
    try:
        load_start = time.monotonic()
        with mm.no_network():
            evaluator = mm.MetricEvaluator(manifest, device=device, requested=EXPECTED)
            receipt["model_load_seconds"] = time.monotonic() - load_start
            receipt["metadata"] = evaluator.metadata
            receipt["evaluator_identity"] = evaluator.identity()
            check("all_six_models_loaded", set(evaluator.models) == set(EXPECTED))
            for name in EXPECTED:
                check(f"{name}:registered_ready", evaluator.metadata["metrics"][name]["status"] == "READY")
                model = evaluator.models[name]
                if name == "ms_ssim":
                    receipt["model_stats"][name] = {"parameter_count": 0, "function": "official ms_ssim", "frozen": True}
                    continue
                params = list(model.parameters())
                check(f"{name}:eval_mode", all(not module.training for module in model.modules()))
                check(f"{name}:frozen_parameters", all(not p.requires_grad for p in params))
                check(f"{name}:fp32_parameters", all(p.dtype == torch.float32 for p in params if p.is_floating_point()))
                receipt["model_stats"][name] = {"parameter_count": sum(p.numel() for p in params), "tensor_count": len(params), "frozen": True, "eval_mode": True, "dtype": "float32"}

            for index, image in enumerate(fixtures()):
                fixture_start = time.monotonic()
                check(f"fixture{index}:registered_shape", tuple(image.shape) == (1, 3, 256, 256))
                check(f"fixture{index}:nonconstant", image.std().item() > 0.05)
                prepared = evaluator.prepare_reference(image)
                pred = int(prepared["resnet50"][0])
                identity = evaluator.score(image, image, [pred], prepared=prepared)[0]
                corrupted = perturb(image)
                changed = evaluator.score(image, corrupted, [pred], prepared=prepared)[0]
                changed_label = (pred + 1) % 1000
                # The identical reconstruction is sufficient to test label leakage:
                # changing scoring labels/flags must never change any model output.
                alternate = evaluator.score(image, image, [changed_label], label_conditioned=True, prepared=prepared)[0]
                flag_only = evaluator.score(image, image, [pred], label_conditioned=True, prepared=prepared)[0]
                check(f"fixture{index}:clip_identity", abs(identity["clip_image_cosine"] - 1) <= 2e-5, identity["clip_image_cosine"])
                check(f"fixture{index}:dinov2_vitl14_identity", abs(identity["dinov2_vitl14_cosine"] - 1) <= 2e-5, identity["dinov2_vitl14_cosine"])
                check(f"fixture{index}:dists_identity", abs(identity["dists"]) <= 5e-5, identity["dists"])
                check(f"fixture{index}:dreamsim_identity", abs(identity["dreamsim"]) <= 2e-5, identity["dreamsim"])
                check(f"fixture{index}:ms_ssim_identity", abs(identity["ms_ssim"] - 1) <= 2e-5, identity["ms_ssim"])
                for metric in METRIC_FIELDS:
                    check(f"fixture{index}:{metric}:finite_corruption", math.isfinite(changed[metric]), changed[metric])
                    check(f"fixture{index}:{metric}:label_invariance", identity[metric] == alternate[metric])
                    check(f"fixture{index}:{metric}:flag_only_invariance", identity[metric] == flag_only[metric])
                check(f"fixture{index}:clip_detects_corruption", identity["clip_image_cosine"] - changed["clip_image_cosine"] > 1e-6, changed["clip_image_cosine"])
                check(f"fixture{index}:dinov2_vitl14_detects_corruption", identity["dinov2_vitl14_cosine"] - changed["dinov2_vitl14_cosine"] > 1e-6, changed["dinov2_vitl14_cosine"])
                check(f"fixture{index}:dists_detects_corruption", changed["dists"] - identity["dists"] > 1e-6, changed["dists"])
                check(f"fixture{index}:dreamsim_detects_corruption", changed["dreamsim"] - identity["dreamsim"] > 1e-6, changed["dreamsim"])
                check(f"fixture{index}:ms_ssim_detects_corruption", identity["ms_ssim"] - changed["ms_ssim"] > 1e-6, changed["ms_ssim"])
                check(f"fixture{index}:classification_identity_both_definitions", identity["resnet50_top1_label"] and identity["resnet50_top1_source_prediction"] and identity["resnet50_source_top1_label"])
                check(f"fixture{index}:classifier_label_invariance", identity["resnet50_prediction"] == alternate["resnet50_prediction"] == pred and identity["resnet50_source_prediction"] == alternate["resnet50_source_prediction"] == pred)
                check(f"fixture{index}:classifier_flag_only_invariance", flag_only["resnet50_prediction"] == pred and flag_only["resnet50_top1_label"] and flag_only["resnet50_top1_source_prediction"])
                check(f"fixture{index}:classifier_denominators_separate", not alternate["resnet50_top1_label"] and not alternate["resnet50_source_top1_label"] and alternate["resnet50_top1_source_prediction"])
                check(f"fixture{index}:conditioned_flag_preserved", not identity["label_conditioned"] and alternate["label_conditioned"])

                repeated_source, repeated_cache = evaluator.expand_reference(image, prepared, 2)
                batch_rows = evaluator.score(repeated_source, torch.cat((image, corrupted), dim=0),
                    [pred, pred], prepared=repeated_cache)
                batch_deltas = []
                for batch_index, (batched, scalar) in enumerate(zip(batch_rows, (identity, changed))):
                    deltas = {metric: abs(batched[metric]-scalar[metric]) for metric in METRIC_FIELDS}
                    for metric, delta in deltas.items():
                        check(f"fixture{index}:batch{batch_index}:{metric}:scalar_agreement", delta <= BATCH_COMPARISON_ATOL, delta)
                    for name in ("resnet50_prediction", "resnet50_source_prediction", "resnet50_top1_label",
                            "resnet50_top1_source_prediction", "resnet50_source_top1_label", "label_conditioned"):
                        check(f"fixture{index}:batch{batch_index}:{name}:exact_agreement", batched[name] == scalar[name])
                    batch_deltas.append(deltas)

                invalid_identity = dict(prepared, evaluator_identity="0" * 64)
                try:
                    evaluator.score(image, image, [pred], prepared=invalid_identity)
                except ValueError as error:
                    check(f"fixture{index}:cache_evaluator_binding", "cache" in str(error).lower())
                else:
                    raise AssertionError("Cache accepted a different evaluator identity")
                try:
                    evaluator.score(corrupted, corrupted, [pred], prepared=prepared)
                except ValueError as error:
                    check(f"fixture{index}:cache_pixel_binding", "cache" in str(error).lower())
                else:
                    raise AssertionError("Cache accepted different reference pixels")
                receipt["fixture_results"].append({"index": index, "source_pixel_sha256": mm.image_digest(image), "corrupted_pixel_sha256": mm.image_digest(corrupted), "source_prediction": pred, "identity": identity, "corruption": changed, "changed_scoring_label": changed_label, "changed_label_and_flag": alternate, "changed_flag_only": flag_only, "batch2": batch_rows, "batch2_vs_scalar_absolute_deltas": batch_deltas, "elapsed_seconds": time.monotonic() - fixture_start})
                atomic_json(destination, receipt)
                print(json.dumps({"fixture_completed": index + 1, "checks_passed": len(checks)}, sort_keys=True), flush=True)

            for name in EXPECTED:
                if name != "ms_ssim":
                    model = evaluator.models[name]
                    check(f"{name}:no_gradients_after_scoring", all(p.grad is None and not p.requires_grad for p in model.parameters()))
                    check(f"{name}:still_eval_after_scoring", all(not module.training for module in model.modules()))

        check("source_files_unchanged", source_bindings()[0] == bindings)
        check("manifest_unchanged", mm.sha256_file(manifest_path) == receipt["modelmanifest_sha256"])
        receipt["status"] = STATUS_PASS
    except Exception as error:
        receipt["status"] = "REAL_MODEL_WEIGHTS_QUALIFICATION_FAILED"
        receipt["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        receipt["finished_utc"] = now()
        receipt["elapsed_seconds"] = time.monotonic() - start
        receipt["passed_check_count"] = len(checks)
        atomic_json(destination, receipt)
        stamp = receipt["finished_utc"].replace(":", "").replace("+", "_").replace(".", "_")
        atomic_json(out_dir / "models_qualification_attempts" / f"{stamp}_{os.getpid()}.json", receipt)
    print(json.dumps({"status": receipt["status"], "passed_checks": len(checks), "receipt": str(destination), "elapsed_seconds": receipt["elapsed_seconds"]}, sort_keys=True), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.manifest, args.out_dir, args.device)
