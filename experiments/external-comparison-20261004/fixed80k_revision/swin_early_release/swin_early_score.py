"""Read committed float reconstructions with the original unified metric suite."""
from __future__ import annotations
from collections import defaultdict
import os
from pathlib import Path
import sys
import time
import numpy as np
from swin_early_common import *


def score(config, config_path, progress, stopped):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from step0_reference_metrics import load_suite, numeric_flags
    root, out, result = Path(config["root"]), Path(config["output"]), Path(config["result"])
    result.mkdir(parents=True, exist_ok=True)
    reconstruction_path = out / "reconstruction_completion.json"
    reconstruction = read(reconstruction_path)
    if (reconstruction.get("status") != "SWIN_EARLY_RECONSTRUCTIONS_COMPLETE"
            or reconstruction.get("synthetic") is not False or reconstruction.get("rows") != ROWS
            or reconstruction.get("sources") != SOURCES or reconstruction.get("physical_frames") != PHYSICAL_FRAMES
            or reconstruction.get("sampler_step_limit") is not None):
        raise RuntimeError("Full registered physical reconstructions are not complete")
    verify(reconstruction["bindings"]); verify(reconstruction["outputs"])
    source_registration = read(out / "reconstruction_registration.json")
    if identity(source_registration) != reconstruction["binding"]:
        raise RuntimeError("Reconstruction registration does not match completed caches")
    records = source_registration["source_identity"]
    own = code_bindings()
    if (out / "completion.json").exists():
        completed = read(out / "completion.json")
        if completed.get("reconstruction_completion_sha256") != sha(reconstruction_path):
            raise RuntimeError("Final evaluation has a different reconstruction population")
        verify(completed["bindings"]); verify(completed["outputs"])
        progress("COMPLETE", sources_complete=SOURCES, rows=ROWS); return
    if stopped(): raise PauseRequested("Stop requested before metric model loading")
    gpu_available()
    # load_suite reuses the admitted historical quality paths, weights, source
    # hashes, original unified numerical flags, and all six new metric loaders.
    evaluator, metric_module, native, lpips, dino, model_bindings, flags = load_suite(root, "cuda:0")
    shared = root / "experiments/unified-metrics-20261002"
    sys.path.insert(0, str(shared))
    import runner
    import replay
    import analysis
    from batch_speed import qualify_and_select_batch, qualified_chunks
    original_registration_path = root / "results/unified_metrics_20261002/metrics_registration.json"
    original = read(original_registration_path)
    if evaluator.identity() != original["metric_evaluator_identity"]:
        raise RuntimeError("New external and original metric evaluator identities differ")
    metric_dir = out / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / "outputs/UNIFIED-METRICS-20261002/modelmanifest.json"
    manifest_sha = sha(manifest)
    batch = qualify_and_select_batch(evaluator, torch.device("cuda:0"), manifest, metric_dir,
        runtime_context=dict(study="SWIN_ONLY_EARLY_FIXED80K_20261004", numerical_runtime=flags,
            reconstruction_completion_sha256=sha(reconstruction_path), model_bindings=model_bindings))
    if numeric_flags(torch) != flags:
        raise RuntimeError("Batch qualification changed the frozen metric numerical flags")
    expected = {}
    for spec in frame_specs(0):
        from external_eval import base_row
        for method in METHODS:
            sample = base_row(spec, method, records[0], source_registration["source_float_sha256"][0])
            key = analysis.group(sample)
            expected[key] = analysis.context(key)
    contrasts = []  # One actual method; no invented HiFi comparison.
    registration = dict(schema_version=1, status="SWIN_EARLY_METRICS_REGISTERED", synthetic=False,
        training_updates=0, policy_selection_updates=0, new_metrics_used_for_selection=False,
        source_ids=[r["image_id"] for r in records], expected_groups=list(expected.values()),
        source_identity=records, source_count=SOURCES, expected_frames=ROWS, contrasts=contrasts,
        source_bindings=own, reconstruction_completion_sha256=sha(reconstruction_path),
        modelmanifest_path=str(manifest), modelmanifest_sha256=manifest_sha,
        metric_evaluator_identity=evaluator.identity(), model_bindings=model_bindings,
        metric_batch_size=batch["chosen_batch_size"], metric_qualified_batch_sizes=batch["qualified_batch_sizes"],
        metric_batch_qualification_sha256=sha(metric_dir / "metric_batch_qualification.json"),
        numerical_runtime=flags, metric_availability={k: "READY" for k in analysis.NEW_METRICS},
        replay_parity_passed=True, replay_parity_basis="actual_waveform_and_cached_float_pixel_identity",
        reference_targets="unchanged_admitted_FINAL_P2048_P3060_float_RGB",
        bootstrap_seed=analysis.SEED, bootstrap_replicates=analysis.REPLICATES,
        classifier_confidence_threshold=.5, mismatch_derangement_seed=20260930,
        KID="DEFERRED_HOLDOUT", FID="NOT_EVALUATED", holdout_access=False)
    seal(result / "metrics_registration.json", registration)
    seal(result / "model_metadata.json", evaluator.metadata)
    binding = identity(registration)
    # Original registered derangement, independent of reconstruction or quality.
    mismatch = replay.ReplayEngine._derangement()
    source_features, source_targets = [], []
    for index in range(SOURCES):
        cp = validate_source(read(out / "source_checkpoints" / f"{index:04d}.json"), reconstruction["binding"], index)
        with np.load(cp["float_reconstructions"]["path"], allow_pickle=False) as archive:
            target = pixels(archive["source_rgb"]).copy()
        if rgb_sha(target) != source_registration["source_float_sha256"][index]:
            raise RuntimeError("Original metric target changed")
        source_targets.append(target)
        with torch.no_grad():
            source_features.append(native.dino_features(dino, torch.from_numpy(target[None]).to("cuda:0"))[0].cpu().numpy())
    source_features = np.stack(source_features)
    started = time.time(); allrows, baselines, output_bindings = [], [], {}
    for index, record in enumerate(records):
        if stopped(): raise PauseRequested("Stop requested at a committed metric source boundary")
        gpu_available(); verify(own)
        if numeric_flags(torch) != flags: raise RuntimeError("Metric numerical flags changed")
        checkpoint = metric_dir / "source_checkpoints" / f"{index:04d}.json"
        if checkpoint.exists():
            done = runner.checkpoint_valid(read(checkpoint), binding, expected_ids(index))
        else:
            original_cp = validate_source(read(out / "source_checkpoints" / f"{index:04d}.json"), reconstruction["binding"], index)
            with np.load(original_cp["float_reconstructions"]["path"], allow_pickle=False) as archive:
                images = archive["images"].copy(); slots = archive["image_slots"].tolist()
            target = source_targets[index]; target_tensor = torch.from_numpy(target[None].copy())
            truth = int(record["class_index"])
            prepared = evaluator.prepare_reference(target_tensor)
            source_prediction = int(prepared["resnet50"][0])
            baseline = dict(source_id=record["image_id"], source_index=index,
                preprocessing_id=record["preprocessing_id"], reference_sha256=rgb_sha(target),
                true_class_index=truth, resnet50_source_prediction=source_prediction,
                resnet50_source_top1_label=int(source_prediction == truth))
            legacy_rows, observed_source_feature, embeddings = native.quality_metrics(target, list(images), lpips, dino, torch.device("cuda:0"))
            np.testing.assert_allclose(observed_source_feature, source_features[index], rtol=1e-6, atol=1e-6)
            with torch.no_grad():
                negatives = torch.from_numpy(source_features[mismatch[index]][None]).to("cuda:0").expand(len(images), -1)
                mismatched = torch.nn.functional.cosine_similarity(torch.from_numpy(embeddings).to("cuda:0"), negatives, dim=1).cpu().tolist()
            scorer = runner.PendingMetricScorer(evaluator, torch, target_tensor, prepared, truth,
                batch["chosen_batch_size"], batch["qualified_batch_sizes"])
            rows = []
            for row, slot in zip(original_cp["rows"], slots):
                value = dict(row, **legacy_rows[slot], dino_mismatched=mismatched[slot],
                    dino_specificity=legacy_rows[slot]["dino_cosine"] - mismatched[slot],
                    mismatch_source_id=records[mismatch[index]]["image_id"],
                    mismatch_source_index=mismatch[index], modelmanifest_sha256=manifest_sha)
                cache_key = replay.metric_cache_key(images[slot], target, evaluator.identity())
                value["metric_cache_key"] = cache_key
                rows.append(value); scorer.add(cache_key, images[slot], value)
            scorer.flush()
            confidence = []
            offset = 0
            for count in qualified_chunks(len(images), batch["chosen_batch_size"], batch["qualified_batch_sizes"]):
                with torch.inference_mode():
                    rgb = torch.from_numpy(images[offset:offset + count]).to("cuda:0")
                    logits = evaluator.models["resnet50"](metric_module.preprocess_resnet50(rgb))
                    probability, prediction = logits.float().softmax(-1).max(-1)
                    confidence.extend(zip(probability.cpu().tolist(), prediction.cpu().tolist()))
                offset += count
            for row, slot in zip(rows, slots):
                probability, prediction = confidence[slot]
                if prediction != row["resnet50_prediction"]:
                    raise RuntimeError("Independent confidence forward changed classifier prediction")
                row.update(resnet50_top1_probability=probability,
                    semantic_error=int(prediction != source_prediction),
                    confidently_wrong=int(probability >= .5 and prediction != source_prediction))
                for key in (*analysis.METRICS, "dino_mismatched", "resnet50_top1_probability"):
                    if not np.isfinite(row[key]): raise RuntimeError("Nonfinite registered metric: " + key)
            done = dict(binding=binding, source_index=index, rows=rows, baseline=baseline,
                reconstruction_source_checkpoint_sha256=sha(out / "source_checkpoints" / f"{index:04d}.json"),
                unique_images=len(images), metric_batch_sizes_used=scorer.batch_sizes_used,
                mismatch_source_index=mismatch[index], confidence_threshold=.5)
            done["payload_sha256"] = identity(done)
            runner.checkpoint_valid(done, binding, expected_ids(index)); seal(checkpoint, done)
        output_bindings[str(checkpoint)] = sha(checkpoint)
        allrows.extend(done["rows"]); baselines.append(done["baseline"])
        progress("RUNNING", sources_complete=index + 1, rows=len(allrows), elapsed_seconds=time.time() - started)
    if len(allrows) != ROWS or [r["replay_row_id"] for r in allrows] != [rid for i in range(SOURCES) for rid in expected_ids(i)]:
        raise RuntimeError("Incomplete or duplicate scored development population")
    # Reuse original source-level bootstrap and contrast implementations. Its
    # historical validate() hardcodes N512/N1024; the exact new inventory is
    # checked here without mutating that frozen implementation.
    groups = defaultdict(list)
    for row in allrows: groups[analysis.group(row)].append(row)
    if set(groups) != set(expected): raise RuntimeError("Metric method/SNR/N inventory differs")
    sources, identities = analysis.original_sources(baselines)
    for key, frames in groups.items():
        if {(r["source_id"], r["noise_seed"]) for r in frames} != {(s, seed) for s in sources for seed in SEEDS} or len(frames) != 300:
            raise RuntimeError("Metric group lacks full paired source/noise coverage")
        for row in frames:
            actual = identities[row["source_id"]]
            if (row["resnet50_source_prediction"] != actual["prediction"]
                    or row["resnet50_top1_label"] != int(row["resnet50_prediction"] == actual["truth"])
                    or row["resnet50_top1_source_prediction"] != int(row["resnet50_prediction"] == actual["prediction"])):
                raise RuntimeError("Classifier metric semantic identity differs")
    availability = set(analysis.NEW_METRICS)
    flags_by_group = {key: True for key in groups}
    bootstrap = analysis.Bootstrap()
    summary, per_source, means = analysis.summarize(groups, sources, availability, flags_by_group, bootstrap)
    paired = []  # No cross-method contrast in this honest Swin-only release.
    for row in paired:
        row["comparison_scope"] = "same_source_noise_seed_and_identical_measured_full_waveform"
    # Additional predeclared semantic-error diagnostics use the same source
    # averaging and the very same bootstrap draws as every existing metric.
    for name in ("semantic_error", "confidently_wrong"):
        for key, frames in groups.items():
            values, observations = analysis.source_values(frames, sources, name, availability)
            means[key, name] = values
            meta = dict(**analysis.context(key), group_id=analysis.group_id(key), is_main_conclusion=True)
            summary.append(dict(**meta, metric=name, direction="lower", **bootstrap.interval(values),
                n_frames=300, n_expected_frames=300, status="EVALUATED",
                aggregation="mean_noise_within_source_then_equal_source_mean"))
            per_source.extend(dict(**meta, source_id=sid, source_index=i, metric=name,
                value=float(values[i]), n_noise=len(observations[sid])) for i, sid in enumerate(sources))
    for name, rows in (("metrics_per_frame.csv", allrows), ("source_baseline.csv", baselines),
                       ("metrics_summary.csv", summary), ("metrics_per_source.csv", per_source)):
        write_csv(result / name, rows); output_bindings[str(result / name)] = sha(result / name)
    for path in (result / "metrics_registration.json", result / "model_metadata.json", metric_dir / "metric_batch_qualification.json"):
        output_bindings[str(path)] = sha(path)
    verify(own); verify(model_bindings)
    seal(out / "completion.json", dict(status="SWIN_EARLY_EVALUATION_COMPLETE", synthetic=False,
        sources=SOURCES, physical_frames=PHYSICAL_FRAMES, rows=ROWS, metric_groups=6, full_comparison_complete=False,
        reconstruction_completion_sha256=sha(reconstruction_path), sampler_step_limit=None,
        outputs=output_bindings, bindings={**own, **model_bindings, str(config_path): sha(config_path)},
        metric_evaluator_identity=evaluator.identity(), numerical_runtime=flags,
        selected_checkpoint_sha256=allrows[0]["selected_checkpoint_sha256"],
        paired_bootstrap_unit="source_mean_after_three_noise_repeats", selection_uses_development=False,
        holdout_access=False, elapsed_metric_seconds=time.time() - started))
    progress("COMPLETE", sources_complete=SOURCES, rows=ROWS)
