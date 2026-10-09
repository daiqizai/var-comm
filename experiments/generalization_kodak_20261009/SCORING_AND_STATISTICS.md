# Frozen Kodak24 scoring and statistics

This directory prepares a new Kodak24 experiment. Source preparation or successful static checks do not mean inference, scoring, statistics, or image inspection has completed. Actual completion files record each executed stage.

## Fixed population and scientific scope

- Use every `kodim01` through `kodim24`, in numeric order. Each method receives the same deterministic 256 by 256 center crop, without resizing. This is a crop experiment, not transmission of Kodak's full-resolution photographs.
- N = 1024; SNR = 4, 10, 19 dB; noise labels = 2001, 2002, 2003. Keep each method's original noise generator. Pair statistics by source after averaging each method's three noises; the equal seed labels do not imply identical received observations.
- Freeze the previously selected source/channel policies and all weights. No Kodak policy selection, training, or calibration. Adaptive BPG still runs its already frozen source encoder search; this is source encoding, not new MCS selection.
- VAR uses the unconditional null embedding (index 1000). No source ImageNet class label enters reconstruction or evaluation. The neural adapter audits the embedding input at every non-gray reconstruction.
- Keep all 864 frames, including prescribed gray outputs and other failures. Swin at 19 dB remains outside its training/calibration range. Do not infer training-data non-overlap from Kodak's collection identity.
- Neural reconstruction uses the previously qualified batch-one endpoint. No bitwise equivalence to the older batch-three common500 run is claimed.
- Independent upper bound: 1080 PHY calls (VAR 432, BPG 432, Swin 216, latent continuous JSCC 0). Only the root controller starts remote work.

## Exact score interface

`score.py --request SCORE_REQUEST.json` consumes:

```json
{
  "schema": "KODAK24_FROZEN_FOUR_METRIC_REQUEST_V1",
  "N": 1024,
  "snr_db": [4, 10, 19],
  "metrics": ["psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "convnext_top1_source_prediction"],
  "records": ["the 24 full records copied from the frozen dataset manifest"],
  "method_inputs": [
    {
      "method": "VAR_UNCONDITIONAL",
      "completion": {"path": "ACTUAL_METHOD/completion.json", "sha256": "ACTUAL_HASH"},
      "frames": {"path": "ACTUAL_METHOD/frames.json", "sha256": "ACTUAL_HASH"},
      "noise_seeds": [2001, 2002, 2003]
    }
  ],
  "backend_request": {"path": "EXISTING_EXECUTED_NATIVE_METRIC_REQUEST", "sha256": "ACTUAL_HASH"},
  "backend_module": {"path": "EXISTING_score_bpg_cached_holdout_r2.py", "sha256": "d8e9f85c83f63a288cf0500e53857bc6e81c205cec93897e4d46f9f177c79d8e"},
  "out": "NEW_INDEPENDENT_SCORE_DIRECTORY",
  "new_metric_call_cap": 864,
  "reference_preparation_cap": 24,
  "max_seconds": 3600,
  "stop_files": []
}
```

The request template abbreviates records and shows one of the four method entries. Supply all four canonical methods: `VAR_UNCONDITIONAL`, `P1024`, `BPG_ADAPTIVE`, `SWIN80K`. Use their actual completion-bound frames tables, never prepared requests. The executed native metric request is mirrored at `.research/main_raw64_20261008_paper_supplement/bpg_metrics_v1/materials_r2/request.json`; the controller resolves its established remote path.

Each method supplies exactly 216 frame records with method, source index/ID, SNR, noise seed, actual status, and `reconstruction: {path, sha256, key: "rgb"}`. The NPZ contains unchanged float32 CHW RGB in [0,1]. The scorer reads references from the manifest's RGB PNG and checks both PNG and contiguous CHW uint8 hashes. It never resizes or re-encodes reconstructions for metrics.

Run scoring on the original Linux evaluation environment, with CUDA_VISIBLE_DEVICES=0 and CUBLAS_WORKSPACE_CONFIG=:4096:8. The original backend checks all original model/source bindings and numerical flags. Its shared visual GPU lock prevents overlap with another owner. Per-source checkpoints permit reuse; a reserved but unfinished source is retained for explicit recovery rather than silently repeated.

## Four metrics without fabricated labels

The scorer calls the existing backend's components directly, never its label-requiring `score` methods:

| Output field | Existing operation |
|---|---|
| `psnr_db` | Per-image float64 RGB MSE, followed by -10 log10; average PSNR values afterward |
| `lpips_alex` | `backend.lpips(reconstruction*2-1, source*2-1)` at batch one |
| `dinov2_vitl14_cosine` | Existing `dinov2_vitl14_features` and cosine similarity |
| `convnext_top1_source_prediction` | Equality of `backend.classifier.predict(reconstruction)` and `predict(source)` |

The last metric is source-prediction agreement, not ground-truth accuracy. Both predicted class indices remain in the frame table so the equality can be checked. DINO-L is not the legacy DINO-S field. The old backend identity is preserved as provenance while every score row labels its actual population Kodak24. Exact identical reconstructed pixels within one source reuse their score. Each original source feature/prediction is computed once. A truly zero MSE is retained as infinity; the statistics script refuses to silently cap or remove an unbounded PSNR.

Scoring emits `rows.csv` (3456 long rows), 24 source checkpoints, and a hash-bound completion. It performs no reconstruction, PHY, channel sampling, or bootstrap.

## New-data statistics and figures

```sh
python experiments/generalization_kodak_20261009/analyze.py --rows SCORE/rows.csv --completion SCORE/completion.json --out NEW_ANALYSIS_DIRECTORY
```

The script first averages the three noises within each of 24 sources. It creates one shared 10,000 by 24 source-resampling array, with seed 2026100901, and percentile 95% intervals. It emits 48 single-method summaries, 36 source-paired VAR-minus-reference contrasts, 1152 source means, and failure-status counts. It does not resample old results or subtract their confidence intervals. LPIPS differences retain their sign. Agreement means are displayed as percent; differences are percentage points.

Two new 2 by 2 figures are exported as vector PDF, editable-text SVG, and 600-dpi PNG. SNR uses numerical spacing. The main Swin 19 dB marker is hollow and the 10-to-19 dB segment is dashed because 13 dB is not measured here. Bounds include all intervals and paired zero lines. Captions, actual plotting data, and a report accompany the figures. The recorded PNG review remains pending until a person or image-viewing tool actually opens the exports.

These are pointwise intervals over 24 source images, not multiplicity-adjusted claims of universal superiority or proof of unseen pretrained data. All selected crops and noises are fixed before quality evaluation.

## Local preparation checks

Both Python files parse and expose their CLI. A pure NumPy known-effect check verifies all 36 source-paired intervals and exact 48/36/1152 output counts. Local matplotlib is unavailable, so no synthetic layout image is presented as an experiment result; actual rendering and PNG inspection follow the completed remote scoring stage.
