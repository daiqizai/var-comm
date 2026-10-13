# Final frozen Kodak baseline statistics

`postprocess.py` consumes completed scoring tables. It never loads a model or executes a transmitter, channel, receiver, reconstruction, training, calibration, or model selection. The root controller starts actual work only after checking the actual completed inputs.

The fixed scope is all24 original256x256 Kodak center crops, N1024, SNR4/10/19, and each method's original noise labels2001/2002/2003 and generator. P100K andSWIN200K contribute432 new logical frames, represented by1728 long-form metric rows. Original NeST raw and adaptive BPG contribute432 verified reused frames. Every failure output is retained.

## Input contract

The new `rows.csv` uses the historical Kodak metric schema:

```text
method,source_index,source_id,snr_db,noise_seed,status,N,population,reference_sha256,reconstruction_sha256,reconstruction_archive_sha256,preprocessing_id,original_metric_backend_identity,metric_backend_identity,true_class_label_available,source_prediction,reconstruction_prediction,frame_level_noise_pairing_claimed,metric,value
```

New method identities are exactly `P100K` and `SWIN200K`. Four metric identities are `psnr_db`, `lpips_alex`, `dinov2_vitl14_cosine`, and `convnext_top1_source_prediction`. The source order is the original `kodak/kodim01` through `kodak/kodim24`. Each source/SNR/method must have all three prescribed noises and all four metrics, including failures. Reference pixel SHA, source prediction, preprocessing identity, and metric backend identity must match the original Kodak score rows.

The new completion status is `KODAK24_FINAL_BASELINES_FOUR_METRICS_COMPLETE_V1`; it supplies `source_count:24`, `frame_count:432`, `metric_rows:1728`, `N:1024`, `source_ids`, `methods`, `metrics`, an explicit `metric_math_identity_proof:{path,sha256}`, and an `outputs` mapping binding the actual metric CSV SHA. A successful process launch does not satisfy this input contract.

The H800 scoring process has its own `metric_backend_identity`; it explicitly retains `original_metric_backend_identity` for the historical mathematical definition. Its SHA-bound `metric_math_identity_proof.json` must have status `PASS_ORIGINAL_KODAK_MATH_AND_WEIGHT_BINDINGS`, all seven Kodak-specific AST expression checks, unchanged DINO source/preprocessing, and the original frozen metric weight/source bindings. The original identity is not claimed as a current host runtime identity, and no cross-host bitwise equality is claimed. Original/reference pixel hashes and source predictions must match exactly across the old and new score rows. The merged long-form CSV retains both identity columns.

The independent anchor reuse audit is a root-controller receipt with status `PASS_FROZEN_KODAK_ANCHOR_REUSE`, `approved_methods:["VAR_UNCONDITIONAL","BPG_ADAPTIVE"]`, `source_count:24`, `N:1024`, `snr_db:[4,10,19]`, `noise_seeds:[2001,2002,2003]`, and true values for `source_pixels_match`, `model_identity_match`, `configuration_identity_match`, and `metric_identity_match`. The actual evidence and per-file bindings belong in that receipt. It records validation of the original data, models, frozen policies, and metric identities rather than inferring identity from filenames.

## Exact historical statistics reuse

The script imports `interval()` unchanged from `experiments/generalization_kodak_20261009/analyze.py`, checks its original constants, and constructs the same shared10000x24 draw array with seed2026100901. The resulting draw SHA must equal the historical analysis completion. Each source first averages its own three noise scores. The function is called only24 times for the two new single-method summaries and24 times for NeST raw minus each new baseline on matched sources. No original summary or paired interval is recomputed, and no two confidence intervals are subtracted.

Historical raw/BPG summary lines, historical per-source means, and12 historical raw-minus-BPG paired lines are copied byte-for-byte into the combined exports. Summary/source rows are checked against the original metric values. Separate new-only tables make the added scientific scope explicit. Input CSVs and historical analysis files are completion-SHA verified and checked unchanged after processing.

## Outputs

- `summary.csv`:48 complete four-method single-method rows; `summary_new.csv`:24 new rows.
- `source_means.csv`:1152 method/source/SNR/metric rows; `source_means_new.csv`:576 new rows.
- `paired.csv`:36 NeST-minus-reference rows, including12 historical BPG contrasts; `paired_new.csv`:24 new P/Swin contrasts.
- `rows_all_four_methods.csv`:3456 metric cells for864 logical method frames, including failures.
- `status_counts.csv`, `plot_data.csv`, `REPORT.md`, `captions.tex`, and a short `manuscript_results.tex` paragraph.
- `kodak24_N1024_final_four_methods.pdf/.svg/.png`:one7-inch2x2 four-metric figure, editable SVG text, vector PDF curves, and600dpi PNG.
- `completion.json`:exact statistical scope, draw/function/script/input/output bindings, and pending actual visual-review status.
- `statistics_reserved.json`:durable one-attempt reservation before constructing any bootstrap indices; unknown statistics failures are not retried automatically.
- `statistics_completion.json`:SHA-bound completed source-statistics checkpoint, written before any Matplotlib rendering.

LPIPS signs are unchanged. ConvNeXt source-prediction agreement is displayed as percent; differences are percentage points. SNR points use numerical spacing. All curves, failure scores, negative effects, and confidence intervals are retained. The final Swin200k training range includes19dB; the old Swin80k result remains historical and is not relabeled.

```sh
python experiments/generalization-kodak-final-20261013/postprocess.py \
  --rows NEW_METRICS/rows.csv --completion NEW_METRICS/completion.json \
  --old-root results/generalization_kodak_20261009 \
  --reuse-audit NEW_RESULTS/anchor_reuse_audit.json --out NEW_RESULTS/analysis_v1
```

Use a fresh output directory once. Do not re-run completed bootstrap or inference to recreate images. If layout fails after `statistics_completion.json` is complete, the controller first investigates the original failure and can explicitly recover the CPU layout alone:

```sh
python experiments/generalization-kodak-final-20261013/postprocess.py \
  --render-only --out NEW_RESULTS/analysis_v1
```

This recovery accepts no new scoring inputs, verifies the checkpoint's code/input/statistical-table SHA bindings, and makes zero bootstrap calls. It refuses incomplete statistics or a delivery that already has `completion.json`. Statistical files must remain unchanged during rendering. Open the exported PNG and record an actual visual review separately before publication. This experiment supports a limited24-crop cross-collection comparison; collection identity does not certify non-overlap with every pretrained foundation-model corpus.
