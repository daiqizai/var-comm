# N2048 confirmation100 scoring and source-paired statistics

This is an implementation contract, not an execution result. Root alone launches
scientific work after T1 evidence is closed, T6 calibration selection is frozen,
and the actual T6 confirmation reconstruction closure exists.

## Fixed inputs and scope

N is 2048 only; SNRs are 4, 10 and 19 dB; source indices are the exact 100 entries
fixed in the metadata-only confirmation registration. Seeds are 9201, 9202 and
9203. The three method IDs are `N2048_RAW_WHOLE_VAR_COMPLETION`,
`N2048_RAW_PARTIAL_VAR_COMPLETION`, and one preselected
`N2048_EC_{STATIC|VAR}_WHOLE_VAR_COMPLETION`. There are exactly 2,700 rows.

The render closure is `T6_CONFIRMATION100_RENDER_COMPLETE`, includes observed
worker exits, seals every reconstruction NPZ and `per_frame.csv`, and binds the
source manifest, calibration freeze and single entropy-family selection. Each
frame includes `N`, `method_id`, `point_id`, source identity, SNR, noise seed,
actual status and `image_path/image_file_sha256/image_key/image_slot/image_sha256`.
All original row fields, including failures and resources, survive scoring.

The new source manifest has schema `T6_CONFIRMATION100_SOURCE_MANIFEST_V1`.
Its source checkpoints seal actual uint8 `pixels[3,256,256]` and tokenizer tokens.
Reference pixels must hash to `preprocessing_id`; the score reference is exactly
`pixels.astype(float32) / float32(255)`. Scoring performs no crop or resize itself:
it consumes the original preprocessing authenticated by the new-source owner.
It does not read old500 source features or old500 reconstruction scores.

The content-duplicate audit must pass for source IDs and preprocessing hashes.
It explicitly preserves available and unavailable prior-hash counts and the
scope `all_available_audited_source_content_hashes`. This is not a universal
never-seen-image claim.

## Frozen evaluator and numerical environment

The completed BPG metric template supplies the original frozen factory
`build_backend`, which constructs the same `SuiteBackend` and independently
frozen ConvNeXt classifier. Factory code, dependency and weight SHA bindings,
original numerical flags and canonical evaluator identity are verified.
No new evaluator or source-specific metric setting is introduced.

The reported metrics are `psnr_db`, `lpips_alex`, `dinov2_vitl14_cosine`, and
`convnext_top1_source_prediction`. PSNR retains float64 MSE. ConvNeXt is source
prediction agreement, not classification accuracy. All models run batch one in
the original UM Python environment, GPU0, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, CPU
affinity 4-9 and nice15, with the original shared visual lock. Reference preparation
is actually performed on each of the new100 sources. The legacy scorer's unused
mismatch input uses the preregistered cyclic permutation of these same100 sources;
no mismatch score is reported or used for selection.

Two fresh SQLite ledgers reserve calls before execution: at most100 reference
preparations and2,700 unique quality calls. Within this run, an exact same-source
RGB plus reference plus evaluator identity may reuse a metric result. Different
sources or different RGB hashes cannot share scores. A started owner is never
automatically retried; unresolved calls stay visible. There are no source encoding,
PHY, VAR rendering or bootstrap calls in `t6_score.py`.

The score child writes `worker_complete.json`, not the final completion.
`close` accepts the existing pinned `run_recorded_child.py` launch/exit receipts,
checks exact child PID, script/request argv, `actual_child_waited=true` and exit0,
then seals `T6_CONFIRMATION100_FOUR_METRICS_COMPLETE`.

## Statistical contract

Every source is averaged over the three registered noises before source-level
statistics. All failures are included, and there is no favorable-noise selection.
Comparisons, fixed before scoring, are:

1. Partial-scale raw minus complete-scale raw.
2. Entropy-coded complete-scale minus complete-scale raw.
3. Partial-scale raw minus entropy-coded complete-scale.

The original common500 statistics source SHA is
`e401a77ac216368dc968ce023853ff7cdb2e0a2c94edbd3dd33407c0ba90054a`.
Its `draws` and `interval` functions are imported in a private module instance;
only `SOURCE_COUNT` is configured to100. Function bodies are source-hashed and
unchanged. The source bootstrap uses10,000 draws with seed2026100701 and the
original2.5/97.5 percentile interval. The returned historical frame-count label
is explicitly changed from1,500 to300. The original file is never changed.
No original confidence interval is recalculated or overwritten.

All paired intervals use source differences, never subtraction of marginal
intervals. LPIPS retains method-minus-reference; negative means better. Agreement
canonical values are fractions and deltas are absolute fractions. Every summary
and paired row also includes `display_mean/display_ci_low/display_ci_high` with
scale100 for agreement: percent in summary, percentage points in paired output.
No relative percent improvement is substituted. Exact-zero vectors produce
exact-zero intervals, and identical source vectors reuse their within-run
intervals. Intervals are pointwise and unadjusted for multiple comparisons.

Outputs:36 summary rows,36 paired rows,3,600 source-mean rows, all-state frame
counts, explicit point/pair mapping, full-precision CSVs and `REPORT_T6.md`.
Only these two budgets have been tested; no interpolated bandwidth-saving
claim follows. Mixed-budget rows are rejected by the consumer.

## Execution shape (root only; placeholders must be real completed artifacts)

```text
python t6_score.py prepare --render-completion <actual T6 render completion> --metric-template <original completed BPG metric request> --original-scoring-module <original score_bpg_cached_holdout_r2.py> --out <fresh T6 metrics output> --deadline-unix <deadline>
python run_recorded_child.py --record-dir <fresh score parent record> --cwd <repo> -- <original metric python> <absolute t6_score.py> run --request <absolute score request>
python t6_score.py close --request <absolute score request> --wait-record <actual score parent record>
python t6_statistics.py --score-completion <closed T6 score completion> --statistics-module <unchanged original common500 statistics source> --out <fresh T6 statistics output>
```

The parent launch must inherit the registered GPU, BLAS, affinity and priority
environment; the scorer checks these and fails closed if they differ. Merely
having a worker output file is not proof of an observed successful process exit.

`t6_score_selfcheck.py` exercises complete/duplicate/missing grids, wrong-budget
rejection, failed-frame inclusion, nonfinite failure handling, noise-mean order,
LPIPS sign, percentage-point units, exact-zero paired differences and actual-wait
admission using synthetic engineering fixtures. It performs no model, PHY or
scientific bootstrap calls.
