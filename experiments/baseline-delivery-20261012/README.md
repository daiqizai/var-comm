# Baseline delivery after frozen training

This independent directory does not change the running scoring-chain code.
The delivered statistics/plot scripts add no training, inference, simulation or
checkpoint selection. The independent timing owner executes only the already
authorized finite timing budget. Original science files are read-only inputs.

`plot_training.py` exports the actual full training and calibration histories.
Training losses use explicitly labeled non-overlapping 500-update means; all
updates are included. Calibration curves retain every completed point. P's
composite objective and Swin's MSE are not compared numerically across branches.
PDF is vector, SVG text remains editable, and PNG is 600 dpi. The three outputs
were opened for visual inspection; the per-SNR legend was moved above all data.

From the repository root, with the frozen scientific Python environment:

```sh
python experiments/baseline-delivery-20261012/plot_training.py \
  --results results/baseline_strength_20261011 \
  --p-history "$BASELINE_TMP/P/training_history.csv" \
  --swin-history "$BASELINE_TMP/Swin/training_history.csv" \
  --output results/baseline_strength_20261011/training_figures_v1
```

`BASELINE_TMP` is the existing personal temporary-storage directory recorded in
the result provenance. The raw training logs remain there and in the verified
local recovery increment. Only lightweight plot data are published.

## Once-only statistics

`statistics.py prepare` validates the restored, SHA-pinned original common500
source means, original intervals, source order and original bootstrap module;
it also validates the published new100 source means. It draws no samples.
Preparation is already complete; do not rerun it over its registration.

`after_scoring.py` is a finite, separately locked CPU successor. It waits for the
existing scoring chain's normal closure (maximum 24 hours), checks actual child
exit-zero receipts and frozen source/model/code identities, then invokes
`statistics.py run` once. Any unknown failure stops without automatic retry.
The admission rejects missing/duplicate frames, changed source pixels, wrong
model steps, wrong noises and reused frames for new models. Eight CPU tests
exercise rejection and retain valid failure/zero rows without resampling.

The original `draws` and `interval` function bodies are preserved, with 10,000
replicates and seed 2026100701. Only source count is adapted to 100 for the
existing-new100 cohort, exactly as in its published implementation. Each source
is first averaged over three original noises. Common500 and new100 remain
separate. Old intervals are copied unchanged; the new calls are limited to
48+24 single-method and 96+24 paired rows. Exact-zero contrasts need no interval
call. LPIPS is not sign-flipped; agreement displays use percent/percentage points.

Required comparisons are new P minus old P, new Swin minus old Swin, raw partial
minus each new baseline on common500, and entropy-coded partial minus each new
baseline on new100. Cross-host bitwise metric equivalence is not claimed; the
frozen mathematical definitions, weights, numerical settings and adapter
provenance are retained. No test scores feed back to checkpoint selection.

After statistics close, the successor exports milestone, bottleneck and final
comparison figures through `plot_evaluation.py`. Generation is not visual QA:
the heartbeat must open the PNGs before publishing their final delivery.

## Formal results delivered

The existing chain and CPU successor have now closed normally, with actual
exit-zero receipts: 18,000 common500 and 1,800 existing-new100 method frames,
72 new single-method intervals and 120 direct source-paired contrasts.
All four final PNG groups were opened and visually reviewed. Vector PDF,
editable SVG, 600 dpi PNG, exact plot rows and captions are under
`results/baseline_strength_20261011/final_comparison_figures_v1`.
The independently audited report and English settings are under
`results/baseline_strength_20261011/formal_delivery_v1`.

Re-export figures without inference, simulation or bootstrap:

```sh
python experiments/baseline-delivery-20261012/plot_evaluation.py final
```

The once-only `package_final.py` validates the closed statistical and figure
receipts and creates the publication report. It has already run; do not rerun
over the existing report. `statistics.py run` must never be repeated simply to
regenerate figures. Published completion records retain the original stage
history; use the independent statistical completion to establish its closure.

## Timing remains pending

`timing_owner.py`, `timing_runtime.py` and `timing_endpoints.py` now provide the
independent six-method entry, guarded by `timing_contract.py`. Ten no-model CPU
tests pass, including actual RX fields, CRC rejection, raw KEEP, fresh fallback
attempts, replay protection and an exact native-library addendum. Registration
and a real resource-gated invocation completed without model/packet calls.
Actual fixed warmups, numerical qualification and timing are still pending.
The current four digital arms are raw whole/partial and conditional-entropy
whole/partial; the old T4 static-entropy arm cannot substitute for the last arm.
Fixed development indices, SNRs 4/10/19, batch1, three warmups and three measured
repeats are retained. The two selected continuous baselines share these settings.
Actual timed components must start from fresh image encoding and real received
fields, include fresh entropy fallback attempts, and exclude disk-journal I/O.

Pure Sionna 2.2.0 and its missing pure-Python import dependency were copied
byte-for-byte from existing frozen assets into personal temporary storage.
Importing the exact original LDPC sources in the visual Python environment now
succeeds without CUDA initialization or packet calls. This is not numerical or
full-endpoint qualification. Original environments remain unchanged. The fixed
16 source archives and their source metadata hashes match; historical cached
codec receipts are not present in that seed and cannot stand in for fresh encoding.

Timing still needs endpoint qualification and the original exclusive resource
window. No competing process is stopped, no GPU global setting is changed, and
old 4090D measurements must not be presented as a fair H800 comparison.

Full final recovery states (P100000 and Swin200000), raw training logs and
metadata were verified at
`D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/final_training_20261012_v1`.
This is an increment depending on the existing original input/model/runtime
recovery archives, not a standalone complete-project package.

### Registered once-only timing entry

Run from the remote repository, after checking actual locks/receipts. The owner
returns WAIT without model imports if any GPU has another resident compute job.
An admitted launch writes a unique launch record and obtains actual child exit.
Never repeat `prepare` over the registration or `run` after an actual attempt.

```sh
python experiments/baseline-delivery-20261012/timing_owner.py run
```

The owner chooses the frozen visual Python interpreter for its child. A successful
resource check is not numerical qualification. Each source/SNR/method uses three
fresh fixed warmups and three fresh measurements, verifying repeat identity and
tokenizer identity against original source metadata, without extra test frames.
No evaluation network runs in the window. TX/RX and software end-to-end intervals
synchronize CUDA; noise generation is separately recorded/subtracted. File
journals, hash audits, STOP-file checks and resource queries are outside the
measurement window; packet charges and model call accounting remain in memory.
Original raw433/whole144/partial360 receive catalogues are retained. P/Swin use
original timing noise2001; raw uses original development6201 and development-slot
counter; entropy uses original T1 full-frame6201 and its original counter.

The already-built token-trellis library has a source/build/library SHA binding
and an explicit private in-memory native inventory addendum. This is not a claim
of old-host native byte identity. Original inventories and environment files are
unchanged. The pre-launch adapter correction and prior registration are retained
as revision evidence; no scientific case existed when the correction was made.
If the real runtime rejects a new dependency or numerical output, retain its
failure and consumed case records, diagnose it, and do not automatically retry.
