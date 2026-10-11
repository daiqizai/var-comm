# Baseline delivery after frozen training

This independent directory does not change the running scoring-chain code.
It adds no training, model inference, channel simulation, checkpoint selection,
or timing budget. Original science files are read-only inputs.

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

## Status and remaining work

This publication delivers the training figures and verified final training
recovery increment, and installs the finite statistics/figure successor. It
does not claim completed holdout results, statistical intervals or fair timing.
Live state is in `results/baseline_strength_20261011/paired_supplement_v1`.
Timing still requires the registered same-device exclusive resource window;
old 4090D measurements must not be presented as a fair H800 comparison.

Full final recovery states (P100000 and Swin200000), raw training logs and
metadata were verified at
`D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/final_training_20261012_v1`.
This is an increment depending on the existing original input/model/runtime
recovery archives, not a standalone complete-project package.
