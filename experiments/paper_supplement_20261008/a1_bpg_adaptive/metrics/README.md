# Adaptive BPG: cached reconstruction scoring and new paired statistics

This separate consumer starts only after the actual adaptive-BPG holdout and its
policy freeze have normal child-process completion receipts. Local code or
prepared requests are not evidence of metric completion.

It imports the unchanged `score_bpg_cached_holdout_r2.py` backend construction,
original SuiteBackend/ConvNeXt weights and flags, and the original source
bootstrap functions. Evaluation is FP32, batch1, GPU0, CPU4–9, nice15. No codec,
channel, PHY decoder, main-system inference, or training is run here. Do not run
this consumer during the dedicated timing window.

The four metrics are PSNR, LPIPS-Alex, `dinov2_vitl14_cosine`, and ConvNeXt source
prediction agreement. Agreement is not label accuracy. Stored raw agreement
values are fractions/bools; presentation may multiply by100. Difference values
then have units of percentage points; LPIPS is never sign-flipped.

## Reuse and source features

A targeted remote inspection found no persistent prepared-reference feature
cache in the original unified500 or native256 BPG outputs. Therefore the first
new run prepares the500 original references once and reports these preparations
separately. The original `backend.prepare` result is saved losslessly as NPZ
arrays plus a JSON tree, with no executable pickle. It is keyed by source ID,
original source NPZ SHA, RGB SHA and exact evaluator identity. Completed entries
are reused after hash checks; interrupted unreceipted preparation is not silently
repeated.

The unchanged original scoring call remains intact, including its internal
checks and any internal legacy reference computation; the cache does not claim
to remove every reference operation inside that original function.

Identical same-source reconstruction hashes already scored for native256 BPG
are reused from its completed quality ledger. This includes identical fixed-gray
outputs. Otherwise each unique source/image/evaluator combination is scored
once and retains every original noise row. At most9000 new reconstruction
quality calls and500 separate initial reference preparations are permitted.

## New statistics only

All500 sources, six SNRs, and all three fixed noise records are required:9000 rows.
The new24 single-method rows and96 paired rows use10000 source bootstrap draws,
seed2026100701, after averaging the three noise results within each source.

Predeclared pairs are adaptive BPG minus:

- The published partial-scale digital transmission with VAR completion.
- The published continuous latent JSCC baseline.
- The published SwinJSCC-80k adaptation.
- The separately retained native256 BPG baseline.

Original per-source means and confidence intervals are loaded unchanged from
the completed source-statistics files. Only the new baseline and new comparisons
are bootstrapped. Pairing is by source; different waveforms are not claimed to be
the same observation. `comparison_summary.csv` copies old means/intervals and
appends the new method without overwriting any scientific input.

## Prepare after actual holdout completion

Download only the actual new control files needed by this preparation step:
`holdout/completion.json`, `freeze/completion.json`, and `freeze/worker_0.json`.
The existing original completion files and template metric request are already
available locally. The large received-image/source arrays remain on the server.

```text
python experiments/paper_supplement_20261008/a1_bpg_adaptive/metrics/prepare_metrics.py --workspace C:/Users/11946/Documents/ChatGPT/comm --adaptive-request results/paper_supplement_20261008/a1_bpg_adaptive/a1b_materials_v1/request.json --actual ACTUAL_A1B_OUTPUT_MIRROR --out results/paper_supplement_20261008/a1_bpg_adaptive/metrics_materials_v1 --deadline-unix EXECUTION_DEADLINE
```

The resulting `execution.json` supplies the exact two-file upload, argument list,
numeric environment, affinity, and minimal download list. Its default processing
window is7200 seconds. Preparation refuses absent or non-normal holdout control
files. It does not start a process or infer completion from a request file.

Actual output directory:
`/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1/`

Deliverables include `summary.csv`, `paired.csv`, `source_means.csv`,
`failure_breakdown.csv`, `comparison_summary.csv`, `rows.json`, individual
source checkpoints, the unique-quality ledger, source feature cache and final
completion receipt. Output status remains incomplete until the real process
has exited successfully and that receipt is verified.
