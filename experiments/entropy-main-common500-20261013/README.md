# Entropy-coded partial-scale common500 extension

This directory implements the authorized E0–E3 continuation of EMAIN-v1:
evaluate the frozen entropy-coded partial-scale method on the original 500
sources at N1024 and 4/10/19 dB, reuse compatible published references, report
actual source lengths and fallback behavior, and continue the existing
six-method timing task. It does not train, recalibrate, select checkpoints, or
start E4. The governing details are in [PROTOCOL.md](PROTOCOL.md).

## Current completion: E1/E2 closed; E3 still unresolved (2026-10-13)

E0 verified all 500 original pixel/token/source-order bindings, 36,000 reused
source-metric means, 72 unchanged historical intervals and 1,500 selected
whole-scale source/SNR length rows. Existing H800 calibration/new100 source
caches do not overlap common500. All 500 fresh entropy source passes used the
verified existing tokens, with zero new tokenizer calls.

All four E1 stages actually exited0: source, PHY, independent visual recovery
and scoring. The full 4,500-frame grid is closed. Independent read-only audit
checks the entire source/SNR/noise grid, original counters, physical-input
pins and 1,500 output hashes. There are 1,500 independent source decodes and VAR
reconstructions plus 3,000 exact received-input reuses, with no unresolved
reservations. All realized frames parsed and independently decoded; each
logical outcome is retained. See [scoring closure](../../results/entropy_main_common500_20261013/scoring_completion.json)
and [final receiver audit](../../results/entropy_main_common500_20261013/receiver_readonly_audit_v1/final_004500.json).

`analysis_owner_v2.py` has actually closed statistics and five figure exports
with exit0. There are 12 new marginal and 60 new source-paired intervals; all 72
original intervals remain unchanged. Actual length, fallback and prefix tables
are complete. The five groups have vector PDF, editable SVG and 600 dpi PNG.
All five PNGs were actually opened and visually reviewed with PASS; all 20
figure-output hashes were verified. The separate
[visual-review receipt](../../results/entropy_main_common500_20261013/visual_review_v1.json)
records this later check; the original export receipt remains unchanged. [Results, full-precision tables and manuscript fragment](../../results/entropy_main_common500_20261013/README.md).

The entropy partial-minus-whole PSNR gains are 0.616/0.705/0.337 dB at 4/10/19 dB;
LPIPS and DINOv2-L improve, but 19 dB agreement changes by -0.2 percentage points
(95% CI [-2.0, 1.6]). PSNR is lower than Swin200k/BPG at all three points and
latent JSCC at 4/10 dB. LPIPS is worse than latent JSCC at 4 dB; the 10 dB
LPIPS difference includes zero. No unfavorable or null result is omitted.

The first E3 shared-H800 v2 attempt actually exited1 during endpoint/model
construction due to a Python module-import path collision. It executed zero
timing cases and zero packet calls; there is no timing result. The original
failed scene remains intact. The independent v3 recovery entry also actually exited with code 1 during
construction: the runtime accessed `s.native.args.multiple_snr`, but the Swin
field is `s.native.multiple_snr`. It executed zero timing cases and zero packet
calls. Both failure scenes and actual waits are retained. 

Registration snapshot recorded at 2026-10-13T09:59:07+00:00 (not a live process-status assertion): v4 was retired during review before launch because its copied module globals would bypass the shared-resource metadata hook. The independent v5 recovery registration SHA is `44b14186a878621ea15e18da8cbf50e185927e6390826d019cff0865afc666f6`; four model-free CPU tests passed. At this snapshot v5 was in final review and had not launched. Original v2/v3 failure scenes remain intact. No timing cases, measured frames or digital packet calls had been consumed from the original 1,728/864/2,304 limits; numerical qualification and latency results were still pending.
 The prior owner must not be blindly restarted.
The existing `var-comm` continuation monitor is ACTIVE; the legacy 15-minute
`var-raw64-holdout` monitor remains PAUSED. E4 is disabled.

## Frozen registrations and inputs

- E1 uses [`registration_v4.json`](../../results/entropy_main_common500_20261013/registration_v4.json),
  SHA256 `2551cd1492e4f8354e8ee882a189b9fd498786f46868ab1b20a5638bf5791956`.
  Earlier registrations and zero-science admission failures are retained.
- E0 [audit](../../results/entropy_main_common500_20261013/e0_reuse/audit.json),
  [reuse inventory](../../results/entropy_main_common500_20261013/e0_reuse/reuse_inventory.csv),
  [ordered sources](../../results/entropy_main_common500_20261013/e0_reuse/source_order.json),
  [source records](../../results/entropy_main_common500_20261013/e0_reuse/source_records.json),
  and [frozen policies](../../results/entropy_main_common500_20261013/e0_reuse/frozen_configs.json)
  bind the reused input identities.
- [`source_means.csv`](../../results/entropy_main_common500_20261013/e0_reuse/source_means.csv)
  and [`summary.csv`](../../results/entropy_main_common500_20261013/e0_reuse/summary.csv)
  are normalized exports of existing references. Original summary rows and
  provenance are retained alongside them; old intervals are not recomputed.
- E2 has its own immutable
  [registration](../../results/entropy_main_common500_20261013/analysis_followthrough_v2/registration.json),
  SHA256 `733ad480eab5ff951efa201dbafbef9f40af295ec170a7b89263accca4f693dc`.
  New statistics are limited to 12 new marginal and 60 source-paired intervals.
- E3 reuses the fixed16 six-method budget under the separately registered
  [timing wrapper v2](../baseline-delivery-20261012/timing_revision_v2/README.md).
  Its current [registration](../../results/baseline_strength_20261011/timing/six_method_v2/registration.json)
  SHA256 is `6c2a531055cd611824130505e820dfdae314c747db31024431399b0561719c97`.

## Execution and delivery

The now completed `owner.py` executed `source.py` → `phy.py` → `visual.py` →
`score.py` serially, waiting for each actual exit. Source probabilities and all
24 independently terminated eligible endpoints are generated once per source.
The receiver uses actual paid received fields, independent CDF states, and the
full qualified 360-profile catalogue. Only an identical actual receiver input
or identical source/reconstruction pixel pair may reuse quality computation;
every noise frame retains its own receive outcome. The 4,500-row denominator
does not shrink when a neural call is reused.

`analysis_owner_v2.py` waited for complete scoring and ran `analyze.py` then
`plot.py`. It verifies the complete source/SNR/noise grid, calculates only the
registered new source-level intervals, and produces quality, paired,
within-representation mechanism, failure, length, fallback and prefix tables.
Figures export vector PDF, editable-text SVG and 600 dpi PNG. Actual visual
inspection remains a delivery step after the files exist. Read-only redraws
must use the completed statistical tables and must not rerun bootstrap.
The original wait-only analysis owner was safely terminated after verifying
zero scientific calls and no child process; its separate
`retirement_zero_science_metadata_v2.json` records this metadata-only
replacement. It does not invent an actual child-wait receipt or erase the
original registration and launch evidence.

Code and lightweight reports live in the sole repository root. The current
quality run's arrays, logs and ledgers live under
`/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/entropy_main_common500_20261013_v1`.
Old `var_comm_runtime_20261010` assets are reused in place. No new checkpoint or
PFS symlink is required.

Each stage has a fresh durable ledger, exact input/code bindings, finite caps,
failure evidence, and an actual child-wait receipt. A real failed attempt is
not automatically retried. Preserve its scene and distinguish completed,
missing and unresolved work before any explicitly reviewed recovery. Do not
hot-edit code bound to a live stage. A registration, launch or progress counter
is not successful stage closure.

## Timing scheduling amendment

After the initial exclusive-window plan, the user explicitly authorized shared
resources on 2026-10-13. The v2 timing wrapper records that amendment while
preserving the endpoint mathematics and the original 1,728-call budget. Its
results must be labeled **shared-environment latency**, not uncontended
latency. It waits until E1 quality GPU work has closed and a receipt-bound
coordination release declares no new quality science until the actual timing
exit. E2 CPU statistics may run concurrently; any such load is part of the
recorded shared environment. No other user's process is stopped and no global
GPU setting is changed. The original unused v1 registration remains intact.

## Checks and reproduction boundary

Repository checks are `python tools/verify_repository.py` and
`python tools/run_cpu_checks.py` from the sole repository root. The second is
an asset-free synthetic CPU suite, not measured image quality. Additional
analysis and timing-wrapper tests similarly do not qualify real endpoints.
The verifier also requires newly reviewed self-written source to be explicitly
staged; a pre-stage failure on those files is not a scientific failure.

Execution is once-only, and its completed owners must not be relaunched. There is no safe
copy-and-paste command here that should launch them a second time. For
reproduction on a separate recovery environment, first restore all pinned
model/data/runtime dependencies, inspect the immutable registrations and
completed-call receipts, and create a reviewed new execution identity. Reuse
completed work instead of rerunning it. E4, new bandwidths, Kodak entropy
experiments, extra SNRs, new methods and training require separate scope.
