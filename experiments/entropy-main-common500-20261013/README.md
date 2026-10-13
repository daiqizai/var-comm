# Entropy-coded partial-scale common500 extension

This directory implements the authorized E0–E3 continuation of EMAIN-v1:
evaluate the frozen entropy-coded partial-scale method on the original 500
sources at N1024 and 4/10/19 dB, reuse compatible published references, report
actual source lengths and fallback behavior, and continue the existing
six-method timing task. It does not train, recalibrate, select checkpoints, or
start E4. The governing details are in [PROTOCOL.md](PROTOCOL.md).

## Publication snapshot: 2026-10-13, source and PHY closed

E0 is complete. All 500 source archives, token hashes and source-order entries
were checked. Six historical reference methods supply 36,000 source-metric
means and 72 unchanged published intervals; raw whole-scale is included for
the mechanism table in addition to the five direct quality references. The
original T1 record supplies 1,500 selected source/SNR length rows. Neither
existing H800 calibration nor new100 source cache overlaps common500, so this
extension needs 500 fresh entropy source passes using the already verified
tokens. E0 performed no model, channel, reconstruction or bootstrap call.

E1 is actually running under the once-only owner. Its source stage exited0
after487.186 seconds, with500 completed source passes and5,000 completed
prior-scale calls, zero unresolved reservations and zero tokenizer calls.
The source completion SHA256 is
`d3907081f9583fc88e5b594726d84a1922409bacf3a870b8e124d2fb01ffd4e7`;
the actual waits are retained in the
[initial execution snapshot](../../results/entropy_main_common500_20261013/initial_execution_snapshot.json).
The CPU PHY stage also exited0. Its
[actual grid audit](../../results/entropy_main_common500_20261013/actual_phy_grid_audit.json)
confirms all4,500 unique source/SNR/noise rows and1,500 source/SNR length rows;
each SNR has1,500 `PAYLOAD_PARSED` outcomes. Source and PHY completion/actual
waits are copied under
[`stage_receipts`](../../results/entropy_main_common500_20261013/stage_receipts).
The owner is now in the independent visual reconstruction stage. Parsed link
payloads are not yet completed entropy source decoding or image quality;
reconstruction, scoring, statistics and measured timing remain pending.

The independent E2/analysis owner is waiting for all 4,500 scored logical rows
and actual exit-zero closure. The existing E3 successor is waiting for the same
closure before obtaining the timing release and resource admission. It has
performed no timing case. Live state is in
[`status.json`](../../results/entropy_main_common500_20261013/status.json),
[`analysis_followthrough_v2/status.json`](../../results/entropy_main_common500_20261013/analysis_followthrough_v2/status.json),
and the [timing successor state](../../results/baseline_strength_20261011/timing/six_method_v2/successor_status.json).
PIDs are diagnostic hints only; verify command lines and actual receipts before
any action. Do not launch duplicate owners.

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

The already running `owner.py` executes `source.py` → `phy.py` → `visual.py` →
`score.py` serially, waiting for each actual exit. Source probabilities and all
24 independently terminated eligible endpoints are generated once per source.
The receiver uses actual paid received fields, independent CDF states, and the
full qualified 360-profile catalogue. Only an identical actual receiver input
or identical source/reconstruction pixel pair may reuse quality computation;
every noise frame retains its own receive outcome. The 4,500-row denominator
does not shrink when a neural call is reused.

`analysis_owner_v2.py` waits for complete scoring and runs `analyze.py` then
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

Execution is once-only, and its active owners already exist. There is no safe
copy-and-paste command here that should launch them a second time. For
reproduction on a separate recovery environment, first restore all pinned
model/data/runtime dependencies, inspect the immutable registrations and
completed-call receipts, and create a reviewed new execution identity. Reuse
completed work instead of rerunning it. E4, new bandwidths, Kodak entropy
experiments, extra SNRs, new methods and training require separate scope.
