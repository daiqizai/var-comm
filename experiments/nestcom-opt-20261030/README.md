# NeST-Com five-route optimization, 2026-10-30

This is the new user-authorized A–E workstream. It does not resume old stopped
selective-block, shared-codeword or learned-residual experiments. The original
`protocol_defaults.json` is preserved as the user's design document; actual
registrations, outcomes and accounting are separate.

Read `results/nestcom_opt_20261030_v1/delivery/README_zh.txt`, `REPORT_zh.txt` and
`stage_completion.json` for actual completion and blocked items. Do not infer
scientific completion from the presence of a script. The original work order
required a review-only package. The subsequent user request explicitly authorizes
normal commit/push; see results/nestcom_opt_20261030_v1/publication_20261030.
The current interpretation is in delivery/ANALYSIS_zh.txt; prior no-push receipts
remain historical snapshots.

## Implementation and execution boundaries

- `opt_common.py`, `control_once.py`, `launch_registered.py`: exclusive immutable
  registration, real process waits, identity checks, finite per-route occupancy
  budgets and at most two owned GPU workers.
- `p0_replay.py`, `qualify_variants.py`: real frozen-model qualification;
  `test_c_contracts.py` contains additional CPU engineering checks.
- `receiver_variants.py`, `receiver_worker.py`: A decoder routing and B fixed
  low-temperature generation. The original arithmetic-decoding probabilities are
  unchanged. The receiver cannot access source RGB, unsent targets or metrics.
- `c_cache.py`, `c_model.py`, `c_train_pair_v2.py`, `c_receiver_v2.py`:
  registered training subset, matched content/no-content controls, zero-residual
  contracts and frozen image evaluation. `c_token_diagnostics.py` reads labels
  only in a separate CPU process after the receiver has exited.
- `grid_codec.py`, `grid_catalogue.py`, `grid_source.py`: nested endpoint grids,
  preserved maximum endpoint/MCS, independent arithmetic termination and a legal
  shared 12-bit catalogue. No truncation of longer streams.
- `tx_score.py`, `tx_selected_plan.py`: E's three fixed feasible endpoints and
  source-side MSE + 0.1 LPIPS objective; future channel outcomes are excluded.
- `extended_frame_worker_v3.py`, `evaluation_receiver_plan_v2.py`,
  `original_receiver.py`: actual changed-wire transmissions and independently
  parsed received prefixes; exact reuse requires full wire/noise/model identity.
- `metrics.py`, `launch_metric_stage_v2.py`: native-array PSNR, LPIPS-Alex,
  DreamSim and CLIP-I, with strict device/precision/preprocessing feature caches.
- `route_statistics_v2.py`, `combined_statistics.py`, `c_screen_decision.py`:
  all-candidate source-paired FIT/CONFIRM reporting with fixed FIT denominators.
- `historical_regression_inputs_v5.py`, `final_statistics.py`: preserved historical
  common500 arrays/scalars and method-native noise, direct source-paired regression.
  Historical v1–v4 read-only schema failures remain auditable.
- Final execution is closed. `final_successors.py` completed RX, then failed
  before any metric process was launched because of resource admission. Its
  original failure remains unchanged. `final_unlaunched_metrics_v2.py` uniquely
  completed those previously unlaunched metrics and statistics. Neither entry
  should be rerun; `EXECUTION_ENTRY_INDEX.txt` and actual waits record the lineage.
- `benchmark_receiver.py`, `benchmark_transmitter_v2.py`, `benchmark_phy.py`:
  actual fixed-source deployment timing. The original transmitter's pre-trial
  hash-convention failure is retained. v2 changes only that assertion; all source
  data and algorithms remain fixed. Timing repetitions are explicitly separate
  from scientific image-quality calls.
- `summarize_timing.py`, `resource_delivery.py`, `build_delivery.py`,
  `draw_results.py`, `validate_delivery.py`: post-exit summaries, native-data plots
  and delivery validation. These must not rerun successful scientific work.

## Reproducibility and storage

The only Git worktree is
`/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM`.
Large arrays, checkpoints, model assets and logs remain on PFS under the pinned
paths in registrations. This directory contains code, not those large artifacts.
The results namespace contains exact command arguments, code/asset SHA-256,
source order, numerical routes, actual wait receipts, reuse counts and failures.
The delivery includes a checkpoint recovery manifest without restarting failed
or conditional training branches.

Versioned old scripts are preserved for lineage. They are not alternative
launchers to try. Use completed receipts and the registered latest entry for a
stage; a fresh reproduction requires a fresh namespace and explicit budget,
never deleting output guards or changing a bound registration. Orchestration
history is for audit and must not be batch-executed.

FIT and CONFIRM are development populations. common500 is previously examined
regression. If NEW200 is blocked, report the missing independent validation;
never substitute or rename historical sources. Failed/NULL gray reconstructions
remain in every denominator. Original V6 paper assets are preserved.
