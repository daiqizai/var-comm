# VAR_COMM collaboration rules

## Single Git worktree (2026-09-23)

- The directory containing this file is the actual project root and the only daily Git worktree. Run, test, edit, commit and push here to `daiqizai/var-comm`, branch `main`.
- `publish/var-comm` is an ignored historical backup. Never import code from it, copy daily changes to it, or require a publication/export step before committing. Do not delete it during this migration.
- Preserve remote history; inspect remote changes before a normal push. No force push, hard reset, clean, or moving `.git`. Use the existing local SSH configuration; never commit credentials.
- Explicitly stage reviewed source, configuration, scripts, tests, tools, reports and lightweight results. Data, weights, tensors, environments, build/cache files and local backups stay untracked. JSON/CSV/figures are not excluded wholesale.
- Run `python tools/verify_repository.py` and `python tools/run_cpu_checks.py` from the root before committing. The latter records synthetic CPU engineering tests, not real image quality.
- Read README.md, RESEARCH_STATUS.md, PROGRESS.md and EXPERIMENTS.md. Older instructions in docs/history are evidence, not live launch authorization.
- Preserve A/B checkpoints, old experiments, failures, source snapshots and historical numerical results. Do not retrain A/B, create methods or claim all repair issues are solved as part of repository migration.
- No source hot-editing under an active affected task: first save/pause only this user's work at a safe checkpoint. Do not affect other users or shared GPU settings.
- RX cannot use unsent targets, hidden TX state, oracle errors or free side information. Keep registered N/E, PHY, losses, data, noise, frozen choices and train/calibration/development boundaries.
- Do not treat old synthetic probe PASS or completion labels as real-weight acceptance. GPU/model/data-dependent work not executed is NOT_RUN.
- Push every authorized commit normally, verify the remote SHA, and report failures honestly. Repository visibility and access stay unchanged.

## Latest experimental scope (2026-09-23 supplement)

- Follow experiments/token_channel_efficiency_20260923/EXPERIMENT_SUPPLEMENT_PLAN.md
  together with its README and protocol. A -> B1 -> B2 -> merged short-prefix C.
- This round defers new holdout and content selectors, overriding older launch
  plans for a500-source new test. No original A/B retraining.
- Check outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/scheduling_gate.json and both
  queues before action. The original controller may be intentionally SIGSTOP-held
  while its cache child continues. Never duplicate it or hot-edit its bindings.
- P3060 is shared across studies. Actual real receipts, not code presence or CPU
  synthetic tests, determine stage completion. Follow pending stages to delivery.

## Scheduling recovery (2026-09-23, latest)

The old short-prefix controller exited after its stopped state was lost following
SSH disconnection. Its attempted m6 launch was rejected before model loading by
GPU ownership checks; no m6 checkpoint exists. The predecessor is now explicitly
RETIRED_VERIFIED in scheduling_gate.json, with immutable evidence and hashes.
Do not SIGCONT its old PID. Use token_efficiency.coordinator_v2 and
scripts/run_budgets_after_retirement.sh for the supplemental budget queue.
The successor holds the original controller's process lock and validates retirement
and source/cache evidence. P2048 resumes its verified step5967 safe checkpoint.
After actual A/B delivery, a separately verified detached launch of the unchanged
original queue may requalify and reuse caches. Until then C stays file-gated,
not suspended as a live SSH-associated process. Never edit active dependencies.

## Software thermal throttling guard (2026-09-23 UTC)

The latest budget launch uses scripts/run_budgets_with_thermal_guard.sh, wrapping
the unchanged coordinator_v2. Its durable status is
outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/thermal_guard_v1/status.json.
It checks hardware AND software thermal slowdown every10 seconds; three
consecutive hot samples (either flag, or temperature >=86C) request the owned
coordinator to stop at a safe checkpoint. Six consecutive samples <=75C with
neither flag active are required before detached resume. It never changes GPU
settings, training code, batch, precision, loss or optimizer. Process identity
and checkpoint SHA are verified; unknown failures do not automatically retry.
For a requested pause, stop the verified thermal supervisor so it cannot resume
the budget queue. Do not independently launch a second coordinator.

## Latest authorized memory utilization change (2026-09-24 local)

User explicitly requested increased GPU memory utilization. P2048 now uses the
versioned microbatch8-v1 runtime, effective batch16 unchanged; P3060 remains on
its original execution until separately qualified. Actual benchmark showed only
about3.4% throughput improvement; no large acceleration is claimed. FP32/TF32,
loss, data order, total updates and hardware settings are unchanged. Floating
point accumulation differs slightly, so do not claim the old trajectory is bitwise
identical. The initial migration checkpoint is step8862 with recorded SHA.
Latest launch: scripts/run_budgets_microbatch_guard.sh, thermal_guard_v2 ->
coordinator_v3 -> budget_train_microbatch for P2048. Inspect thermal_guard_v2/status.json,
coordinator_identity_v3.json and per-checkpoint execution_identity. Do not start
the old guard/controller alongside it. Old registrations/checkpoints/receipts are
historical and immutable; the runtime extension records actual microbatch, code,
benchmark and migration identity. Selected loading verifies this extra lineage.

## Full authorized continuation (2026-09-24)

The user requests execution of all currently authorized experiments, including
calibration, evaluation and delivery. Inspect the registered delivery_chain_v1
status as well as thermal_guard_v2. Its detached entry is
experiments/token_channel_efficiency_20260923/scripts/run_delivery_chain.sh.
It waits for the existing budget supervisor, then owns all remaining GPU stages.
Never duplicate either process or hot-edit its bound source. C now uses the
versioned C_first_matrix/C_followups orchestrators with original model/train
engines and a signal-aware C_cache adapter; never resume the retired controller.
Keep scope: no new holdout, no learned selector, no original A/B retraining.
A source-informed calibration diagnostic is not a deployable content selector.
Historical reference compatibility/necessary reevaluation and final reviewed
publication/independent remote checkout remain required after the real grids.
An execution milestone with pending items is not complete study delivery.
