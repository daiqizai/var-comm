# Token/channel efficiency supplement
User authorized EXPERIMENT_SUPPLEMENT_PLAN.md on 2026-09-23. This supplements the
short-prefix study; do not duplicate m6/m7 or P3060 baselines. Latest scope defers
new holdout and content selectors. Existing active cache/training must finish at
safe boundaries. The separate scheduling gate holds only the old controller;
its current m6 cache child continues unchanged. Never edit active dependencies.
Order new work A -> B1 -> B2, then resume unstarted C from its original records.
Reuse actual legacy protocols/results only under verified identity. New m6/m10
headers and16QAM are separately versioned. Source bits are not complex channel
uses.16QAM fixed constellation scaling has average rather than per-frame energy.
Use GPU0 only, no hardware setting changes, root Git only, preserve all evidence.

Latest recovery: predecessor is RETIRED_VERIFIED, not a live SIGSTOP process.
Use coordinator_v2/run_budgets_after_retirement.sh. Validate retirement evidence
and acquire the original controller lock. Never signal old PIDs. Relaunch C only
after A/B delivery, with source/checkpoint/cache verification and detached session.

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

Latest automatic continuation: inspect delivery_chain_v1/status.json. It waits
for thermal_guard_v2 to finish both initial budget milestones, then serially
performs calibration-based extensions, actual-weight qualification, B1/B2 and C.
C_first_matrix/C_followups reuse original train/model functions; C_cache adds
signal-aware thermal pause handling without changing cache mathematics. The
retired controller stays retired. Do not launch legacy controllers alongside
this chain. Source bindings prohibit hot edits. Historical reference identity
review/required reevaluation and final verified publication remain delivery items.
