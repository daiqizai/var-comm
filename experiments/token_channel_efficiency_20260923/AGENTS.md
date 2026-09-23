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
