# N2048 full raw calibration CPU scheduling amendment

Registered before preparing or starting the full raw calibration request.
The existing pilot keeps its original eight processes and is not restarted.
The subsequent full raw calibration uses sixteen independent source-shard CPU
processes, which the frozen implementation already admits (1 <= workers <= 16).
Each process retains the original two CPU threads, one interop thread, CPU-only
PHY implementation, numerical flags, source-index counter and source/seed noise.

This changes scheduling only. The complete pilot is still required first.
For each SNR the full stage retains the union of the WHOLE pilot top five,
PARTIAL pilot top five, all nine full-budget WHOLE actions, and all three m10
actions. All 1,000 original calibration sources and seeds 4101/4102/4103 remain.
The exact logical frame count and twice-that-count physical packet ceiling are
registered from this unchanged finite schedule before execution. There is no
additional candidate, source, retry, optimization step or confirmation access.

The parent initializes the shared durable packet ledger once and launches each
source shard exactly once, with 0.3-second startup spacing. It records actual
child PIDs and wait/exit codes. A failure stops owned peers and requires review;
the caller does not start a duplicate cohort or expand the budget.

The CPU phase runs after the pilot GPU closure and the completed entropy GPU
calibration, so these additional CPU processes do not overlap a timed window
or the following raw GPU cohort. GPU work remains two qualified processes and
must use the exact post-CPU received-state plan. No frozen scientific source,
existing execution request, completed result or published policy is changed.

Reason: the user requested parallel execution and higher utilization. The
server has 32 logical CPUs; sixteen independent two-thread CPU-only workers
use the available capacity while preserving per-process numerical settings.
