# M1 N2048 independent GPU workers

This is an execution revision of the already registered M1 N2048 experiment.
It introduces no new method, training, source population, action grid, noise
seed, precision or metric. Every model instance keeps batch size one and the
original six CPU intra-op threads. Frozen original source files remain unchanged.

## Qualification and choice

Recompute the same six already completed calibration sources, indices 0–5,
with one and then two independent GPU processes. Compare all 2,070 frame rows
per source, including reconstructed floating-point image hashes, against the
original sealed checkpoints. Timing fields outside the scientific rows are
reported separately. No development result selects the execution setting.

Compare steady throughput, startup cost, memory and temperature. Four workers
may be measured if two leave enough GPU capacity. Use parallel production only
when exact row parity passes and measured throughput improves by at least 15%.
No benefit with valid results means returning to the original serial execution.
Numerical discrepancies or unknown runtime failures stop for review.

## Ownership and preservation

The coordinator owns the original execution lock while workers run. Each worker
has a disjoint set of complete sources and an additional per-source file lock.
Benchmark reconstructions go to separate scratch directories. Production writes
only missing original calibration source checkpoints with their original binding;
already completed checkpoints are validated and reused.

Only the registered worker cohort may share this GPU. Membership checks use the
PID, process start time, user ID and complete launch arguments. Other GPU users
remain excluded. Original thermal checks and five-second healthy polling remain
active; hardware settings and shared GPU services are unchanged.

After production workers exit, the original serial calibration entry aggregates
source rows in the original order and selects policies. The original development,
unified metrics and publication chain then continues at the same output paths.
The downstream UEP queue keeps waiting for that original publication receipt.

Parallel throughput is an offline evaluation measurement. Receiver latency is
still measured with isolated execution.
