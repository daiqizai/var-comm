# T6 RAW full1000 calibration: actual CPU-affinity amendment

This appendix records a resource scheduling amendment applied to the two existing T6 RAW full1000 calibration GPU workers. The original execution request remains unchanged, with SHA-256 `700115527dd9f05da0263aab38f98758d16a6be8d0e1be93623f6fda33090b75`. Its inherited CPU placement of 4-9 describes the initial segment; the separately recorded amendment supersedes that resource setting for these workers from the documented application event onward.

The intent was recorded before the change at Unix time `1791561157.1761322`. All recorded thread changes were complete by `1791561157.1963463` (2026-10-09T15:52:37.196346+00:00). The intent snapshot contained 147 completed source checkpoints and 4,120 completed new state checkpoints. Those counts identify the snapshot, not a forced source boundary; work already in flight may span the scheduling transition.

| Worker | PID | Start ticks | Observed TIDs | Before allowed CPUs | After allowed CPUs |
| --- | --- | --- | --- | --- | --- |
| 0 | 3898548 | 220775240 | 22 | 4-9 | 4-31 |
| 1 | 3898549 | 220775240 | 22 | 4-9 | 4-31 |

The receipts preserve identical PID, process start ticks, UID and argv before and after. All 44 observed TIDs were changed and verified at application completion. The amendment restarted zero workers and introduced zero additional scientific calls. Scientific inputs, source ordering, models, code, seeds, candidates, samples, batch sizes, deadlines and call budgets remained unchanged. Torch intra-op/inter-op settings remained 6/2, and no BLAS/OMP settings, precision settings, deterministic flags or environment variables were changed.

The independent code review found no affinity enforcement/reset or CPU-count-dependent scientific algorithm selection in the inspected frozen runtime path. Existing exact concurrent numerical qualifications were performed under the original 4-9 mask. No numerical qualification was repeated under 4-31, and the earlier qualification is not relabelled as a new-mask qualification. This disclosed calibration scheduling change does not establish a new bitwise-equivalence result.

Calibration resource and throughput records must be separated at this amendment event. This amended calibration run must not be used as T4 same-environment timing evidence. This appendix records the application event only; it makes no claim about later completion or performance benefit.

Evidence identifiers (original receipt bytes were checked locally):

- `intent.json`: SHA-256 `d49b3270b3cccfb3258c612285de5a902ee5c7145c3a056f31c8040106db8f6a`.
- `applied.json`: SHA-256 `f9a96b3f5ef0eee46fa836a717f3c1b4c5dff5d23e34fd1a149dd216813fc21a`; binds the preceding intent SHA.
- `independent_review.json`: SHA-256 `53a1cd0e94614ff7c7ad347c2c48ad706c708fb97f18da6f153de5999c11e717`; lists 19 inspected source paths and their matching frozen SHA-256 values.
- `independent_review.md`: SHA-256 `dfd319b591fb0b981095e4a6a4348be44dc7a9015c058e701afc1d04f2222a10`.

Original execution request: `/home/liulu/projects/VAR_COMM/outputs/WCL-EVIDENCE-CLOSURE-20261009/T6_raw_calibration_full1000_v1/execution_request.json`.

This appendix was generated from the downloaded actual receipts without SSH, scientific execution or modifications to frozen scripts or receipt files.
