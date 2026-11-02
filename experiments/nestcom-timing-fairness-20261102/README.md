# 当前：编译成对prefill正式复测已闭合

原32源、320次收发计时全部完成，独立前缀核验160/160。TX 114.55 ms，RX 2949.76 ms；成功资格精确复用，三次环境失败保留。当前GPU总上限18000秒，绝对截止不变。数值实现变化与旧质量分开报告。见[当前结果](../../results/nestcom_timing_fairness_20261102_v1/paired_prefill_delivery/REPORT_zh.md)。以下为较早阶段的范围、协议和状态记录。

---

# Timing fairness audit and bounded rerun, 2026-11-02

Current outcome: all 19 final timing stages and the separate four-prefill
diagnostic have actually exited successfully with complete accounting. All
2,440 canonical timing outputs match their historical event identities. The
prefill check is **not equivalent to the frozen incremental CDF/stream route**;
this negative result is retained. Author-aligned Ada end-to-end timing remains
uncompleted. See [the final report](../../results/nestcom_timing_fairness_20261102_v1/delivery/REPORT_zh.md).

Latest clarification concerns unnecessary Ada implementation overhead. The
author's IV-E/Table IV describes encoder prefill and `torch.compile`, with
19 ms encoder and 1012 ms decoder on an RTX5090/Xeon8260, for tokenizer, AR
probabilities and softmax/arithmetic coding. Our serial encoder and unused
unconditional TX branch are not intrinsic Ada costs. Correcting CPU topology
alone does not make this an author-aligned latency comparison. The main paper
withholds Ada timing and any NeST/Ada speedup ratio; raw results remain as an
implementation audit. See [the audit](FAIRNESS_AUDIT_zh.md).

`ada_prefill_identity.py` is a separate bounded diagnostic after all timing
workers close: two original sources, conditional-only and original-two-branch
prefills, four model calls total. It compares all 256 integer CDFs and terminated
prefix streams with the preserved incremental-route references. It reuses exact
tokens and classes; no tokenizer/classifier/render/PHY/quality calls or training.
Its resource use is charged to the original deadline and three-hour GPU total.
It is not a timing benchmark or a claim of reproducing the author's compiled
implementation. Non-equivalence is retained without automatic retry or new scans.

User authorization: “计时的公平性应该有问题，看看能不能重新跑一下”.

This is an independent timing study. It does not resume the completed E5 queue,
reset an old deadline, train models, select policies, or recompute quality scores.
The original results and failure records remain immutable.

The old affinity `[38,39]` selects the two SMT siblings of physical core 19 on
socket 0, although the manuscript described two CPU cores. The intra-op pool was
six threads. This study uses `[38,40]`, two different physical cores, with two
intra-op threads and the original two inter-op threads. Loaders first verify the
historical numerical route; the effective timed thread setting is recorded
separately after loading. TF32, precision, attention kernels, sampling, models,
MCS, candidate sets, source data, and received observations stay fixed.

Scope: the original 32 timing sources, N=1024, 10 dB, WHOLE, NeST-Com,
Ada-TokenCom adaptation, SwinJSCC, Latent JSCC, and BPG+LDPC. Each direction has
two warmups and three measured trials per source: 12 workers, 1,920 calls total.
TX starts with native source pixels and includes source coding and the digital
transmitter; RX starts with the recorded noisy symbols and includes recovery and
rendering. Setup, input loading, hashing of final outputs, and saved logs are
outside the timer. All CUDA work is synchronized at both boundaries. No timing
output or intermediate source/model result is reused inside the timed path.

Every new emitted-wave or recovered-image hash is checked against its original
E5 event **outside** the timer. A mismatch stops the affected stage. A failed
stage is not retried automatically; other independent stages may finish.

Four additional diagnostic workers use only the first two fixed sources for
Ada and NeST, TX and RX (40 calls). They move Ada's redundant logits-finiteness
audit after the timer, omit its unused per-token CDF hashes, and omit NeST's
unused completed-token hash. Protocol parsing, length checks, CRC, canonical
arithmetic re-encoding, actual token generation, and device precision remain.
These diagnostics quantify instrumentation sensitivity; their two-source times
do not replace the 32-source main comparison or justify an algorithmic speedup.

Maximum: one owned GPU, serial workers; four hours from the new first dispatch,
three GPU device-hours, one CPU process-wall-hour for BPG, one hour per worker.
An idle PRO6000 is selected once. Foreign GPU use interrupts only this study's
worker; no other process or device settings are changed. Resource telemetry,
PID/start ticks, actual wait status, and a unique call ledger are retained on PFS.

The initial scope did not include historical N512, 7 dB, granularity or ARPC.
The finite addendum below includes only the three missing matched-grid TX cells.
N512, 7 dB, grid RX and ARPC remain historical; their resource protocol is explicit
where retained.

## Registered additions and resource revision

`granularity_timing_addendum.json` adds Whole/Quarter/Eighth TX at the already
fixed ceiling `(8,126)`: 32 sources, five calls each. The Token cell reuses the
identical main NeST TX stage. This adds 480 calls, not new image-quality trials.

The initial GPU0 queue encountered an unrelated diagnostic process. Its completed
WHOLE calls and interrupted NeST calls remain on PFS with their original ledgers,
exit statuses and diagnostics. The original queue and first-launch-only extension
were stopped at the study boundary; no foreign process was changed.

`clean_gpu6_v1/resource_revision.json` registers one clean replacement session on
GPU6. It waits for the previous owners' actual exits, then measures all five GPU
methods on that one physical RTX PRO 6000. The two completed BPG CPU stages are
reused because their CPU topology, binaries, configuration and inputs exactly
match. No GPU0 timing enters the new comparison. Prior resource use is deducted
from the same limits, and the absolute deadline remains Unix 1793640317.2021759
(2026-11-03 01:25:17 Asia/Shanghai). The revision permits no further automatic
device retry.

The canonical comparison contains 2440 boundary calls: 1920 main-method calls,
480 grid TX calls and 40 diagnostic calls. Of these, 320 CPU calls are reused
from the initial session and 2120 are newly performed on GPU6. Excluded and
interrupted initial GPU0 calls are counted separately, not erased or mislabeled
as reused. `finalize_clean.py` aggregates the explicit final-stage manifest only
after actual owner/worker closure and output-identity checks. `summarize.py` is
the superseded union entry and is not the final aggregation command.

## Interpretation and reproduction

Equal device and CPU resources establish a common execution envelope. Models,
attention backends and serial work remain implementation-specific: the Ada
adaptation evaluates token probabilities sequentially with its frozen FP32 math
attention route, while NeST predicts scales. The main timing retains both
implementations' original audit operations. The two-source diagnostics only
test selected audit overhead; they are not optimized replacements for the main
comparison and are not subtracted from its means.

`test_aggregation.py` contains CPU-only checks of within-source averaging, paired
TX+RX sums and incomplete-cell rejection. The new source-bootstrap bank uses
seed 2026110201 with 10000 resamples of the fixed 32-source order. Its intervals
describe source variation in this session, not all possible host interference.
The clean-versus-historical deltas change session/device and CPU resources
together, so they cannot isolate the causal effect of CPU affinity.

All registration files are one-shot execution evidence. Inspect the explicit
PFS paths and current process identities before any action; do not run a saved
launcher again to reproduce a completed session. Large arrays, per-call native
outputs, models, build caches and logs stay on PFS.
