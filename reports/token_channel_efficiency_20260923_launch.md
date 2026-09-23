# Supplement implementation and scheduling milestone

The user's2026-09-23 supplement is integrated at the sole runtime/Git root.
A source representation and actual bitstream ledger precedes B1 three-budget
quality/resource curves, B2 actual16QAM and the existing short-prefix C.
The latest scope defers new holdout and content selectors. Original A/B and
historical valid P4084/digital/mixed results are preserved.

Implemented: m6--m10 actual integer arithmetic/raw stream writing and independent
RX-probability roundtrip;20 source reference paths per image; calibration-derived
joint quality-target freeze; fresh P2048/P3060 models and shared existing training
engine; explicit low-budget masks and normalization; paid3-bit digital mode domain;
real Gray16QAM and Gaussian-mixture soft demodulation; explicit capacity/overflow;
durable stage coordinator and safe process-identity handoff.

CPU acceptance:110 tests pass, including20 new synthetic engineering regressions.
Repository verification and release checks pass. These do not certify quality,
real-weight GPU acceptance, training or completed B1/B2 results. The run first
executes a two-source real-weight A qualification, then full calibration A and
freezes targets before original development. Budget models receive actual-device
optimizer-isolation/gradient/energy/bitwise-resume qualification before20k training.
All dependent work stops on failed acceptance. No GPU result is fabricated here.

At integration the original m6 train-cache generator was healthy and unchanged.
Its controller was intentionally SIGSTOP-held while that child continued to a
safe complete cache; no training weights were being updated. Original queue and
cache source bindings were reverified unchanged. A separate lock-protected
coordinator checks PID start times, commands, full cache hashes and GPU occupancy.
The held controller must be resumed by the monitor after A/B delivery; it must
never be treated as permission to start a duplicate C queue.

Reuse audit: original frozen F/token/image shards are reused under hashes/order;
legacy digital m7/m8/m9 protocols remain unchanged. The new full mode domain costs
70/96 header uses, so old68/94-use results cannot be relabeled as its measurements.
Actual new source streams and full source-reference grid are missing and must be
measured. Valid legacy communication points can later be included under their
own protocol identities. P3060 is registered once for both B1 and C.

Remaining NOT_RUN: real source/quality results at commit time; P2048/P3060 training;
complete new QPSK quality/evaluation grid and policy freezing; online timing sharing
actual TX/channel/RX; real16QAM image results and E distributions; selected-model
statistics/figures; calibration-driven continuation and remaining short-prefix
confirmation. The initial coordinator stops explicitly at the implemented source+
low-budget20k milestone and records later tasks pending. It does not pretend to
have completed the scientific plan. The active heartbeat continues implementation
and acceptance from those durable records.

Local runtime: outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/status.json and
scheduling_gate.json. Source bits, tokens, weights, original pixels and tensor
caches remain local. Only reviewed source, configuration, CPU receipts and this
scope report are committed in this milestone.
