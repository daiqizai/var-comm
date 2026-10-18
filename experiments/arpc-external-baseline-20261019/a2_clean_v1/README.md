# A2a fixed-cal100 clean source screening, stage 1

The user authorized A0 through gated A3 on 2026-10-19. A1 independent roundtrips
and actual visual/resource review are closed READY_FOR_CALIBRATION. This new
entry covers only the remaining clean source preparation; no channel, metric,
bootstrap, training or automatic A2b/A3 call is included.

The original cal100 hash order is fixed. Ranks0–7 retain their original eight
captions, tokenizations and full probability traversals. A1's 24 exact RX images
at endpoints1/4/7 are reused. Only ranks8–99 get new source calls (92 each).
Their seven endpoints plus the first eight sources' missing endpoints2/3/5/6
produce676 independent RX prefix decodes and images. Total clean states700.
There are no new TX reference renders. Truth is used only after RX exit by the
CPU auditor. The receiver sees copied actual streams, paid endpoint profiles,
public seed0 and the original shared model; no TX CDF/source/embedding access.

`native.py` verifies the SHA of the closed A1 engine, applies eight exact,
reversible population/control edits and executes it. All top-level helper
definitions and all model/codec/preprocessing/render numerical bodies remain
unchanged. Full seven-scale conditioning remains intact. No wire or checkpoint
change. The separately qualified CUDA index4/PCI/driver-UUID mapping is reused;
NVML UUID still identifies the physical GPU and shared lock. No GPU setting is
changed. Four math threads, two interop threads, native mixed arithmetic,
TF32 off, deterministic seed0 and A1 memory admission are unchanged.

Actual A1 TX took80.69s for8 sources including24 redundant reference renders;
RX took53.90s for24 images including loading. The conservative clean-stage
window is5400s: TX≤1200, RX≤3600, audit≤600, with an outer5700s cleanup bound.
This is a new explicitly finite A2 stage, not a reset of A1's closed budget.
Unknown failure preserves all data and reservations, with no automatic retry.

After static qualification, execute `execute.py prepare` once. After its real
exit0 and parent/child gone, execute `execute.py dispatch --expected-registration
<actual SHA>` once. Each child has an actual wait/PID/start/argv/stdout/stderr
receipt. Existing control directories prevent duplicates. Only this owned
process tree may be terminated on timeout. CPU audit emits700 source lengths,
prefix proofs, clean-state provenance and a closed manifest. The original A1
registration, outputs and engineering failure remain immutable.

Next requires a separate frozen DINO-L/PSNR/LPIPS score stage, matched existing
12-MCS reliability at4/10/19dB, actual per-source fallback and an uncertainty
gate with at most two candidates/SNR. No quality is inferred from capacity
frequency, no wireless performance from these clean images, no missing BLER is
set to zero, and no evaluation source selects a strategy.
