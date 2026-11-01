# E5 runtime repair — 2026-11-01

The remaining timing queue is active. This is a repair and qualification delivery,
not a completed E5 study or final manuscript delivery.

The common timing path now binds the existing pure-Python Sionna dependencies
without changing the frozen Torch/NumPy routes. Actual CPU runtime construction
passed. CUDA and NVML UUIDs are separately pinned to the same physical PCI device.
The supervisor now reaps its actual child; the completed first TX is reused with
exact registration, rows and waveform hashes. No completed measurement is rerun.

ARPC retains its original system-Python model runtime (Torch2.10), while the
unchanged, strictly qualified LDPC runtime executes under frozen_ldpc_py311
(Torch2.11). A memory socket connects the processes. All per-call transfers are
inside the measured TX/RX wall boundaries; model/PHY loading remains excluded.
Both processes share the same registered two CPU cores and the original GPU.
This implementation cost must remain visible in the timing interpretation.

The independent CPU qualification actually sent and received two synthetic full
digital frames: one paid payload and one paid NULL. The payload and status matched;
both original model-free parent and PHY child exited0/gone. These are engineering
checks, not image-quality or wireless benchmark results. The first attempt's
metadata-key failure occurred before any frame calls and is preserved verbatim.

E5_ARPC_IPC_queue_v2 is uniquely registered and live. It waits for the actual exit
and disappearance of E5_timing_recovery_v6, then owns only the two previously
unlaunched E5_dep5_g09_tx/rx stages (160 calls each:64warmup+96measured).
At this snapshot ARPC GPU timing has NOT started. The source32, numerical model
route, PHY, samples, seeds,72GPU/72CPU-hour limits and absolute deadline are unchanged.
Additional PHY-service process wall time is separately recorded and reserved in
the CPU budget. Unknown failures stop the affected chain without automatic retry.

E1/E2/E3/E4/E6 remain closed and unchanged. Full E5 statistics, its ARPC timing,
final V7 documents and final publication acceptance remain pending. Old lineage
summary helpers v2/v3 were never run and are not valid for the current dep5 map.
