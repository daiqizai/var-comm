# Software processing latency: retained measurement conditions

The table reads all 15 existing mean rows at 4/10/19 dB directly from
`inputs/timing_summary.csv`. Median, p95 and unrounded values remain in that
unchanged CSV. No new sample, average over SNRs, interval or timing curve is added.
Total = TX + RX (source/PHY transmitter and PHY/native receiver components).
Displayed entries use two decimal places; audit uses unrounded values.

- N=1024; fixed16 development sources; batch size 1; three fresh warmups then
  three measurements per method/source/SNR, 48 measured calls per table row.
- Same PRO6000 GPU (UUID `GPU-f5047136-08e5-9416-3c86-87786d0b355e`), host
  `aibox-r59688baa77c-57c794f5c6-44lm7`, and CPU affinity 0–5; Torch intra/inter-op threads 6/2.
- Original software stacks are retained: visual methods Python3.10/Torch2.11/CUDA12.8;
  ARPC Python3.12/Torch2.10/CUDA13.0; CPU codecs use their frozen Python3.11 environment.
  TF32 is disabled. ARPC retains FP32 VAE/prior parameters and its original mixed
  FP16/FP32 text encoder. Do not describe all internal arithmetic as identical FP32.
- Shared load was recorded, not eliminated. The existing stage-boundary GPU
  utilization snapshots range from 0% to 54%; they do not establish exclusive
  access or characterize every timed call. Full resource snapshots remain in
  `inputs/timing_resources.json`.
- BPG uses its unchanged binary with the disclosed child-only CPU-count adapter.
  The adaptive resolution/QP search and candidate decoding are included.
- Required BPG codec subprocess/file conversion/cleanup is timed. Model setup,
  AWGN synthesis, cross-process handoff and audit/result persistence are excluded.
- Noise is fixed within each method's repeated calls; namespaces differ across
  methods. All formal receiver outcomes remain in the denominator.
- These are **software processing latency** measurements, not wireless
  transmission or air-interface-inclusive end-to-end latency. Historical H800
  six-method measurements are not combined.

The original reports retain the pre-case receiver import failure, incomplete BPG
attempt and explicit continuations; this table does not relabel those attempts
as successful. See `inputs/timing_report.md` and `inputs/timing_method_notes.md`.
