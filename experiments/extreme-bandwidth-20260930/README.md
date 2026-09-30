# N512 extreme bandwidth experiment

This is the explicitly authorized first stage of the revised execution plan. It trains one fresh P512 and compares calibrated raw digital chains and frozen receiver plugins at 1/4/7/13/19 dB. N1024 and generative training require a later decision.

## Frozen boundaries

- Continuous and QPSK use N512 and actual per-frame E1024. The paid digital header uses68 symbols and the body444.
- QPSK permits m4/m5. The separately registered16QAM family permits m4/m5/m6 and retains the original fixed constellation scaling. Its actual energy is reported.
- Digital C uses only the class decoded from the paid header. Digital U ignores that class for completion. Header failures preserve the original gray image with no actual received latent; all image quality metrics include failures. Latent errors are conditional on a valid estimate with counts.
- P512 follows the original fresh P2048 recipe with explicitly new seed2026093001, no communication parent weights or optimizer state. It first trains20000 updates; the registered full1000 calibration gate can extend to30000/40000. Development cannot choose checkpoints or stopping rules.
- Digital policies use all1000 calibration sources. Receiver plugins recalibrate on200. Development uses the frozen100 sources and three noise repeats. D0 renders the same digital latent/action as Dc.
- H_D and H_R use source-level paired bootstrap intervals. H_R must pass one common quality branch against all three controls. BYPASS is not an incremental gain.

## Execution

Run CPU checks and the real discarded-update GPU qualification before formal training. The detached supervisor owns the sequential stages in stages.json and retries only safe resource pauses (exit75). Unknown failures stop and preserve evidence. Bound source files are immutable while the run is active.

```bash
python3 experiments/extreme-bandwidth-20260930/test_training_rules.py
python3 experiments/extreme-bandwidth-20260930/test_digital.py
python3 experiments/extreme-bandwidth-20260930/test_analysis.py
python3 experiments/extreme-bandwidth-20260930/qualification.py
python3 experiments/extreme-bandwidth-20260930/publish.py --register-only
python3 -u experiments/extreme-bandwidth-20260930/supervisor.py
```

The supervisor runs source references, complete digital calibration, training, plugin and digital development evaluation, analysis, and checked normal Git publication. It never resumes old queues or starts N1024.

Outputs and weights remain under outputs/EXTREME-BW-20260930-R1. Lightweight results go to results/extreme_bandwidth_20260930_R1 and the report to reports/extreme_bandwidth_probe_20260930_R1_N512.md. Full digital calibration and large measured tables are archived using the existing byte-preserving table format, with exact hashes and row counts; checkpoints and tensor caches remain outside Git.
