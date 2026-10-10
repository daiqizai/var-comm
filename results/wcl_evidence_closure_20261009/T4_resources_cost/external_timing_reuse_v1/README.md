# Reused external single-output timing

This directory restores a direct delivery link to the already completed P1024 (public label: Latent continuous JSCC), SwinJSCC-80k adapted, and BPG+LDPC native256 timing measurements. It performs no new timing, inference, channel simulation or bootstrap.

`external_timing_single_output.csv` contains the nine external rows from the published A3 table without changing their numeric strings. `external_summary_original_precision.csv` preserves the original seconds/bytes summaries and available-sample counts. Byte-identical original full tables, per-method completions, raw per-call timings, components, model storage and requests are supplied. Every output listed by the three method completions was checked against its SHA; the entire local verification list is included.

All historical points use N1024 and SNR7/13/19dB, the same fixed16 development sources, batch1, one warm-up and three measured repetitions per source/workpoint (48 measured attempts per point). These repeats are not three independent noise trials. P/Swin use seed2001; BPG uses its A3 development namespace with seed2001. Hardware is RTX4090D, Torch2.11.0+cu128, six CPU threads/two interop threads, affinity4–9, nice15. Full frozen flags and endpoint boundaries are in population_environment_and_boundaries.json and original requests.

The original boundary is CPU CHW uint8 input -> complete CPU1024x2 I/Q -> received CPU I/Q -> one CPU float32 RGB. Actual synchronized outer E2E is measured once; simulated channel time is separately included and subtracted. TX+RX is not substituted for E2E, and inclusive nested component times must not be added to their parent totals. Model loading, disk I/O and quality scoring are excluded. This is software timing, not over-the-air latency or a universal deployment-tail guarantee.

BPG here is native256, not the adaptive-downsampling quality baseline. At7/13/19dB, only1/4/10 of the16 sources can fit, hence only3/12/30 measured link samples. Its TX/RX/link-E2E means are conditional on a transmitted frame. The45/36/18 SOURCE_UNFIT attempts retain real encoding-attempt costs, while full-frame TX/RX/link-E2E remain unavailable. These are not zero latency and cannot be used as full-population deployment means. Swin19dB remains outside training and calibration range.

The historical neural series receipt records actual waits/exit0 for P/Swin and old raw anchors. BPG is authenticated here by its published completion, request, sealed outputs and completed publication audit; a separate BPG process-exit receipt is not present in this local synchronized A3 folder. This export does not invent a new wait observation.

The newer WCL four-arm timing remains independently sealed at `../summary_completed_v2/`, with actual timing source at `.research/wcl_evidence_closure_20261009/t4_actual_timing_v3/`. It uses4/10/19dB and three warm-ups plus three measured repeats. Even where hardware/frozen numerical settings coincide, the historical raw values are retained as historical anchors, not overwritten by or pooled with the new raw measurements. No4/10dB external timings are imputed from7/13dB. Adaptive-BPG and HiFi do not acquire new unified timing rows from this export.

Original published entry: `results/paper_supplement_20261008/a3_timing/final_v1/timing_single_output.csv`, described in `reports/paper_supplement_20261009.md`. The copied original README retains all existing limitations and memory/weight-size scope.

Reproduce to a fresh directory:

```text
python experiments/wcl-evidence-closure-20261009/scripts/t4_reuse_external_timing.py --root . --out results/wcl_evidence_closure_20261009/T4_resources_cost/external_timing_reuse_NEW
```
