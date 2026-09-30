# Frozen P4084 receiver study: Step2 A

This run completed on 2026-09-30 with zero training updates. It uses the C priority first training seed 2026092304, P4084 selected27500, its matching frozen Dc, the official ten-scale VQ and unconditional VAR. The original N=4084/E=8168 transmit/receive chain produces one shared AWGN observation per source/SNR/noise.

- [Execution plan](EXECUTION_PLAN.md)
- [Reviewed conclusions and full tables](../../results/rx_posterior_step2_A_20260930_R1/report_reviewed.md)
- [Results and exact table reconstruction](../../results/rx_posterior_step2_A_20260930_R1/README.md)

## Completed result

The policies select BYPASS at 1/4/7/13 dB, so the final images equal B2. At the out-of-training-range points -5/-2 dB, V exceeds B2/A1/A2 under the preregistered G1 rule; the content advantage remains with common A1 fusion weights. Running V costs about 125 ms versus 13 ms for B2. High-SNR protection comes from bypass; unbypassed candidates worsen LPIPS. This is an exploratory reused-development study, with no new holdout or training-seed replication.

## Assets and entry points

Use the existing VAR_COMM Python/GPU runtime. The code requires the external official VAE/VAR, matching Dc, C priority P4084 selected27500 and its lineage, Step1 v3 calibration statistics/cache, original calibration/development pixels, LPIPS alex and DINO weights. Weights, datasets, tensors, caches and environments remain outside Git. Paths resolve from this project root and the existing environment configuration in Step1 `run_preflight.py`.

```bash
python3 experiments/rx-posterior-step2-A-20260930/run.py --preflight-only
python3 -u experiments/rx-posterior-step2-A-20260930/supervisor.py
```

`run.py` performs the science pipeline. `supervisor.py` owns only this A inference and retries only safe resource pauses (exit 75); unknown failures stop. `receiver.py` implements pure closed-loop correction; `analysis.py` recomputes CPU statistics.

The fixed R1 output/results directories are already complete. `run.py` returns immediately when its completion receipt exists. Preserve R1 and register a new run ID for any separately authorized replication; do not delete receipts to rerun it. The commands above document the entry points, not authorization to launch B or old queues.

## Frozen boundaries and evidence

Calibration uses the existing 200 sources and seeds 4101/4102/4103. All six SNR grids finish before lambda, per-method alpha and SNR-level actions are frozen. Only then are the fixed 100 development sources and seeds 2001/2002/2003 opened. Config and model/source hashes, cached observation checks and development pixel identities protect these boundaries.

Lambda 0 still quantizes/fuses and skips VAR; it is an A1 identity alias. True BYPASS returns the same P4084 feature and reuses the same B2 image. Clean F and original tokens are isolated to TX/calibration/scoring, never used to change the deployed RX decisions. Common alpha is exactly calibration A1 alpha; it does not select a new lambda.

Four necessary real-model selfchecks passed. B2 reproduced the original published four-SNR result. All 25,200 metric rows, 108,000 token rows, 360 timing records, source-paired bootstrap intervals and frozen weights were audited. Timing covers CPU waveform through P.receive/correction/Dc to CPU image; single-frame timing images were not compared pixelwise with batch quality images.

To inspect/recompute the published tables on CPU:

```bash
python3 tools/archive_rx_step2_tables.py restore
step2_review=$(mktemp -d)
cp -a results/rx_posterior_step2_A_20260930_R1/. "$step2_review/"
python3 experiments/rx-posterior-step2-A-20260930/analysis.py \
  --out "$step2_review/run_receipt" \
  --results "$step2_review"
```

Restoration and analysis do not run GPU inference or select parameters. This example analyzes a copy to preserve the original report, figures and result byte hashes.

B/P_low and Step3 require a separate user decision. The old queues remain paused.
