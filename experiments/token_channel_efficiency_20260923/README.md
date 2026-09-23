# Token/channel efficiency supplement (2026-09-23)

User-authorized scope is the verbatim [supplement](EXPERIMENT_SUPPLEMENT_PLAN.md).
This is an extension of the registered short-prefix study, not a second copy of
its m6/m7 experiments. Latest priority is A -> B1 -> B2 -> merged C. No new
holdout or content selector is authorized in this round; this supersedes the
older future500-test/selector launch instructions without rewriting their evidence.

## Implemented milestone and live scheduling

- A: actual integer raw/arithmetic streams m6--m10, independent autoregressive
  token roundtrip, D0/Dc x F/Fq and m6--m9 cumulative/VAR source reconstruction.
  Source references have no finite-channel N. Calibration and development stay
  separate. Three joint PSNR/LPIPS targets are frozen from complete calibration
  before any new development access. Actual bitstreams remain local; only
  source lengths, hashes, receipts and measurements are publishable.
- B1: fresh budget-specific P2048/P3060 output heads and masks, actual N from
  step0, existing shared loss/update/calibration implementation, frozen Dc,
  full calibration every2500 and checkpoint every500 through the initial20k
  milestone. P3060 is the single shared baseline for C. Existing P4084 10k
  remains valid only under its original identity; no renamed/truncated model.
- B2 engineering support: versioned paid3-bit mode header (70 raw/96 arithmetic
  QPSK uses), actual mother FEC and rate matching,16QAM Gray mapping and Gaussian
  mixture soft demodulation with the decoder's half-LLR convention. Actual
 16QAM E varies by frame; its tables must be separate from per-frame2N.

Initial CPU engineering regressions used synthetic inputs and are not quality
measurements. Experiment A is now complete for1000 calibration and100 original
development sources; see reports/token_channel_efficiency_20260923_source_A.md
and results/token_channel_efficiency_20260923/source_{calibration,development}.
Quality targets are frozen and committed. Real-device gradient/energy/optimizer/
resume qualification passed for P2048/P3060; budget training remains ongoing.
Later stages are complete only when their actual receipts exist.
The coordinator runs real2-source calibration qualification first, then full A,
then real-device budget qualification, then P2048/P3060 training. Any exception
stops dependent work. The complete digital quality/evaluation grid, online timing,
calibration-driven continuation, B2 results and combined statistics are still
NOT_RUN; they require subsequent implementation/qualification and monitoring.
Reaching this queue's final20k milestone is not completion of the supplement.

The original short-prefix controller is intentionally held before its first m6
training command. Its existing m6 train-cache child continues unchanged to a
complete, hash-checked cache. The independent coordinator checks PID start time,
command, controller stopped state, cache identity and GPU ownership before
running A/B. It does not modify active dependencies or resume C prematurely.
After B1/B2 delivery, the monitor must verify the stored process identity and
send SIGCONT to that held controller; never start a duplicate short-prefix queue.
If the old process no longer exists, inspect durable original stage receipts and
its source binding before an explicit recovery. Do not signal a recycled PID.

## Execution and receipts

From the sole repository root:

```bash
bash experiments/token_channel_efficiency_20260923/scripts/run_source_and_budgets.sh
```

Use this once or after inspecting a stopped/failed coordinator; it takes a
process lock. Runtime entry is
`outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/status.json`, with
`scheduling_gate.json`, stage receipts, registration hashes, logs and checkpoints.
The previous queue remains `outputs/SHORT-PREFIX-20260923/status.json`.
Run only on authorized GPU0. Foreign work or sustained thermal throttling causes
waiting/safe pause; hardware settings are unchanged.

## Remaining acceptance and output

After actual source A, publish source_codec_per_image.csv,
source_codec_summary.csv, payload_ledger.csv and frozen quality_targets.json.
For B1/B2, implement common real online TX/channel/RX for quality and timing,
verify legacy-cell reuse identities, calibrate policies then evaluate original
100 development only. Keep all failures and explicit unencodable cells. Aggregate
same source/SNR/seed objects, bootstrap source images10000 times and retain all
quality targets and unmet outcomes. Publish actual resource curves, minimum tested
N, failure/energy distributions, fixed samples with permitted data, and lineage.
Do not publish weights, source pixels, tokens, bitstreams or large tensor caches.

## Shared online execution follow-up

`token_efficiency.execution` now provides actual online TX/channel/RX for both
new continuous budgets and versioned digital cells. `execute` times CPU endpoints
using the same `transmit`, `apply_channel` and `receive` functions used for
quality; channel noise is applied once outside RX timing. Continuous TX does not
read class/token/VAR state. Arithmetic RX preserves legal partial candidates
after body failure, and reuses the actual decoding stream latent rather than
repeating a second VAR completion. Full m10 raw uses cumulative quantized latent.
The selected-budget loader checks selected step, arm, N, registration and actual
checkpoint SHA; it does not assume step10000.

Five synthetic interface/selected-loader tests accompany this module. The real
calibration-only entry `python -m token_efficiency.qualify_execution` is NOT_RUN
until its acceptance receipt exists in `outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/
execution_qualification_v1/`. It must run with GPU0 free before the new digital
quality grid; do not start it alongside the existing coordinator's GPU child.
Selected trained continuous models require replay qualification again at evaluation.
These modules are additive and do not alter the running cache or coordinator.

## Strict frame statistics follow-up

`token_efficiency.statistics.FrameTable` requires the external frozen source/RGB
identity map, SNRs, noise seeds and explicit eligible method roster. It rejects
missing or duplicate frames, nonfinite values, changed run/model/protocol/noise
context, wrong preprocessing, noninteger resource ledgers and false energy claims.
Unencodable candidates belong in the separate candidate-status ledger and cannot
be represented by dropping selected images. All actual failure frames remain.

Main means and paired intervals share one validated data object: noise means per
source/SNR, then source means;10000 source bootstrap resamples. Metadata records
both method contexts and makes no claim that equal seed numbers imply identical
observations across waveform lengths or namespaces. Training-seed variation is
not included in these intervals.16QAM actual E distributions stay separate.

Digital policies minimize calibration MSE+0.1LPIPS for each family/N/SNR/MCS/energy
constraint, with deterministic method-name ties and DINO report-only. A policy may
select an entire qualified protocol version per pre-shared cell; it cannot splice
per-source protocols for free. `context_sha256` binds shared model/Dc/PHY/numerical
configuration and must remain the same between calibration and development;
`run_id`, population/preprocessing and actual seed grids bind each evaluated run.
Development never selects the digital mode. Continuous candidates must be the
actual calibration-selected checkpoints, one per tested N. Minimum-N tables use
only measured budgets, retain every frozen target, negative savings and explicit
not-reached cases. Mean target attainment and per-frame joint success are distinct.
The14 added tests are synthetic statistics tests, not scientific results. No
quality/resource conclusion is complete until the real frame grids pass this gate.
