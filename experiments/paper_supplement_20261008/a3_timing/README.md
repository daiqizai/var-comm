# A3 fixed16 unified timing

Status at preparation: **NOT_RUN**. The prior independent three-method, eight-source timing is complete and is authenticated as backend validation. It is not counted as the requested new 16-source measurements. This directory contains a fresh bounded request and new timing code; it does not change scientific result files.

The first four methods are partial-scale digital transmission with VAR completion, whole-scale digital transmission with VAR completion, latent continuous JSCC, and adapted SwinJSCC-80k. Each method runs in its own process on the same UM environment, one GPU, six CPU threads / two interop threads, affinity 4–9, nice 15, FP32, batch size one. Other GPU and CPU jobs must be paused for the timing window. No thread settings are changed twice in a process.

The exact historical 16 development/display source order is `[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]`, read from the completed fixed16 cost request. Each source is run at 7, 13 and 19 dB with one warmup and three measured repetitions. The request pins source archives, historical models, protocol implementations and frozen configurations. RAW uses original development counters and noise seed 6201. P and Swin use their original deterministic channel helpers with noise seed 2001. Noise is fixed across repetitions; there is no noise selection. The 19 dB Swin work point remains outside its training/calibration range.

The TX boundary is CPU CHW uint8 input to complete CPU 1024×2 I/Q. The RX boundary is actual received CPU I/Q to one final CPU float32 RGB image. Encoder/VQ, real source coding, FEC/modulation, required transfers, reception/decoding and VAR/Dc are freshly executed as applicable. A failed header yields the original gray image and remains in the timing results.

An actual outer `perf_counter` window encloses TX, channel simulation and RX with GPU synchronization. Its raw elapsed time and the explicitly measured channel window are retained. Software E2E excluding channel equals the actual outer window minus that measured channel window; it is **not** TX+RX. Both values come from the same pass, not an independent second measurement. Unclassified overhead is retained explicitly. Channel simulation includes deterministic noise generation and addition; none of these measurements is air-interface latency. Nested component windows are inclusive and must not be added to their parent totals. The measurement instrumentation/synchronization overhead is included.

File reading/writing, model loading, source/weight hashing, output validation and metrics are outside all timing windows. Metric networks are moved to CPU before measuring. P additionally moves its unused VAR model to CPU. GPU memory is PyTorch allocated/reserved memory with the active loaded model baseline and per-case peak; it is not device-wide memory telemetry. Full loaded VAE storage is disclosed (including its unused decoder), unique parameter objects are deduplicated, and checkpoint disk bytes are reported separately from model tensor bytes. These values do not claim a minimal deployment package, energy usage or joules.

All prior 48-case migration checks are authenticated through completed independent timing receipts. New RAW initializes the same qualified primitive classes directly. It does not call the old R3 constructor's request-equivalence assertion or impersonate the old request. No new qualification or migration packets are sent. Four processes require 768 total fresh frames; independent paid-PHY ceilings are partial 384, whole 384, Swin 192, P zero, totaling at most 960. A header rejection may skip a body. The original scientific ledger is not opened or mutated. Existing failed output directories cannot be silently retried or overwritten. STOP files, a required current absolute deadline, a method wall-time cap and the shared visual lock are enforced.

Prepare materials locally with:

```text
python experiments/paper_supplement_20261008/a3_timing/prepare.py
```

`local_preparation.json` lists the exact three uploads (request, benchmark, RAW endpoint) and their remote destinations. Preparation does not run the experiment. The parent coordinator alone uploads and schedules these serial commands after reserving an idle GPU/CPU window:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 4-9 nice -n 15 \
 /home/liulu/projects/VAR_COMM/outputs/UNIFIED-METRICS-20261002/environment/bin/python \
 /home/liulu/projects/VAR_COMM/outputs/PAPER-SUPPLEMENT-20261008/a3_timing_v1/runtime/benchmark.py \
 --request /home/liulu/projects/VAR_COMM/outputs/PAPER-SUPPLEMENT-20261008/a3_timing_v1/materials/request.json \
 --method RAW64_PARTIAL --deadline-unix CURRENT_AUTHORIZED_DEADLINE
```

Run the other methods serially by replacing the method with `RAW64_WHOLE`, `P1024`, and `SwinJSCC80k`. The coordinator supplies the actual future deadline; no stale deadline is embedded in the request. Each method defaults to a 3600-second cap, configurable at preparation up to 7200 seconds without changing packet counts. Environment and path identities must match the already completed prior timing run; this command does not create an environment or download weights.

Outputs under `.../a3_timing_v1/results/<method>/` are `attempt.json`, `model_storage.json`, one immutable JSON per actual frame, `timings.csv`, `components.csv`, `summary.csv`, `packet_events.json`, and only on successful completion `completion.json` with output hashes. On failure, `failure.json` preserves the actual attempted packet ledger. The summary uses 48 measured samples per SNR, 16 distinct sources, and reports mean, median and P95 with this limited sample scope. Warmups are stored but excluded.

BPG is an additional pending adapter and is not silently included in these four completed-or-pending neural methods. Its memory codec must first pass CLI byte/pixel parity. BPG QP search belongs in fresh TX; cached selected bitstreams cannot be timed as online coding. A source that cannot fit a legal container must be recorded as `SOURCE_UNFIT`, with actual source-codec cost but no invented I/Q or RX/E2E-link timing. Until this adapter is delivered, the full five-method A3 table remains incomplete.
