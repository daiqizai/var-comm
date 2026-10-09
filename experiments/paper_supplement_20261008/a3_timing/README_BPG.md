# Additional native-resolution BPG timing

The independent BPG entry is `benchmark_bpg.py`. It does not modify the already prepared/running four neural executables or request. Its metadata-only preparation command is `python experiments/paper_supplement_20261008/a3_timing/prepare_bpg.py`.

Before preparation, the actual in-memory BPG library passed complete byte and decoded uint8 pixel equality against the original CLI on 20 calibration sources at QP 29 and 51 (40 cases, zero PHY packets). The BPG request and calibration-only MCS freeze are the already completed native256 baseline, not the newly screened adaptive-resolution variant. The seven descriptors bind the real policy, original request, existing PHY wrapper, memory wrapper, library build receipt, actual parity completion and endpoint implementation. Source coding still performs fresh QP 51 feasibility, binary search and ±2 boundary checks on every TX. Its small within-call search memo is not a cross-case bitstream cache.

The protocol uses the same fixed 16 development sources, 7/13/19 dB, one warmup and three measured attempts as the neural methods. The code includes actual complete-frame FEC/modulation and received-only PHY/BPG decoding. CPU input layout conversion, PNG encoding in memory, fresh compression, all required transfers and final RGB conversion are inside their respective total windows. Model loading and all disk I/O remain outside. This classical codec has zero learned parameters; actual shared-library bytes are reported separately.

`SOURCE_UNFIT` means no complete container fit the frozen paid frame. Such an attempt retains its actual source-coding cost and elapsed source-attempt window. Complete-frame TX, RX, channel and software link E2E are missing (`null`/blank), not zero, because no frame exists. A separate gray fallback may be saved for display and is explicitly tagged as having no received frame. When a frame exists, header/body/parser or codec failures preserve the original gray behavior and their actual RX/E2E timing. Summaries show 48 attempted measurements per SNR, available counts for each metric, source-unfit counts and gray counts.

As in the neural run, E2E is a genuine outer window containing TX, channel and RX. The actual channel window is retained and subtracted separately. No air-interface latency or independent second-pass measurement is claimed. Nested timings cannot be added to totals. P95 is an empirical value from this limited set of repetitions, not a population tail guarantee.

This request adds at most 192 fresh source attempts and 384 paid packet decodes; source-unfit attempts call no PHY and header failures may skip the body. It does not rerun qualification, alter the old scientific ledger, change policy, train, score quality, or access holdout images. The parent coordinator runs it serially after the existing four-method process sequence, in the same exclusive CPU/GPU timing window:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 4-9 nice -n 15 \
 /home/liulu/projects/VAR_COMM/outputs/UNIFIED-METRICS-20261002/environment/bin/python \
 /home/liulu/projects/VAR_COMM/outputs/PAPER-SUPPLEMENT-20261008/a3_timing_v1/runtime/benchmark_bpg.py \
 --request /home/liulu/projects/VAR_COMM/outputs/PAPER-SUPPLEMENT-20261008/a3_timing_v1/materials_bpg/request.json \
 --deadline-unix CURRENT_AUTHORIZED_DEADLINE
```

The prepared upload manifest names only the new runner, new endpoint adapter and separate BPG request. Existing neural bytes and request hashes are checked unchanged. Output is `.../a3_timing_v1/results/BPG_LDPC_native256/`, with per-attempt immutable JSON, CSV summaries, component details, storage, actual packet events and a hashed completion receipt only on success. Failure preserves its actual attempted ledger and cannot silently overwrite/retry the directory.

For A5 display reuse, the final float RGB from measured repetition 1 is saved after timing for every source and SNR as `rgb_sourceNNNN_snrS_rep1.npz`, key `rgb`, CHW float32. The per-case record pins archive and RGB hashes, original source ID, actual work point, noise namespace/seed and whether a real received frame exists. It is the first predetermined repetition, never a per-method best-noise selection. These new development displays do not replace the published holdout results.
