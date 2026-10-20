## 2026-10-20 Current five-method N1024 software timing

`results/five_method_timing_20261020_v1/delivery/` contains the current m10
NeST-Com, Latent JSCC100k, SwinJSCC200k, adaptive BPG+LDPC and ARPC+LDPC
TX/RX/software-total latency on the same PRO6000/CPU allocation. The original
fixed16 development sources, batch1, SNR4/10/19, three fresh warmups and three
formal calls per source/setting yield1440 completed calls and720 measurements.
All receiver outcomes remain in the denominator. Means, medians, p95 and
unrounded per-call ms/image, resource/software versions and receipt audit are
included. There are no new quality metrics, bootstrap, training or N2048 calls.

Two engineering failures are retained: a receiver library import before any
case/model construction, and one incomplete BPG encoder attempt that stalled
when legacy x265 created pools for180hostCPUs. The explicit continuations
reuse successful stages. BPG's unchanged binary now detects its actual6CPU
affinity through a child-only resource adapter; original presets/math/search
remain fixed. Original software stacks and this resource difference are
disclosed. These timings are separate from the historical H800 six-method set.
Source/registration/recovery entries are `experiments/five-method-timing-20261020/`.
The existing paper figures and open quality Table I are unchanged.

---

## Runtime and interpretation

All calls use one PRO6000 GPU and CPU affinity 0–5, with batch size 1. Torch intra-op/inter-op thread settings are 6/2. TF32 is disabled. Original method environments are preserved; sharing the device does not imply identical framework versions or internal precision.

| Method | Role | Python | Torch | CUDA | Setup outside timing (s) |
|---|---|---|---|---|---:|
| NeST-Com | tx | 3.10.12 | 2.11.0+cu128 | 12.8 | 21.568 |
| NeST-Com | phy | 3.11.15 | CPU codec | — | 16.324 |
| NeST-Com | rx | 3.10.12 | 2.11.0+cu128 | 12.8 | 16.216 |
| Latent JSCC | tx | 3.10.12 | 2.11.0+cu128 | 12.8 | 12.190 |
| Latent JSCC | rx | 3.10.12 | 2.11.0+cu128 | 12.8 | 10.083 |
| SwinJSCC | tx | 3.10.12 | 2.11.0+cu128 | 12.8 | 8.863 |
| SwinJSCC | rx | 3.10.12 | 2.11.0+cu128 | 12.8 | 6.233 |
| Adaptive BPG + LDPC | tx | 3.11.15 | CPU codec | — | 0.016 |
| Adaptive BPG + LDPC | phy | 3.11.15 | CPU codec | — | 28.397 |
| Adaptive BPG + LDPC | rx | 3.11.15 | CPU codec | — | 0.005 |
| ARPC + LDPC | tx | 3.12.3 | 2.10.0+cu130 | 13.0 | 57.688 |
| ARPC + LDPC | phy | 3.11.15 | CPU codec | — | 7.367 |
| ARPC + LDPC | rx | 3.12.3 | 2.10.0+cu130 | 13.0 | 27.846 |

ARPC retains FP32 VAE/prior parameters and the original mixed FP16/FP32 text encoder. Its native identity and precision configuration are in the execution review. BPG uses its unchanged codec binary with a child-only CPU-count adapter so legacy x265 observes the six allocated CPUs. Its full adaptive resolution/QP search and candidate decoding are timed.

## Outcomes retained in the latency denominator

| Method | SNR (dB) | All formal outcomes (48 calls) |
|---|---:|---|
| NeST-Com | 4 | ARITHMETIC_SOURCE_DECODED: 48 |
| NeST-Com | 10 | ARITHMETIC_SOURCE_DECODED: 48 |
| NeST-Com | 19 | ARITHMETIC_SOURCE_DECODED: 48 |
| Latent JSCC | 4 | CONTINUOUS_DECODED: 48 |
| Latent JSCC | 10 | CONTINUOUS_DECODED: 48 |
| Latent JSCC | 19 | CONTINUOUS_DECODED: 48 |
| SwinJSCC | 4 | SWIN_DECODED: 48 |
| SwinJSCC | 10 | SWIN_DECODED: 48 |
| SwinJSCC | 19 | SWIN_DECODED: 48 |
| Adaptive BPG + LDPC | 4 | BPG_DECODED: 48 |
| Adaptive BPG + LDPC | 10 | BPG_DECODED: 48 |
| Adaptive BPG + LDPC | 19 | BPG_DECODED: 48 |
| ARPC + LDPC | 4 | ARPC_DECODED: 48 |
| ARPC + LDPC | 10 | ARPC_DECODED: 48 |
| ARPC + LDPC | 19 | ARPC_DECODED: 48 |

A reject or gray result may return sooner than a successful reconstruction. These timings retain such outcomes and are software-processing measurements, not a new quality comparison. Repetitions share one fixed noise realization within each setting; method-specific historical noise namespaces differ.

## Observed shared load

| Method | Role | System load1 range | GPU free-memory minimum (MiB) |
|---|---|---:|---:|
| NeST-Com | tx | 16.18–23.20 | 94316 |
| NeST-Com | phy | 22.59–25.26 | 97250 |
| NeST-Com | rx | 20.03–31.45 | 94258 |
| Latent JSCC | tx | 26.46–31.43 | 94348 |
| Latent JSCC | rx | 29.64–32.15 | 94390 |
| SwinJSCC | tx | 26.02–29.50 | 96162 |
| SwinJSCC | rx | 26.02–28.70 | 96164 |
| Adaptive BPG + LDPC | tx | 22.04–34.25 | 97250 |
| Adaptive BPG + LDPC | phy | 20.38–25.31 | 97250 |
| Adaptive BPG + LDPC | rx | 22.19–28.06 | 97250 |
| ARPC + LDPC | tx | 22.46–29.01 | 76128 |
| ARPC + LDPC | phy | 21.77–24.70 | 97250 |
| ARPC + LDPC | rx | 20.88–28.00 | 72618 |

Load1 is host-wide runnable/uninterruptible workload, not a utilization percentage. Recorded snapshots disclose shared load and do not prove exclusive use. See the execution review for GPU utilization ranges, sample counts, actual exits and source/model identities.

The original PROTOCOL.md and registrations are immutable pre-run snapshots; actual status is established by the closed delivery and execution review. Reproduction must not overwrite or automatically rerun completed measurements.
