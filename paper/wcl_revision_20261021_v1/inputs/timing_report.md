# Current five-method fixed16 N1024 timing

One initial receiver import failed before any case or model construction. One BPG encoder attempt later stalled and produced no completed case. Both failures are retained. Explicit library-path and child-only CPU-count continuations reused completed stages without replaying successful calls. The unchanged x265 binary now detects the allocated six CPUs; the 256px qualification reports six pool workers and two frame threads.

Each method/SNR uses the same 16 development sources, batch size 1, three fresh warmups and three measurements per source. All 48 measurements per method/SNR are retained, including rejected/gray outcomes.

TX and RX are independently measured in separate processes on the same registered CPU/GPU pair. Total software processing is their sum; model loading, AWGN synthesis, cross-process handoff and audit/result persistence are outside this total. BPG subprocess and necessary codec file conversion costs are included. This is not radio airtime or an end-to-end RPC timer.

Original per-method Python/framework stacks are retained and disclosed in compute_and_resources.json. These values are not merged with the historical H800 six-method experiment, which predates m10 and excludes BPG and ARPC. Shared load is sampled, not claimed absent.

| Method | SNR (dB) | TX (ms/image) | RX (ms/image) | Total (ms/image) |
|---|---:|---:|---:|---:|
| NeST-Com (entropy-coded, m10 corrected) | 4 | 185.50 | 358.71 | 544.21 |
| NeST-Com (entropy-coded, m10 corrected) | 10 | 220.89 | 398.82 | 619.71 |
| NeST-Com (entropy-coded, m10 corrected) | 19 | 302.35 | 472.27 | 774.62 |
| Latent JSCC (100k) | 4 | 16.15 | 24.50 | 40.65 |
| Latent JSCC (100k) | 10 | 17.97 | 24.51 | 42.48 |
| Latent JSCC (100k) | 19 | 16.99 | 24.43 | 41.42 |
| SwinJSCC (adapted, 200k) | 4 | 24.02 | 18.59 | 42.61 |
| SwinJSCC (adapted, 200k) | 10 | 24.81 | 19.89 | 44.69 |
| SwinJSCC (adapted, 200k) | 19 | 24.90 | 18.43 | 43.32 |
| Adaptive BPG + LDPC | 4 | 2285.93 | 18.50 | 2304.43 |
| Adaptive BPG + LDPC | 10 | 2269.87 | 23.15 | 2293.01 |
| Adaptive BPG + LDPC | 19 | 2269.24 | 27.26 | 2296.50 |
| ARPC + LDPC-QAM (adapted) | 4 | 716.44 | 719.05 | 1435.49 |
| ARPC + LDPC-QAM (adapted) | 10 | 719.12 | 801.18 | 1520.30 |
| ARPC + LDPC-QAM (adapted) | 19 | 718.19 | 868.69 | 1586.89 |

The table shows arithmetic means. Unrounded per-call values, medians and p95 values are supplied as CSV. No quality metrics or bootstrap were recomputed. Output hash matches do not reduce the actual measured-call count.
