# Six-method shared-environment latency

Shared-environment latency was measured on the same NVIDIA H800 with batch size one, the fixed 16 development sources, and SNRs 4, 10, and 19 dB. Each method/source/SNR condition used three fresh warmups followed by three fresh measured repetitions. All 1,728 calls completed, including 864 measurements; all receiver outcomes were retained. Repeated transmitted waveforms, received observations and reconstructions matched within every condition. TX includes fresh source encoding and all actual entropy fallback attempts where applicable. RX includes the actual receive path and reconstruction. Software E2E excludes separately reported software noise-generation time and explicitly measured pure-provenance audit costs. Raw wall and excluded durations are reported separately. These measurements reflect shared resident workloads and fixed method order; they do not establish uncontended latency or a cross-hardware speedup. No latency confidence interval was bootstrapped. The complete grid combines 576 unchanged successful calls from the stopped v5 attempt and 1,152 successful calls from the reviewed v6 recovery. The original failed first entropy warmup is archived and is not counted as a completed warmup. It adds one known fresh visual-encoder attempt: the two attempts have 1,729 known fresh visual attempts/endpoint entries for 1,728 completed end-to-end calls. The shared GPU/software match, but resident workload conditions can differ between segments.

| Method | SNR | TX (ms) | RX (ms) | E2E (ms) | Channel (ms) |
|---|---:|---:|---:|---:|---:|
| Whole-scale + VAR (raw) | 4 | 59.708 | 573.399 | 633.176 | 0.193 |
| Whole-scale + VAR (raw) | 10 | 104.967 | 643.206 | 748.241 | 0.180 |
| Whole-scale + VAR (raw) | 19 | 136.290 | 571.628 | 707.984 | 0.173 |
| NeST-Com (raw) | 4 | 35.690 | 388.409 | 424.158 | 0.159 |
| NeST-Com (raw) | 10 | 47.923 | 391.715 | 439.693 | 0.156 |
| NeST-Com (raw) | 19 | 105.755 | 349.499 | 455.314 | 0.158 |
| Whole-scale + VAR (entropy-coded) | 4 | 297.331 | 353.284 | 650.661 | 0.116 |
| Whole-scale + VAR (entropy-coded) | 10 | 555.706 | 391.767 | 947.523 | 0.123 |
| Whole-scale + VAR (entropy-coded) | 19 | 243.002 | 468.737 | 711.784 | 0.121 |
| NeST-Com (entropy-coded) | 4 | 777.820 | 379.722 | 1157.590 | 0.124 |
| NeST-Com (entropy-coded) | 10 | 1007.165 | 423.195 | 1430.408 | 0.122 |
| NeST-Com (entropy-coded) | 19 | 938.258 | 503.974 | 1442.280 | 0.120 |
| Latent-space JSCC (100k) | 4 | 18.103 | 23.953 | 42.104 | 0.222 |
| Latent-space JSCC (100k) | 10 | 16.003 | 21.815 | 37.864 | 0.218 |
| Latent-space JSCC (100k) | 19 | 15.586 | 22.620 | 38.253 | 0.219 |
| SwinJSCC (adapted, 200k) | 4 | 26.926 | 20.626 | 47.599 | 0.156 |
| SwinJSCC (adapted, 200k) | 10 | 28.404 | 22.162 | 50.615 | 0.159 |
| SwinJSCC (adapted, 200k) | 19 | 28.083 | 21.943 | 50.074 | 0.158 |

The complete CSV additionally reports medians and p95, raw wall durations, provenance exclusions and all receiver outcomes.

- Shared resident workloads were present. These are shared-environment latency measurements, not uncontended latency.
- Resource observations occur at case boundaries, not continuously inside each timed window.
- Methods execute in a fixed sequential order; shared load and temporal drift can confound cross-method speed comparisons.
- The three repetitions reuse the registered noise realization with fresh computation; they are timing repetitions, not independent quality-noise trials.
- Qualification establishes within-run repeatability and source/model/code bindings; no new cross-host numerical-equivalence claim is made.
- Software timing excludes model construction, disk persistence, evaluation networks and radio airtime. Energy consumption was not measured.
- No same-hardware HiFi or BPG timing is supplied. No ratio to historical 4090D measurements is reported.
- Parameter counts describe loaded components. P communication parameters alone are not the total system parameter count.
- The worker receipt has no start tick; postprocessing conservatively requires its PID to be absent, even if that PID might have been reused.
