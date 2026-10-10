# A3 unified single-output timing

All five requested methods have actual completed receipts.

Each method uses the same 16 pre-existing development sources at 7/13/19 dB, with one warmup and three measured repetitions per source/work point. Summaries therefore contain 48 measured attempts per SNR. Mean, median and empirical P95 describe these repetitions, not independent noise trials or a population tail guarantee.

The software E2E values come from an actual synchronized outer window. The separately measured software channel window is reported both included and subtracted. TX+RX is not substituted for E2E; unclassified overhead remains explicit. This is one pass, not a separately repeated E2E run, and is not air-interface latency. Nested components are inclusive and cannot be summed with their parent totals.

All neural/source-codec work is fresh. Model loading, filesystem I/O, quality scoring and verification are outside timing. Required device copies are inside the endpoint totals. Header/decoder failures are retained. BPG SOURCE_UNFIT attempts have a real source-coding cost, but no transmitted frame, RX or link E2E: these fields are unavailable, not zero. Tables give each metric its available-sample count.

BPG TX/RX/link-E2E averages are conditional on an actual transmitted frame. At 7/13/19 dB these comprise only 3/12/30 measured samples (1/4/10 distinct sources, each repeated three times). They are not full-population deployment averages; source-unfit encoding-attempt costs are separately retained in the CSV.

Memory is the maximum measured PyTorch peak allocated/reserved memory per SNR, including active loaded model baselines. Metric networks are on CPU; P also offloads its unused VAR. Full VAE modules remain loaded and include unused decoder parameters. Parameter counts are unique objects; checkpoint file size can include training metadata and is separate from tensor bytes. This is not a minimal deployment-size or energy measurement. Classical BPG has no learned weights; codec library bytes are separate.

SwinJSCC-80k is the frozen adapted model; 19 dB is outside its training and calibration range. RAW uses frozen development noise seed 6201; P/Swin use their original deterministic noise seed 2001. BPG uses its documented A3 development namespace and seed 2001. No method chooses a favorable noise realization.

All receipt-listed output hashes and case grids were checked before export. Published holdout quality results were neither recomputed nor modified. `timing_table.csv` uses milliseconds and MiB/GiB as labelled. `all_summary_rows.csv` preserves original seconds/bytes values and summary precision; `model_storage.csv` retains byte counts and scope notes.

`timing_single_output.csv` is an identical content alias of `timing_table.csv`, matching the experiment-plan filename. It adds no measurements or statistics.

| Method | SNR (dB) | TX mean (ms) | RX mean (ms) | Software E2E excl. channel mean (ms) | Gray / 48 | Source unfit / 48 |
|---|---:|---:|---:|---:|---:|---:|
| Partial-scale digital + VAR | 7 | 20.66 | 172.58 | 193.27 | 0 | 0 |
| Partial-scale digital + VAR | 13 | 20.94 | 174.09 | 195.05 | 0 | 0 |
| Partial-scale digital + VAR | 19 | 71.71 | 188.40 | 260.15 | 0 | 0 |
| Full-scale digital + VAR | 7 | 20.57 | 172.26 | 192.86 | 0 | 0 |
| Full-scale digital + VAR | 13 | 78.51 | 187.99 | 266.54 | 0 | 0 |
| Full-scale digital + VAR | 19 | 58.95 | 192.30 | 251.27 | 0 | 0 |
| Latent continuous JSCC | 7 | 8.66 | 12.79 | 21.48 | 0 | 0 |
| Latent continuous JSCC | 13 | 8.64 | 12.77 | 21.44 | 0 | 0 |
| Latent continuous JSCC | 19 | 8.64 | 12.77 | 21.43 | 0 | 0 |
| SwinJSCC-80k (adapted) | 7 | 15.18 | 11.90 | 27.10 | 0 | 0 |
| SwinJSCC-80k (adapted) | 13 | 15.18 | 11.77 | 26.97 | 0 | 0 |
| SwinJSCC-80k (adapted) | 19 | 15.14 | 11.74 | 26.90 | 0 | 0 |
| BPG + LDPC (native 256) | 7 | 201.60 | 15.62 | 217.24 | 45 | 45 |
| BPG + LDPC (native 256) | 13 | 221.63 | 15.23 | 236.89 | 36 | 36 |
| BPG + LDPC (native 256) | 19 | 253.31 | 17.24 | 270.58 | 18 | 18 |

Runtime: NVIDIA GeForce RTX 4090 D, Torch 2.11.0+cu128, CPU threads 6, interop threads 2.

Reproduce this read-only report with:

```text
python experiments/paper_supplement_20261008/a3_timing/summarize.py --source "C:\Users\11946\Documents\ChatGPT\comm\results\paper_supplement_20261008\a3_timing\actual_v1\results" --out NEW_EMPTY_OUTPUT_DIRECTORY
```
