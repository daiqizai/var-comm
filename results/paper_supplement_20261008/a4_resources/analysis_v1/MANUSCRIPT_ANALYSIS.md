# A4 manuscript analysis and coverage

Data version: `252176e041758ecb2d3e81b6fde5b587e7e17bb7`; N=1024; the original 500 sources and three noises per SNR. This document reads the completed A4 v2 CSVs and the exact hash-pinned original paired.csv. It does not modify science files or create new estimates or confidence intervals. Numbers in the prose and tables are rounded for presentation; the cited CSVs retain full precision.

## Manuscript-ready text

The frozen partial-scale policy increases transmitted source content while retaining the same legal modulation and coding choices as the full-scale control. At 10 dB it carries 230 rather than 155 raw tokens (2760 versus 1860 source bits). Its selected 16QAM code has actual information/transmitted length 2776/3824 (rate 0.72594), whereas the full-scale control uses 64QAM with 1876/5628 (rate 1/3). Header/body/padding allocations are 68/956/0 and 68/938/18 complex symbols, respectively. The partial-scale system has higher all-frame quality despite 66 body-CRC rejections among 1500 frames, compared with none for the full-scale control. Thus the quality difference is accompanied by a reliability tradeoff and cannot be attributed solely to fewer channel errors.

At 19 dB both digital policies have zero observed header rejections, body-CRC rejections, wrong acceptances or token errors in their 1500 frames. The partial-scale policy transmits 397 tokens versus 255, with source information efficiencies of 4.65234 and 2.98828 bit per complex channel use. Its frame-use fraction is 1.00000, compared with 0.66797 for the selected full-scale policy. The latter has 340 known padding symbols; these symbols are energized QPSK and provide neither image information nor additional FEC. Both families had legal full-budget protection candidates, so the selected padding does not establish that the full-scale control was forced to waste resources or that either selection is theoretically optimal. The 19 dB PSNR, LPIPS and DINOv2-L changes favor partial-scale transmission, but its prediction-agreement difference is inconclusive at the reported 95% interval. Source-bit/quality curves vary in both SNR and selected policy and are descriptive, not a controlled causal allocation experiment.

KEEP separates CRC rejection from display failure. At 1 dB the partial-scale receiver rejects the body CRC in 75/1500 frames, yet none of these frames is gray. Their descriptive frame-weighted PSNR is 14.844 dB and DINOv2-L similarity is 0.37069, compared with all-frame values of 16.125 dB and 0.45514. At 10 dB the 66 rejected frames have PSNR 18.225 dB and similarity 0.62263. These subsets contain different sources and noise realizations; their means do not estimate the causal benefit of KEEP over DROP. The 7 dB rejected subset contains only one frame, whose 100% source-prediction agreement must not be generalized. No new subset or gain intervals are supplied.

Mean token error fractions at 1, 7 and 10 dB are 0.719635%, 0.003441% and 0.434783%, respectively, although their median and 95th percentile are all zero. The largest per-frame error fractions are 87.6712%, 5.1613% and 45.2174%. This sparse tail is hidden by the zero 95th percentile. At 1 dB the mean correctly received contiguous prefix is 70.311 tokens despite 73 planned and received tokens; at 10 dB it is 221.618 despite 230 received tokens. These are offline diagnostics against the source tokens, which are never supplied to the receiver. No RAW header rejection, wrong acceptance or gray output was observed; this finite sample does not imply zero population failure probability.

The result is not uniformly improved at every operating point or metric. At 7 dB the two policies coincide exactly and all paired increments are zero. At 13 dB the PSNR increment is only 0.06249 dB, and the prediction-agreement difference is 0.00 percentage points with a 95% interval of [-1.80, 1.80]. At 4 dB the agreement interval also includes zero. Prediction agreement refers to agreement with the source image's classifier prediction, not ground-truth classification accuracy or verified semantic correctness. The VAR-versus-direct comparison remains a separate same-reception ablation; its increments are not added to the partial-versus-full increments.

## Original paired increments (partial-scale + VAR minus full-scale + VAR)

Each entry is the original paired mean [95% interval]. LPIPS keeps its original sign; negative is favorable. Prediction agreement is converted to percentage points.

| SNR (dB) | PSNR (dB) | LPIPS | DINOv2-L | Agreement (pp) |
|---|---:|---:|---:|---:|
| 1 | 0.51367 [0.45713, 0.56953] | -0.02961 [-0.03192, -0.02724] | 0.04558 [0.03412, 0.05720] | 6.73 [2.87, 10.73] |
| 4 | 0.36047 [0.32393, 0.39945] | -0.01595 [-0.01716, -0.01476] | 0.02675 [0.01811, 0.03550] | 2.40 [-0.60, 5.40] |
| 7 | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] | 0.00 [0.00, 0.00] |
| 10 | 1.17613 [1.11504, 1.23781] | -0.04043 [-0.04236, -0.03856] | 0.07586 [0.06555, 0.08618] | 6.87 [3.73, 10.13] |
| 13 | 0.06249 [0.05397, 0.07144] | -0.00145 [-0.00174, -0.00117] | 0.00363 [0.00076, 0.00655] | 0.00 [-1.80, 1.80] |
| 19 | 1.34299 [1.29378, 1.39300] | -0.03475 [-0.03598, -0.03355] | 0.05206 [0.04558, 0.05873] | 1.00 [-1.80, 3.80] |

## Actual code and resource allocation

The effective rate is k/n including the 16 body-CRC bits; it is not the source payload rate and does not use the internal filler-extended k. Every raw header carries 12 configuration bits, 16 CRC bits and 6 tail bits, producing 136 transmitted bits over 68 QPSK complex symbols. The nominal rate label alone can differ substantially from the actual rate after full-budget allocation.

| SNR | Scheme | m, K | Tokens | Modulation | Nominal label | Actual k/n | Effective rate | Allocation category | Header/body/padding |
|---|---|---|---:|---|---|---|---:|---|---|
| 1 | Full-scale | 5, 0 | 55 | QPSK | 1/2 | 676/1912 | 0.353556 | full_budget | 68/956/0 |
| 1 | Partial-scale | 5, 18 | 73 | QPSK | 1/2 | 892/1912 | 0.466527 | full_budget | 68/956/0 |
| 4 | Full-scale | 6, 0 | 91 | 16QAM | 1/2 | 1108/3824 | 0.289749 | full_budget | 68/956/0 |
| 4 | Partial-scale | 6, 16 | 107 | QPSK | 5/6 | 1300/1912 | 0.679916 | full_budget | 68/956/0 |
| 7 | Full-scale | 7, 0 | 155 | 16QAM | 1/2 | 1876/3824 | 0.490586 | full_budget | 68/956/0 |
| 7 | Partial-scale | 7, 0 | 155 | 16QAM | 1/2 | 1876/3824 | 0.490586 | full_budget | 68/956/0 |
| 10 | Full-scale | 7, 0 | 155 | 64QAM | 1/3 | 1876/5628 | 0.333333 | nominal | 68/938/18 |
| 10 | Partial-scale | 7, 75 | 230 | 16QAM | 3/4 | 2776/3824 | 0.725941 | full_budget | 68/956/0 |
| 13 | Full-scale | 8, 0 | 255 | 64QAM | 2/3 | 3076/5736 | 0.536262 | full_budget | 68/956/0 |
| 13 | Partial-scale | 8, 9 | 264 | 16QAM | 5/6 | 3184/3824 | 0.832636 | full_budget / nominal | 68/956/0 |
| 19 | Full-scale | 8, 0 | 255 | 64QAM | 5/6 | 3076/3696 | 0.832251 | nominal | 68/616/340 |
| 19 | Partial-scale | 8, 142 | 397 | 64QAM | 5/6 | 4780/5736 | 0.833333 | full_budget / nominal | 68/956/0 |

Identical token counts are not merged across actions: the full-scale 7/10 dB policies both send 155 tokens but use different modulations and code lengths; its 13/19 dB policies both send 255 tokens but use 3076/5736 and 3076/3696. Candidate identity, wire identity and profile ID remain in the original table. `full_budget / nominal` means both construction labels identify the same legal action, not two transmissions.

Frame energy follows the saved actual waveform, not a forced per-frame energy of 2048. The mean and [5th, 95th] percentiles are:

| SNR | Scheme | Actual frame energy mean [P05, P95] |
|---|---|---:|
| 10 | Full-scale | 2048.448 [1985.524, 2110.857] |
| 10 | Partial-scale | 2048.706 [1995.200, 2107.280] |
| 19 | Full-scale | 2048.842 [2000.343, 2099.828] |
| 19 | Partial-scale | 2049.107 [1984.381, 2114.667] |

## Explicit diagnostic example

This example is chosen retrospectively as the first CRC-rejected 1 dB record in the existing source/noise order; it is not an independently selected qualitative success sample. Source `n04380533/ILSVRC2012_val_00019316_n04380533`, source index 1 and noise seed 6202, receives all 73 planned tokens at 1 dB, but 12 are wrong and only the first 5 form a correct contiguous prefix. The body CRC rejects, KEEP retains the hard tokens, and the output is not gray. For the same source and noise-seed identity at 19 dB, both frozen digital schemes accept correct bodies: full-scale receives 255 correct tokens and partial-scale 397. Their waveforms are different, so this is not a shared received observation across schemes or SNRs. No new image or channel simulation was produced for this explanation.

## A4 requirement coverage

| Requested item | Existing evidence / delivery | Status and limit |
|---|---|---|
| Six-SNR m/K, MCS, nominal and actual rate | `v2/resource_summary.csv` | Covered; 24 transmitter configurations, exact k/n and filler fields. |
| Source, CRC/control, coded bits and budget | `v2/resource_summary.csv` | Covered; actual transmitted n, header payload/CRC/tail, body CRC and 1024-use identity verified. |
| Nominal/full-budget permissions; full-scale opportunity | `v2/resource_summary.csv`, `v2/configuration_provenance.json` | Covered from frozen policy/catalogue evidence; does not claim exhaustive optimality. |
| Two separate efficiency definitions | `v2/resource_summary.csv` | Covered; frame-use fraction and raw source bits/use. Continuous symbol schemes are not assigned fictitious raw bits. |
| Actual energy distribution | `v2/resource_summary.csv` | Covered; mean, min, P05/P50/P95, max and standard deviation. |
| Token errors and true received prefix | `v2/token_error_summary.csv`, `v2/frame_diagnostics.csv` | Covered; mean/quantiles/max, absent-token diagnostic, actual correct prefix, not the transmitted plan. |
| Header/CRC/wrong acceptance/gray | `v2/failure_state_counts.csv` | Covered; zero-count states retained; KEEP rejected frames not mislabeled gray. |
| State-conditioned and full-frame quality | `v2/state_quality.csv`, `v2/published_quality_reused.csv` | Covered; 144 full-frame means validated, existing full-frame CIs reused, conditional means descriptive with source/frame counts. |
| Distinct actions at equal token count | Candidate/profile/wire fields in resource and frame tables | Covered; 7/10 and 13/19 examples above are not merged. |
| Budget, source-information and failure plots | `v2/figures_r2/` four PDF/SVG/600-dpi PNG groups | Generated and visually checked; SNR is numerical; no smoothing, erased failures or recomputed statistics. |
| Low-SNR and 19 dB explanation | Manuscript text and explicit diagnostic example above | Covered without new inference; negative/reliability tradeoffs and null intervals retained. |

The original v2 completion correctly binds 25 closed files; its console.log hash differs because the final log line was appended after hashing. This is a runtime-log issue, not a CSV discrepancy; the original completion is retained unchanged. The rendering revision verifies all 16 closed output hashes and all five input bindings. All four revised PNGs were actually opened: no clipped labels, legends or intervals were observed; overlapping point text is removed. The 7 dB one-frame conditional result remains visible and is explicitly limited in captions and subset_counts.csv.
