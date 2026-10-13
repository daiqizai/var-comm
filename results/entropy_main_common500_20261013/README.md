# Original common500 entropy-coded partial-scale results

E1 quality and E2 source analysis are complete. All 4,500 registered logical
method frames at N1024 and SNR 4/10/19 dB have actual source, PHY, independent
reconstruction and scoring exit-zero receipts. Five figure groups were
exported. All five PNGs were actually opened and visually reviewed with PASS,
and all 20 figure-output hashes were verified; the separate
[visual-review receipt](visual_review_v1.json) records that check. E3 timing has no
valid result after the v2 and v3 construction-stage failures. This directory does
not claim complete E1–E3 delivery.

## Measured quality and interpretation

The new method is **NeST-Com (entropy-coded)**: frozen conditional entropy
coding with permitted partial scales and VAR completion. Source membership,
pixels, models, calibration-selected policy and three original T1 noises
remain fixed. This is a subsequent evaluation on the already used original
common500 population, not a first blind test. New100 and Kodak are separate.
Numbers below are presentation-rounded from CSV values at their original
precision; authoritative means and intervals are in the linked CSVs.

| SNR (dB) | PSNR (dB) | LPIPS-Alex | DINOv2-L cosine | ConvNeXt agreement (%) |
|---|---:|---:|---:|---:|
| 4 | 17.448 | 0.2633 | 0.5718 | 65.8 |
| 10 | 20.014 | 0.1715 | 0.7465 | 80.4 |
| 19 | 22.160 | 0.1182 | 0.8234 | 82.4 |


The direct partial-minus-whole entropy PSNR gains and 95% paired intervals are
0.616 [0.564, 0.673], 0.705 [0.639, 0.770], 0.337 [0.305, 0.369] dB at 4/10/19 dB. LPIPS and DINOv2-L improve at all
three points with paired intervals on the improving side of zero. ConvNeXt
source-prediction agreement improves at 4/10 dB but changes by -0.2 percentage
points at 19 dB, with 95% CI [-2.0, 1.6]. Agreement is not classification accuracy.
The entropy-minus-raw-partial PSNR gains are 0.281/0.299/0.595 dB; its 19 dB agreement
difference is also -0.2 percentage points, CI [-2.2, 1.8]. Raw and entropy
interfaces also differ in framing and failure rules, so this is not a pure
orthogonal two-factor ablation.

The tradeoffs against strong references remain explicit. Entropy partial has
lower PSNR than adapted Swin200k and adaptive BPG+LDPC at all three SNRs,
and lower PSNR than latent-space JSCC at 4/10 dB. At 4 dB its LPIPS is worse
than latent JSCC by 0.044469
(CI [0.041268, 0.047645]); at 10 dB the LPIPS
contrast is -0.001791
(CI [-0.003857, 0.000251]), including zero.
DINOv2-L and prediction agreement are higher than each of these three strong
references at all measured points. These results support a measured
quality tradeoff, not dominance across every metric or a claim of superiority
over all generative receivers.

## Actual source lengths and fallback

| SNR (dB) | Actual bits, mean | Actual tokens, mean | Capacity utilization (%) | Target not fitting (%) | Final K=0 sources /500 |
|---|---:|---:|---:|---:|---:|
| 4 | 1158.898 | 115.836 | 93.08 | 99.6 | 3 |
| 10 | 2399.242 | 244.594 | 95.21 | 99.6 | 190 |
| 19 | 4495.872 | 487.488 | 94.63 | 95.6 | 126 |


All 500 sources find a legal terminated stream at each SNR. A target that does
not fit triggers the original deterministic length-based fallback; it is not
a channel failure. Whole-scale fallback is permitted and actually occurs,
including 190/500 sources at 10 dB. Actual arithmetic bits include termination;
capacity is k−29, accounting for 13 length bits and 16 CRC bits. Mean per-source
same-token source-length saving, 1−b/(12T), is
16.16/17.77/22.53%.
This is source representation saving, not a reduction in the fixed 1024
wireless-symbol budget. Length analysis counts 500 sources per SNR, not 1500
noise realizations. All 4,500 realized frames have `PAYLOAD_PARSED` and
`ARITHMETIC_SOURCE_DECODED`; no rejected outcome was dropped from the rules.

## Data, statistics and figures

- [New method summary](analysis_v1/summary.csv): 12 mean/interval rows.
- [All seven method summaries](analysis_v1/summary_all_methods.csv): 84 rows,
  including raw whole for the mechanism table and 72 unchanged old intervals.
- [Five direct source-paired contrasts](analysis_v1/paired.csv): 60 rows,
  always new entropy partial minus the named reference.
- [All 42,000 source-metric means](analysis_v1/source_means.csv) and
  [30,000 source-paired differences](analysis_v1/source_paired_differences.csv).
- [48-row four-arm mechanism table](analysis_v1/mechanism_four_methods.csv),
  [actual 4500 frame rows](frame_rows.csv), and
  [failure breakdown](analysis_v1/failure_breakdown.csv).
- [Actual source streams and attempts](source_streams.csv),
  [source length summary](analysis_v1/length_summary.csv),
  [3,000 source/SNR/method length records](analysis_v1/length_utilization.csv),
  and [actual prefix distribution](analysis_v1/prefix_distribution.csv).
- [Short English manuscript fragment](manuscript_results.tex).

Each source's three noises are averaged before the original source-level
bootstrap, 10,000 replicates and seed 2026100701. Only 12 marginal and 60 paired
intervals were newly computed. Old intervals are copied unchanged; paired
intervals use source difference vectors, not marginal interval subtraction.
Intervals are pointwise. LPIPS is not sign-flipped. Agreement means and
intervals use percent; agreement differences use percentage points.

Five plot groups exist in [figures_v1](figures_v1), each as vector PDF,
editable SVG and 600 dpi PNG:

1. [Main five-method comparison](figures_v1/fig_main_common500_entropy.pdf).
2. [Four systems with the shared prior](figures_v1/fig_same_prior_four_methods.pdf).
3. [Entropy partial-minus-whole paired changes](figures_v1/fig_entropy_partial_minus_whole.pdf).
4. [Actual prefix endpoint distributions](figures_v1/fig_prefix_endpoint_distribution.pdf).
5. [Actual source lengths and capacity utilization](figures_v1/fig_source_lengths_and_utilization.pdf).

[Plot data](figures_v1/plot_data.csv) and [English captions](figures_v1/captions.tex)
are supplied. No further model calls or bootstrap are needed to redraw these
completed tables. Actual PNG appearance has passed the separate recorded
visual review, including fonts, axes, intervals, units and negative results.

## Provenance and unfinished work

[Scope and frozen protocol](../../experiments/entropy-main-common500-20261013/PROTOCOL.md),
[E0 reuse audit](e0_reuse/audit.json),
[actual scoring closure](scoring_completion.json),
[final read-only receiver audit](receiver_readonly_audit_v1/final_004500.json),
[statistical closure](analysis_v1/completion.json), and
[actual statistics/figure child waits](analysis_followthrough_v2/completion.json)
record the full chain. Independent receiver reconstruction uses actual received
fields and independently generated CDFs. Source truth is used only in a
post-decode equality audit. All 1,500 distinct reconstruction archives and the
full 4,500 logical references were read-only checked; 3,000 logical outputs
reuse identical received inputs. No new tokenizer or training call was made.

The shared-H800 timing v2 attempt actually exited1 during endpoint/model
construction because of a Python module-import path collision, before any
timing case or packet call. No latency values or speed ratios are available.
The failed scene and actual wait are preserved. The independent v3 recovery
entry also actually exited with code 1 during construction: the runtime
accessed `s.native.args.multiple_snr`, but the Swin field is
`s.native.multiple_snr`. It executed zero timing cases and zero packet calls.
Both failure scenes and actual waits are retained. 

Registration snapshot recorded at 2026-10-13T09:59:07+00:00 (not a live process-status assertion): v4 was retired during review before launch because its copied module globals would bypass the shared-resource metadata hook. The independent v5 recovery registration SHA is `44b14186a878621ea15e18da8cbf50e185927e6390826d019cff0865afc666f6`; four model-free CPU tests passed. At this snapshot v5 was in final review and had not launched. Original v2/v3 failure scenes remain intact. No timing cases, measured frames or digital packet calls had been consumed from the original 1,728/864/2,304 limits; numerical qualification and latency results were still pending.
 The eventual
shared-resource result must be labeled shared-environment latency, not
uncontended latency. The original fixed 16-source/six-method/1,728-call budget remains.
No HiFi or BPG timing is added.

The existing `var-comm` continuation monitor is ACTIVE for remaining authorized
verification, timing recovery, publication and backup. The legacy 15-minute
`var-raw64-holdout` monitor remains PAUSED. Actual PNG review is complete;
verified repository publication and local backup remain final-delivery steps
and are not claimed from file presence. E4, additional SNRs/bandwidths, entropy Kodak,
training and further model selection remain disabled.
