# B: Direct residual correction and subsequent propagation

Status: scientific outputs and source-paired statistics are closed. This secondary mechanism analysis is preserved while the predicted-class Ada comparison and controlled endpoint granularity receive first priority. It does not replace either external comparison or the existing WHOLE/PARTIAL table.

## Fixed scope and arms

The original external_common100_v1 subset, N=1024, 10/19 dB, and three existing noise realizations per source are retained. There are 600 original received events and 1,800 logical arm results. No new transmission, PHY decoding, tokenizer, training or policy selection occurred.

- A0: do not overwrite the received within-scale boundary tokens.
- B0: correct the final latent with the received boundary residual, while keeping subsequent VAR states on the uncorrected path.
- C0: predict, overwrite the actually received positions, and advance the corrected state to finer scales.

Only actual received tokens enter these receivers. Source truth is used for evaluation, never for correction. K=0 and rejected receptions retain their original outputs and denominator. The 135 unique positive-K receiver inputs produce 405 unique arm outputs. Historical cross-device pixel mismatches are preserved; all three arms use the explicitly registered same-device route, including the two completed qualification C0 outputs. The 270 A0/B0 and 133 additional C0 outputs were computed once. There were 331 unique new native image pairs per metric and 585 identical old logical scores reused; all counts refer to their stated layer, not to additional channel experiments.

## All-source paired results

Each source first averages its three original noise results. Differences are formed at source level before applying the existing 10,000×100 bank, whose explicit source IDs were aligned. Positive differences favor the first arm for PSNR/CLIP-I; negative differences favor it for LPIPS/DreamSim. Intervals are pointwise source-bootstrap 95% intervals. They are not differences between marginal confidence limits.

| SNR | Difference | PSNR (dB) | LPIPS-Alex | DreamSim | CLIP-I |
|---:|---|---|---|---|---|
| 10 | B0 − A0 | +0.219742 [+0.176135, +0.263992] | -0.010030 [-0.012026, -0.008026] | -0.008454 [-0.011357, -0.005643] | +0.005695 [+0.001547, +0.010296] |
| 10 | C0 − B0 | +0.440407 [+0.330421, +0.557623] | -0.016620 [-0.020494, -0.012862] | -0.015372 [-0.019880, -0.011113] | +0.011484 [+0.006107, +0.017790] |
| 10 | C0 − A0 | +0.660149 [+0.515203, +0.813679] | -0.026651 [-0.032093, -0.021339] | -0.023826 [-0.029964, -0.017961] | +0.017179 [+0.009866, +0.025160] |
| 19 | B0 − A0 | +0.300995 [+0.247005, +0.357939] | -0.005097 [-0.005972, -0.004263] | -0.002994 [-0.004506, -0.001694] | +0.004492 [+0.001606, +0.007740] |
| 19 | C0 − B0 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| 19 | C0 − A0 | +0.300995 [+0.247005, +0.357939] | -0.005097 [-0.005972, -0.004263] | -0.002994 [-0.004506, -0.001694] | +0.004492 [+0.001606, +0.007740] |

At 10 dB both the direct correction and the subsequent conditional propagation improve these four metrics in this fixed sample. At 19 dB, C0−B0 is exactly zero for every metric: the accepted positive-K inputs lie in the final scale, leaving no finer scale for conditional propagation. This boundary case is retained, not omitted. PSNR is nonlinear; these are empirical paired arm contrasts, not a theoretical additive decomposition of PSNR.

The complete CSV also reports the predeclared positive-K and final-scale positive-K diagnostics. These are post-reception subsets and have different eligible source counts. The all-source table keeps the complete 100-source denominator. The same existing evaluation subset has been used in prior work; this is not a new blind test or evidence of a general causal advantage of a representation order.

## Provenance

- `frame_metrics.csv`: every logical arm/noise result, actual output pixel hash and reuse origin.
- `source_means.csv`: source-level noise averages, including predeclared diagnostic subsets.
- `summary.csv` and `paired_differences.csv`: complete four-metric statistics; full precision retained.
- `statistics_identity.json`: exact bank/source order and subset aggregation.
- `completion.json`: immutable statistics manifest; this report is a later explanatory companion.

Original source/receiver failure records and subsequent explicit same-device amendment remain unchanged. No rescoring, rendering, or bootstrap draw was performed to produce this report.
