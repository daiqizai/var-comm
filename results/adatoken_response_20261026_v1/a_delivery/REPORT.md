# A: Controlled endpoint granularity — completed

Fixed external_common100_v1 sources, original indices, N=1024, 4/10/19 dB and three original noises. The model, tokenizer, decoder, within-scale raster order, common maximum m10 endpoint and MCS are fixed across the four nested endpoint sets. This is a separate controlled diagnostic; it does not overwrite the historical jointly calibrated WHOLE/PARTIAL comparison or choose a new main configuration.

All 3,600 logical results are present: 100 sources × 4 grids × 3 SNRs × 3 noises. Exact waveform/profile/counter/noise identity reduces this to 2,214 physical events and 4,428 packet decodes. All events in this fixed run were accepted; no failed event was removed. There are 738 unique actual receiver outputs. The independent qualification contributed 49 source decodes, of which six were exactly reused; the main RX added 732 source decodes and 738 renders. The remaining 100-source source-probability work used one traversal per source, no new tokenizer call, with all49 finished streams on the qualified same-device route. Historical first-source hash/byte-route and import failures remain preserved.

## 10 dB focus

| Grid | Candidates | Arithmetic utilization | Mean tokens | PSNR (dB) ↑ | LPIPS-Alex ↓ | DreamSim ↓ | CLIP-I ↑ | Length checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WHOLE | 7 | 78.15% | 200.00 | 18.951576 | 0.201821 | 0.131529 | 0.858758 | 3.55 |
| HALF | 13 | 89.21% | 227.50 | 19.395750 | 0.183760 | 0.114775 | 0.870527 | 5.55 |
| QUARTER | 25 | 95.38% | 243.77 | 19.611725 | 0.175170 | 0.107702 | 0.875937 | 9.49 |
| EIGHTH | 49 | 97.39% | 249.11 | 19.734671 | 0.172939 | 0.106972 | 0.874694 | 17.58 |

Quarter versus whole gives ΔPSNR = +0.660149 dB [0.515203, 0.813679], ΔLPIPS = −0.026651 [−0.032093, −0.021339], ΔDreamSim = −0.023826 [−0.029964, −0.017961], and ΔCLIP-I = +0.017179 [0.009866, 0.025160]. These are direct source-paired intervals.

Eighth versus quarter gives a smaller PSNR gain, +0.122946 dB [0.078557, 0.169317], and LPIPS reduction −0.002231 [−0.003228, −0.001310]. Its DreamSim difference −0.000730 [−0.001956, 0.000477] and CLIP-I difference −0.001244 [−0.004249, 0.001502] both span zero. Thus this sample supports improved budget use and pixel fidelity with finer endpoints, but does not establish a monotonic improvement in every perceptual or semantic metric. The quarter grid is not declared globally optimal. No strategy changes follow from this diagnostic.

Quarter-minus-whole PSNR gains at 4/10/19 dB are respectively +0.570932, +0.660149 and +0.300995 dB. All four metrics, all three SNRs and all six grid pairs remain in the CSVs. Length checks are comparisons against already finished streams, not additional probability-model passes or timing measurements.

## Metric reuse and statistics

The native arrays and source/model/preprocessing identities were verified. Existing B pairs cover 1,263 logical rows; only 468 new unique image pairs per metric were scored. Exact closed DreamSim/CLIP feature caches were reused, including source features; new feature counts are 469 DreamSim and 468 CLIP. Repeated logical occurrences retain their weights.

Every source first averages three noise results, then all100 sources receive equal weight. The existing 10,000×100 bank is reused in its explicit source-ID order. There are 48 means/pointwise 95% source-bootstrap intervals and 72 directly paired differences/intervals. No new bootstrap indices, marginal-CI subtraction or significance stars are used. This existing subset is not a new blind test.

Arithmetic utilization is the actual independently terminated source-stream length divided by the legal k−29 source capacity after the 13-bit length and CRC16 body fields. The common 68-symbol header and 956-symbol body remain paid. It is not total-airtime efficiency. The endpoint distribution lists transmitted endpoints; acceptance is reported separately.

## Outputs

- `summary.csv`, `paired_differences.csv`: all48 means and72 direct paired contrasts, full precision.
- `frame_metrics.csv`, `source_means.csv`: complete logical and source-level records.
- `efficiency.csv`, `transmitted_endpoint_distribution.csv`: real lengths, capacity and endpoint counts.
- `statistics_identity.json`, `completion.json`: exact statistics provenance.
- `../figures_a_v2/`: editable SVG, vector PDF, 600-dpi PNG, print/grayscale previews and actual visual review. The failed pre-export v1 layout check remains documented; it involved no new science.

### Captions

**All-SNR quality.** Controlled endpoint-granularity comparison on the fixed100-source subset at N=1024. Whole, half-, quarter- and eighth-scale endpoint sets share the same maximum endpoint, model, MCS and actual terminated-length fitting rule. Each source averages three fixed channel realizations. Error bars are pointwise95% source-bootstrap confidence intervals; lines connect measured categorical endpoint sets.

**10-dB increments.** Direct source-paired metric differences between successive nested endpoint sets at N=1024 and10dB. Positive PSNR/CLIP-I and negative LPIPS-Alex/DreamSim differences favor the finer set. Error bars are95% source-bootstrap intervals. Zero-crossing and negative outcomes are retained.

These figures supplement the controlled mechanism study and do not constitute the new Ada external comparison. Different tokenizers, priors and image decoders in that comparison preclude attributing its entire system gap to endpoint granularity.
