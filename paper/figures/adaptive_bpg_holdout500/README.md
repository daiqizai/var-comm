# Adaptive BPG supplementary holdout figures

Original main result commit: `252176e041758ecb2d3e81b6fde5b587e7e17bb7`. New adaptive-BPG statistics and unchanged old-method estimates are read from `/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1`. This is a later-added baseline comparison, not a rewritten preregistration. Both BPG settings retain calibration-only selection; adaptive source resolution is explicitly part of its codec.

Inputs are completion-bound `comparison_summary.csv` and `paired.csv`. Main rows use500 sources×3 noises; per-source noise means and existing95% intervals are used at CSV precision. The plotting code does not train, infer, simulate a channel, recompute old statistics, bootstrap, or choose policy. Failure outcomes remain included. The paired contrast is adaptive−native, using paired intervals directly. Pairing by source is not a claim of shared noisy observations.

| Public label | Internal point template |
|---|---|
| Proposed (raw + partial + VAR) | `RAW64_PARTIAL_VAR_COMPLETION_SNR_{snr}` |
| Latent continuous JSCC | `P1024_SNR_{snr}` |
| SwinJSCC-80k (adapted) | `SWIN80K_N1024_SNR_{snr}` |
| BPG + LDPC (native 256) | `BPG_LDPC_N1024_SNR_{snr}` |
| Adaptive-resolution BPG + LDPC | `BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024_SNR_{snr}` |

The first three colors/styles/markers match `plot_paper_mainraw64.py`; the continuous system is labelled Latent continuous JSCC publicly. Native BPG is gray/dotted/diamond and adaptive BPG purple/custom-dashed/down-triangle. Swin19 dB is hollow with a dashed13–19 segment; it remains outside training/calibration. LPIPS is not flipped. ConvNeXt agreement is a percentage; its paired increment is percentage points, not accuracy or relative percent.

Each complete group has a seven-inch2×2 composite plus four separately laid-out single-column panels. PDF/SVG curves and text are vector (SVG text editable); PNG is600 dpi. Real SNR spacing and all finite intervals, zero/negative differences are retained. No spline, broken axis, or per-point number labels are used.

Reproduce to a fresh directory:

```text
python scripts/plot_paper_adaptive_bpg.py --data-dir "/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1" --output-dir NEW_EMPTY_OUTPUT_DIRECTORY
```

`--validate-only` checks real completed inputs without making images. `validation.json` records exact missing rows and stopped groups. `plot_data.csv` retains the precise original CSV strings and only the required factor100 display conversions. Existing paper figures and scientific result files are not overwritten. Before paper use, open both combined PNGs and check layout, labels and failure/zero points.
