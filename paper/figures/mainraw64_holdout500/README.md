# Frozen holdout500 paper figures

Data commit: `252176e041758ecb2d3e81b6fde5b587e7e17bb7`. Source directory: `results/main_raw64_20261007/final_common500_r6/`.
All seven input files are SHA256-pinned in the plotting script and unchanged after export.
No development, MAIN16, H, C-REAL, BPG, HiFi or other bandwidth results are used.

Run from the actual VAR_COMM repository root (existing environment):
```sh
outputs/UNIFIED-METRICS-20261002/environment/bin/python -B scripts/plot_paper_mainraw64.py
```
An ordinary Python environment with matplotlib and NumPy can run the same script.
Use `--data-dir PATH --output-dir PATH` for a byte-identical local data mirror.

## Identity and units

| Method | Exact point template | Color / style / marker |
| --- | --- | --- |
| Proposed (raw + partial + VAR) | RAW64_PARTIAL_VAR_COMPLETION_SNR_{snr} | blue / solid / circle |
| P1024 | P1024_SNR_{snr} | vermillion / dashed / square |
| SwinJSCC-80k (adapted) | SWIN80K_N1024_SNR_{snr} | green / dash-dot / triangle |

SNR is numeric: 1,4,7,10,13,19. Swin's 19 dB marker is hollow and 13--19 dB is dashed.
DINOv2-L uses `dinov2_vitl14_cosine`, never `dino_cosine`.
ConvNeXt uses `convnext_top1_source_prediction`, agreement with the source prediction, not label accuracy.
ConvNeXt means AND CI endpoints are scaled by exactly100: percent for Fig.2, percentage points for Figs.3--4.
LPIPS differences retain their original sign; negative is better.

Fig.3 reads PARTIAL_VAR_COMPLETION minus WHOLE_VAR_COMPLETION from paired.csv.
Fig.4 reads PARTIAL_VAR_COMPLETION minus PARTIAL_DIRECT_DC from paired.csv.
Both comparisons use the matching SNR suffix. Their increments are never added.
The exact zero at7dB and small/zero values at13dB are included.

## Files and precision

15 figures (3 composites +12 individually laid-out single panels), each in PDF, SVG and600dpi PNG.
Composites are7 by4.9in; single panels are3.45in wide with their own legend/margins.
PDF uses vector paths and embedded TrueType fonts; SVG retains editable text.
Y limits cover all plotted CI endpoints with padding; delta panels include zero.
No smoothing, fitted curves, broken axes or per-point number labels are used.

`plot_data.csv` preserves original CSV `mean`, `ci_low`, `ci_high` text precision.
The `plot_*` columns record the exact Decimal unit conversion used for display;
`source_csv` and `source_row` identify the original row. Lines only connect measured SNR points.
Existing500-source/three-noise means and pointwise95% intervals are read, never recomputed.
No training, inference, channel simulation, bootstrap or strategy selection is invoked.

`captions.tex` defines three English caption commands. `validation.json` records checks and output SHAs.
Missing or duplicate rows stop their affected figure group and are listed precisely; no measurements are requested.
The publication filters are unchanged; files are generated in the working tree without committing.

| Composite | Export |
| --- | --- |
| fig02_main_N1024 | [PDF](fig02_main_N1024.pdf) |
| fig02_main_N1024 | [SVG](fig02_main_N1024.svg) |
| fig02_main_N1024 | [PNG](fig02_main_N1024.png) |
| fig03_partial_vs_whole | [PDF](fig03_partial_vs_whole.pdf) |
| fig03_partial_vs_whole | [SVG](fig03_partial_vs_whole.svg) |
| fig03_partial_vs_whole | [PNG](fig03_partial_vs_whole.png) |
| fig04_var_vs_direct | [PDF](fig04_var_vs_direct.pdf) |
| fig04_var_vs_direct | [SVG](fig04_var_vs_direct.svg) |
| fig04_var_vs_direct | [PNG](fig04_var_vs_direct.png) |
