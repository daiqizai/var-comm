# WCL Fig. 2, Fig. 3 and Table I — frozen-data redraw

All scientific values come from commit `a5fc2de0c0039689f2999ad0ed6a1c12c1487d06`. The live HEAD matched that commit at input capture. This directory is a new presentation version; historical figures and Fig. 1 (`system_diagram_v3_entropy.pptx`) are unchanged. No training, channel decoding, reconstruction, neural scoring, or bootstrap was performed.

## Files

- `figure2.pdf` / `figure2.svg`: 178 × 106 mm, five methods, 2 × 2 panels.
- `figure3.pdf` / `figure3.svg`: 178 × 62 mm, PSNR paired gain and source-capacity utilization.
- `figure2.png` / `figure3.png`: 600 dpi previews. The `_print_size.png` files are 96 dpi inspection copies of the same physical page; the `_grayscale.png` variants are for print checks.
- `table_i.tex`: booktabs table fragment, 8.5 pt text, mean and interval on separate lines. Include `\usepackage{booktabs}` in the manuscript preamble. The fragment fits the 88.5 mm column used in `table_i_preview.tex`; no scaling box is required.
- `captions.md`: English captions for the two figures and the table.
- `draw.py` and `style.json`: reproducible rendering and method/style mapping.
- `input_manifest.json` and `inputs/`: exact byte copies of the five specified CSVs, with their repository paths and SHA256 hashes.
- `render_audit.json`: every plotted mean and interval, the six table metric cells at original precision, axis limits, and display conversions.
- `review/`: actual PDF renderings, compiled table preview, export checks and visual review.

## Reproduce

Run `python -B draw.py` in an existing environment with NumPy, Matplotlib and Pillow. Use `--output /path/to/new/output` to keep the reviewed exports intact. The script verifies all five input hashes before rendering. Compile the optional table preview with `pdflatex -interaction=nonstopmode -halt-on-error table_i_preview.tex` from the output directory. No font files are distributed.

The reviewed rendering used the existing frozen CPU environment (Matplotlib 3.10.8 / NumPy 2.2.6). Text remains editable in SVG; PDF contains vector paths and embedded subset text, with no image XObjects. Pages retain their exact physical size without tight cropping. SVG trailing whitespace is normalized before hashing and inspection; numeric coordinates and text are unchanged.

## Data and interpretation

Fig. 2 reads `display_mean`, `display_ci_low` and `display_ci_high` directly. Agreement is already a percentage and is never multiplied again. All 60 means and their intervals use real 4/10/19 dB positions, with no horizontal offsets, smoothing or significance markers. The short labels are mapped in `style.json`; method identities, frozen checkpoint choices and T1-C 19 dB values remain unchanged.

Fig. 3(a) selects only `common500 / PARTIAL_minus_WHOLE / psnr_db` from the repaired paired CSV. It does not subtract marginal intervals. Fig. 3(b) uses only `mean_utilization × 100`, with WHOLE and PARTIAL kept separate. Its p05/p95 fields are not used as confidence intervals. The gray bars are hatched and the blue bars solid for grayscale distinction.

Table I selects adaptive rows at 4/10 dB and full-representation rows at 19 dB. All differences retain the CSV's NeST-minus-ARPC direction. In particular the 19 dB DISTS interval is strictly positive and favors ARPC. The three conditions are not presented as one configuration curve. Original precision is retained in the input copies and audit; only table entries and bar labels are rounded.

## Design references actually read

Installed figures4papers skill: `C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md`, plus `references/design-theory.md`, `references/common-patterns.md` and `references/demos.md` in that directory. Cached examples read: `.research/figures_redesign_v1/references/figure_ophthal_review__plot_trend.py` and `.research/figures_redesign_v2/references/figure_VIGIL__plot_posttraining.py`.

Applied principles: restrained blue/green/red/gray/purple semantic palette, clean spines, consistent panel geometry, a detached shared legend, marker/line redundancy and hatched comparison bars. The examples' oversized canvases, large type, categorical trend positions and opacity ramps were not carried over: this task uses the specified millimeter dimensions, numerical SNR positions and fully visible baselines.
