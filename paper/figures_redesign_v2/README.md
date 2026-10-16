# NeST-Com Figures 2 and 3 — redesign v2, feedback revision

This revision implements the user's title, symbol, zero-line and caption feedback. It contains only Figures 2 and 3 and their six individual panels. The original v2 delivery remains in Git commit `6897e7939688951bb665bdb26a2e598095aedb0b`.

| Figure | PDF / editable SVG / 600 dpi PNG | Exact page size |
|---|---|---|
|2: four-method performance|`fig2_main_1x4.*`|181.353259 × 55 mm|
|3: within-family paired gains|`fig3_paired_two_panels.*`|88.567870 × 50 mm|

`panels/` contains all six independent panels in the three formats. Typography is native, not a scaled crop. `review/actual_size_preview.pdf` places both figures at the current IEEEtran manuscript widths; print at 100% without fit-to-page. PDF and SVG contain vector paths and text, with no embedded raster plots. SVG text is editable. The actual font is DejaVu Sans, the available fallback; no font files are distributed. PNGs are rendered natively at 600 dpi.

## Changes in this revision

- Figure 2 panel (d) reads “Agreement (%) ↑”; all four titles are single-line.
- Blue main-method circles are 4.5 pt in both figures. The hollow red square retains its contour at the overlapping 10 dB LPIPS point; data positions are unchanged.
- Figure 3 uses filled dark-gray squares with white outlines, with intervals below the markers. Short intervals may be partly hidden by symbols; cap tips remain at the exact interval endpoints, without artificially enlarging their height.
- Figure 3 titles are “ΔPSNR (dB)” and “ΔDreamSim”. The plotting area is taller within the same 50 mm canvas. Its lower limit is zero and the detached dashed zero line is removed. Both metrics' minimum lower CI bounds were verified positive before this change; no value or interval is clipped.
- Captions are shortened. Agreement and both difference directions are defined positively. The m10 and family-specific evaluation settings are recorded once in `experiment_settings_note.md`.

## Data and interval provenance

Scientific input snapshot: `24fa1ecf3a6a959e6023d47ff460df41a711d30f`. The feedback revision started from `6897e79`, with no intervening science changes. All three selected input CSVs are byte-identical to the first v2 delivery; see `review/feedback_revision.json`.

Figure 2 uses common500, N1024, SNR 4/10/19 dB, and final P100k/Swin200k baselines. The entropy-coded 19 dB rows are the corrected T1-C evaluation. Figure 3 reads direct partial-minus-whole paired results within each representation family, including both corrected entropy families at 19 dB. DreamSim reduction negates the mean and exchanges the negated bounds. Marginal intervals are never subtracted.

Historical raw, supplemental entropy and T1-C DreamSim match implementation/file hashes, weights, ensemble, normalization, square 224 bicubic antialiased preprocessing and no added quantization. All 500 source archive and preprocessing identities match. GPU hardware differs; this is mathematical/input compatibility, not a claim of bitwise device identity. See `review/dreamsim_identity.json`. The newer supplement lacks raw-WHOLE; its genuine historical paired result is traced explicitly.

No new100, cal1000, Kodak or fixed16 data are mixed in. No scoring, resampling, reconstruction, channel execution or training was performed. Historical qualitative images and H800 timing are unchanged.

`plot_inputs/point_manifest.csv` maps all 60 displayed points to exact source CSV rows and full-precision values. `input_manifest.json` lists the source files. `review/numerical_validation.json` records the original exact-row validation. Only visual changes were made in this feedback revision.

## Reproduction and review

Run `python plot_figures.py` using an existing Matplotlib environment. Rendering reads only the included selected CSVs. `audit_exports.py` checks PDF/SVG/PNG dimensions, vector/text contents and native resolution. `prepare_plot_inputs.py` is the historical initial selection recipe; it is not needed for redrawing and predates this visual feedback revision. Use the current `resolved_style.json`, `methods.json` and `figure_style_v2.json` for the revised design.

The style JSON records the original specification plus the latest user feedback. Font sizes remain: axis labels 8.3 pt, ticks/legend 7.3 pt, panel labels 8.8 pt. Lines remain opaque; Figure 2 uses 1.8/1.3 pt curves and numeric SNR spacing. Figure 3 uses categorical offsets ±0.12 and no connecting lines.

English captions: `captions.md`. Settings note: `experiment_settings_note.md`. Skills actually read and applied: `SKILLS_USED.md` and `skill_read_manifest.json`. Visual inspection: `review/visual_review.json`. Technical checks: `review/technical_audit.json`. On-screen size views are approximations; PDF page boxes and the LaTeX preview establish exact physical dimensions.
