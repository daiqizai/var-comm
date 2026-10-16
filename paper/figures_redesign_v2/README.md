# NeST-Com Figures 2 and 3 — redesign v2

This package contains only the two requested scientific figures. It awaits style confirmation; no other figures, qualitative photographs, timing, experiments or statistics were changed.

| Figure | PDF / editable SVG  / 600 dpi PNG | Exact page size |
|---|---|---|
|2: four-method performance|`fig2_main_1x4.*`|181.353259  × 55 mm|
|3: within-family paired gains|`fig3_paired_two_panels.*`|88.567870  × 50 mm|

All six panels are also exported separately in `panels/` in the three formats. Their typography is native, not a scaled crop; shared legends belong to the assembled figures. `review/actual_size_preview.pdf` places both figures at the current IEEEtran manuscript widths. Print at 100% without fit-to-page. PNG resolution comes from native rendering at 600 dpi, not changing a DPI label. PDF and SVG contain vector paths and text, with no embedded raster figures. SVG text is editable and uses DejaVu Sans; install/substitute your own compatible font when editing. No font files are distributed.

## Data versions

The repository was read at `24fa1ecf3a6a959e6023d47ff460df41a711d30f`. Compared with the attachment's `9e59f62`, only the previous Figure 2 preview package and release manifest had been added. No newer scientific results replaced the requested inputs.

- Figure 2 uses common500, N1024, SNR 4/10/19 dB and final P100k/Swin200k. The 19 dB entropy-coded partial rows come from T1-C. Every other requested row uses its frozen report or completed supplemental score table.
- Figure 3 uses original raw partial-minus-whole paired rows, entropy 4/10 dB paired rows, and T1-C `common500/PARTIAL_minus_WHOLE` at 19 dB. Both entropy families are corrected in that comparison. No marginal CI subtraction occurs.
- **DreamSim is retained**: historical raw, supplemental entropy, and T1-C match implementation revision/file hashes, checkpoint hashes, ensemble, normalization, square 224 bicubic antialiased processing, and no added 8-bit quantization. The 500 original source archives and preprocessing identities match the current common500 registration. GPU hardware differs; this is mathematical and input compatibility, not a claim of bitwise device identity. See `review/dreamsim_identity.json`.
- new100, cal1000, Kodak and fixed16 are excluded. Older raw reports supply only the raw within-family contrast, never old external baselines. The latest six-method supplemental package still lacks raw-WHOLE; this figure uses the traced historical paired result, not invented new scores.
- The twelve selected Figure 3 intervals happen to have positive lower bounds after the specified direction transform. This is an observation of the loaded rows, not an exclusion rule: zero/negative/cross-zero values are accepted and never filtered. No synthetic epsilon, new CI or mechanism claim is added.

`plot_inputs/point_manifest.csv` maps all 60 displayed points to exact source CSV rows, file SHA256, raw mean/bounds, direction conversion and version. `input_manifest.json` lists all input files. `review/input_remote_verification.json` records a fresh read-only remote match. Only display scaling (agreement × 100), explicit DreamSim sign/bound reversal, filtering and layout were performed.

## Reproduction and styling

Run `python plot_figures.py` with existing Matplotlib; it reads only the included selected CSVs. `python audit_exports.py` checks PDF/SVG/PNG containers and builds separate review images with Pillow/pypdf. The original source tables are not required for rendering. No experiment code is imported. `prepare_plot_inputs.py` records how inputs were selected and verified in the original workstation/repository; it is not needed for redrawing.

`figure_style_v2.json` preserves the user's style specification. `resolved_style.json` records exact dimensions and axis limits; `methods.json` maps public names to final checkpoint IDs. The actual font selected was DejaVu Sans because Arial, Helvetica and Liberation Sans were absent from the rendering environment. Axis labels 8.3 pt, ticks/legend 7.3 pt, panel labels 8.8 pt; Figure 2 curves 1.8/1.3 pt. All method lines remain opaque. Figure 2 keeps prior y limits and numeric SNR spacing. Figure 3 has categorical offsets ±0.12, no connecting lines, and a visible zero reference.

At 10 dB LPIPS, the hollow warm-red 4.2 pt square surrounds the blue 3.6 pt circle without moving either observation. True interval endpoints are drawn above glyphs. Figure 3 uses 3.6 pt symbols and 2.6 pt caps so narrow intervals remain visible. Some intervals are intrinsically shorter than the marker height; they are never artificially enlarged.

English captions: `captions.md`. Actual skill files/references: `SKILLS_USED.md` and `skill_read_manifest.json`. Final visual findings: `review/visual_review.json`. Technical page, raster, text and font checks: `review/technical_audit.json`. Display-sized PNGs are approximations; exact physical sizing is established by PDF page boxes and the LaTeX preview.
