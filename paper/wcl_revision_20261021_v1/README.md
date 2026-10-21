# WCL revision package

Title: **NeST-Com: Fixed-Budget Next-Scale Token Communication**

This package revises the outline and actual figures/tables; it is not a full paper draft. The read-only baseline is `a5f8ceb0233b0e1f599ee8bb902c1a094d47119d`. The sole new experiment is the completed, separately registered ARPC decoder sensitivity study.

- `outline_zh.md`: Chinese paragraph tasks; `outline.docx`: the same outline with actual figures/tables.
- `latex/main.tex`: English outline skeleton with actual figures/tables, not full prose.
- `figure1.pptx`: editable modular source reflowed from the user's exact original PPTX; `figure1.pdf/png`: final export.
- `figure2.pdf/svg/png`: 1×4 main comparison, 7.16×2.165 inches.
- `figure3.pdf/svg/png`: selected development examples, complete field of view plus equal per-row ROIs. Native 256×256 pixels are preserved; larger display does not imply additional image detail.
- `table_i–iii.csv/tex`: full-precision data and formatted tables. Table I is now the N1024 mechanism table; Table II is the N2048 paired table; Table III is existing 10 dB software processing latency.
- `captions.md`, `results_outline.md`, `input_manifest.json`, `data_validation.json`, `visual_review.json`: captions, mapping, provenance and checks.
- `sensitivity/REPORT.md` and CSV: all prespecified ARPC settings and direct paired differences. Main common500 curves are unchanged.

The old paired-gain/utilization figure remains archived in v1; its mechanism results now appear in Table I. The previous v2 2×2 performance plot becomes the 1×4 Fig.2. Fig.3 retains the two existing image cases and adds deterministic same-coordinate ROIs. No old figures or user source files were overwritten.

## Reproduction

Run `python draw.py` in this directory using the original NumPy/Matplotlib/Pillow environment. It reads only `inputs/` and checks plotted data and embedded pixels; it performs no models, channel work or bootstrap. `scripts/` contains the deterministic PPTX reflow and size-normalization sources and Word outline builder. Use the bundled artifact-tool runtime for PPTX authoring; final PDF is a native PowerPoint vector export because the bundled direct PDF exporter had an unavailable font alias. No managed package was changed. Font files are not distributed.

Fig.1 and Fig.2 PDFs contain vector artwork/text. Fig.3 necessarily embeds the original raster photographs together with vector annotations. The editable SVG retains the source image pixels and text. See `visual_review.json` for final-size inspections and the deliberate 7 pt minimum for diagram auxiliary text.

N2048 CLIP and complete manuscript prose remain outside this execution. No main quality grid, training, timing, source encoding or channel experiment was rerun.

## Fig.1 in the open LaTeX outline (2026-10-21)

At the user's request, `latex/main.tex` now uses the original supplied
`system_diagram_v3_entropy.pdf` at 7.16-inch width, preserving its complete
layout and vector content. The previously reflowed `figure1.*` and the Word
outline remain archived as delivered; this change targets the open LaTeX
document only. The original editable source is retained in
`inputs/system_diagram_v3_entropy.pptx`.
