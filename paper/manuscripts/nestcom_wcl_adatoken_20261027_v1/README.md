# NeST-Com WCL manuscript — 27 October 2026

Anonymous complete manuscript based on frozen public commit `6118401ac749939127445ee6cca6f0449fb7a242`. The only new image calculations are the two explicitly authorized Ada development examples; main common500/Kodak statistics are unchanged.

## Reading copies

- `main.pdf`: normal IEEEtran 10pt, double-column, 5 pages.
- `main_singlecolumn.pdf`: 12pt single-column, double-spaced, 12 pages; identical content.
- `manuscript_review.docx`: editable text, equations and two tables; figures are embedded for review.
- `main.tex`, `sections/`, `references.tex`: full manuscript source.
- `captions.md`: synchronized English figure/table captions.
- `CHANGELOG_zh.md`: Chinese explanation of changes, provenance and remaining research.

Fig.1 preserves the user's native bird images, residual-VQ geometry, CDF paths and paid profile path. `figures/figure1.pptx` contains editable native objects; `inputs/figure1_original_editable.pptx` preserves the original. Fig.2 is a two-panel six-system plot. Fig.3 is a two-by-five full-view panel of Pastries/Dog with Original, NeST, Ada, Latent and ARPC. The selected development examples are not statistical test sources.

## Provenance

`input_manifest.json` pins full-precision source CSV paths, sizes and SHA256 hashes. NeST at 19 dB is the corrected T1-C PARTIAL result inherited from the frozen six-system summary. `figure3_metrics.csv` pins each displayed reconstruction, source, noise record, native float file, saved PNG and single-realization score. The legacy manifest contains method-specific seed values (including NeST 6201); these were retained, rather than overwritten with the planning document's blanket 2001 description. New Ada alone uses the separately registered 2001 event. No cross-method noise equivalence is claimed.

`tables/table_i.csv` distinguishes the 500/24-source main-policy comparison from the 100-source controlled diagnostic. `tables/table_ii.csv` preserves all 36 LPIPS/CLIP means and their existing CIs. `figures/figure2_data.csv` preserves all 36 PSNR/DreamSim means and CIs. The original 72-row input, full four-metric figures, Kodak plots and all A/B analyses remain in the prior result packages.

## Rebuild without experiments

From this directory, with Python/matplotlib/Pillow:

```text
python scripts/build_tables.py
python scripts/draw_figure2.py
python scripts/draw_figure3.py
python scripts/validate_delivery.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main_singlecolumn.tex
pdflatex -interaction=nonstopmode -halt-on-error main_singlecolumn.tex
```

No rebuilding command samples a model, simulates a channel, computes a metric, or resamples a CI. Fig.1's native editing script uses the first-party artifact-tool on the preserved original PPTX, followed by native PowerPoint PDF export. SVG was exported from that PDF using PyMuPDF with editable vector paths and outlined labels; label text remains editable in the native PPTX. The PNGs are actual renders; their DPI metadata does not invent photograph detail. Fig.3 retains original saved 256×256 photographic pixels with vector labels in PDF/SVG. No font files are distributed.

`scripts/build_word.py` is the document-only Pandoc/OOXML conversion. It expects an installed Pandoc, Poppler and python-docx (or the documented Codex bundled runtime). The canonical Word renderer was attempted but LibreOffice was absent; installed Microsoft Word generated the inspected review PDF instead. `visual_review.json`, `pdf_structure.json`, `data_validation.json` and `validation/` record actual checks.

## Unperformed follow-ups

Alternative calibration objectives, SNR mismatch, extra Swin training and Ada same-protocol timing are not completed results and were not run. The manuscript reports only the measured five-system timing. It retains Ada's perceptual advantages at relevant working points, Swin/BPG PSNR advantages, and the existing N2048 ARPC DISTS advantage. No universal superiority is claimed.
