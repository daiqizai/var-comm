# NeST-Com WCL manuscript — 27 October 2026

Anonymous complete manuscript revised from public commit `1f2a86a41b65fe90c0bf245076212b0a27852deb`. The completed P1 diagnostic is separately reported in `results/wcl_revision_20261027_v1/p1_delivery/`; main common500/Kodak statistics and all three figures are unchanged. The two Ada examples were completed in the prior version and are reused.

## Reading copies

- `main.pdf`: normal IEEEtran 10pt, double-column, 5 pages.
- `main_singlecolumn.pdf`: 12pt single-column, double-spaced; identical content.
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

## Revision evidence and scope

`CHANGELOG_zh.md` lists each location, change and exact evidence. `revision_inputs.json` pins the new policy/count/diagnostic records. The original `input_manifest.json` is preserved: its older base identifies the inherited figure/table data, not the manuscript revision parent.

The new 1800-outcome TX-policy mismatch diagnostic is complete. It shares actual AWGN vectors across families, keeps true RX variance, includes all failures, and preserves the original post-loop engineering assertion failure plus independent metadata closure. No completed science was rerun. See the full diagnostic report for 24 means, 12 paired intervals and compute/reuse counts.

P2 (Ada timing/decoding sensitivity, alternative calibration objectives) was not started. No new main-grid experiment, calibration or training was run. Ada has no same-protocol latency number. The old automation remains paused.

## 2026-10-27 editorial corrections after 35736084

- II.B: replaced the ambiguous no-feasible-endpoint sentence with an explicit conditional; separated the one-realization 100-image pilot from the three-realization 1000-image final evaluation. Final full-scale candidates and fixed tie order are described without development-history language.
- III.A/B: directly report observed reception counts; keep A0/B0/C0 definitions and results while removing registration and defensive phrasing.
- III.D/IV: report only the observed 8/10/12 dB mismatch outcomes; remove the unmeasured sharp-transition claim and the mismatch sentence from the conclusion.
- Added `results/wcl_revision_20261027_v1/p1_bootstrap_clarification_v1.md` and its JSON companion. Original registration/statistics bytes and hashes, figures, tables and all scientific results are preserved. No scientific or bootstrap computation is performed by this edit.

`editorial_revision.json` records the subsequent text-only correction and immutable scientific-file check. `revision_inputs.json` continues to describe the preceding P0/P1 release; the result-directory README gained only the companion metadata-note link. Final review evidence is `visual_review.json` and `validation/*editorial*`; older compilation/review receipts remain historical.
