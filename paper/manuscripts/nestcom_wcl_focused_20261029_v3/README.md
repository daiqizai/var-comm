# NeST-Com focused V3 — perceptual narrative revision

Title: **NeST-Com: Fixed-Budget Next-Scale Token Communication**. Anonymous review version; baseline `e2ed72c95ddeab15cc4ca42b1dd7f2e0e5236c8c`. V2 remains intact.

- `manuscript_doublecolumn.pdf`: 5 pages, 10 pt, abstract 90 words.
- `manuscript_singlecolumn.pdf`: 14 pages, double-spaced.
- `manuscript_review.docx`: native editable text, three native tables, six display equations and 69 OMML expressions.
- `manuscript_review_preview.pdf`: actual Microsoft Word export, 14 pages.
- `supplementary.pdf`: 20 pages; added configuration, reception decomposition, perceptual paired evidence and explicit incomplete timing diagnosis.

Fig.2 now presents LPIPS and DreamSim; Table II retains all PSNR and CLIP-I means. Tables I/III lead with perceptual paired effects. Fig.1 and the N1024 Pastries/Dog 2×5 Fig.3 retain V2 asset bytes. All 72 six-system means remain available in the main figures/table. No new quality experiment, objective or training was introduced.

All pages were actually inspected, including a repaired Word Table I layout. `validation/` contains fresh output hashes and exact page-image identities. `figure_table_sources.json` and `input_manifest.json` map current outputs to original frozen data and historical identities. Scripts use the same input CSVs for figures/tables and Word is built from the TeX content.

**Outstanding measurement:** the bounded Quarter/Eighth TX timing failed the source-token identity check at source 89 after 176 validated measured calls. Its original failures and partial results are reported, not promoted to a complete benchmark. See `final_report_zh.md` and the result-directory timing report. This limits only the requested full quality–time comparison.

Build: run `python scripts/draw_figure2.py`, `python scripts/build_tables.py`, `python scripts/build_v3_supplement.py`; compile `main.tex`, `main_singlecolumn.tex`, and `supplementary.tex` with pdflatex twice. `scripts/build_word.py` uses Pandoc plus python-docx and editable OMML. Installed Word provided the inspected export because bundled LibreOffice was unavailable. Rebuilding documents does not authorize re-running experiments.

Git: publication is confined to the three V3 experiment/result/manuscript roots in the sole remote main repository; no local commit, forced update or merge. Actual publication/check receipts are recorded by the task and final user response; this file records the pre-publication baseline rather than an invented self-referential commit.
