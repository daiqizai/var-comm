# NeST-Com final evidence manuscript

Anonymous review version, based on `8bab318fbc8f8f115e96c9aeffeef10d3c2bb4b6`.

- `manuscript_doublecolumn.pdf`: 5 pages, 10 pt, three figures and three tables.
- `manuscript_singlecolumn.pdf`: 15 pages, double-spaced.
- `manuscript_review.docx`: 15 pages in Word 16; native text, tables and editable OMML equations. `validation/manuscript_word_render.pdf` is its actual Word render.
- `main.tex` / `main_singlecolumn.tex`: shared sections, tables and 17 references.
- `figures/figure1.pptx`: native editable diagram. SVG labels are outlined. All three figures have PDF/SVG/PNG exports and print-size previews.
- `figure3_metrics.csv` and `inputs/figure3_native_mapping.json`: native input identities and displayed values for the first channel repetition, selected before reading reconstruction quality.
- `claim_evidence_matrix.md`, `figure_table_sources.json`, `validation/`, `completion_checklist.md` and `final_report_zh.md`: evidence and acceptance records.

## Artifact rebuild

Run `python scripts/build_tables.py` for tables, and `python scripts/draw_figure2.py` / `python scripts/draw_figure3_n512.py` for figures. Compile each TeX entry twice with pdfLaTeX, placing build files in a separate scratch directory. Existing confidence intervals are reused, not recomputed.

Build Word with `python scripts/build_word.py`, using Pandoc, python-docx and Poppler. Set `PANDOC`, `PDFTOPPM` and `NEST_DOC_WORK` if needed. Text, tables and equations come from the shared TeX sources.

For Fig. 1, run `edit_figure1_evidence.mjs`, `finalize_figure1.mjs`, `export_figure1_powerpoint.ps1`, then `export_figure1_svg.py`. The editor requires `CODEX_NODE_MODULES`; the finalizer accepts `CODEX_ARTIFACT_RUNTIME` and `PRESENTATIONS_SKILL`. Installed presentation tooling and Microsoft PowerPoint are required. Candidates and previews are temporary build files. See `validation/toolchain_identity.json` for installed identities and the final-output filename adjustment.

E1/E2 are complete; do not rerun existing registrations or output folders. Full results are in `results/nestcom_evidence_20261029_v1/`. Arrays, weights and full logs remain on PFS. Prior N512/N1024 experiments and paused branches were not resumed.
