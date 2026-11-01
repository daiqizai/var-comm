## Current figure revision — 2026-11-02

Fig. 2 is now a 181.353259 × 64 mm four-panel horizontal figure, integrated in all three main manuscript exports. All 92 means and 92 confidence intervals are unchanged. Actual output: 5-page double-column, 15-page single-column, 14-page editable Word; all 34 revised pages inspected. Build the figure with `python scripts/draw_figure2_1x4.py`. See [revision and provenance](FIG2_1X4_REVISION_20261102.md). The historical publication paragraph below describes the preceding V9 delivery; this revision has its own outer publication receipt.

# NeST-Com V9

Anonymous manuscript: **NeST-Com: Fixed-Budget Next-Scale Token Communication**.

- `manuscript_doublecolumn.pdf`: 5 pages, 10 pt, 99-word abstract.
- `manuscript_singlecolumn.pdf`: 15 pages, double spaced.
- `manuscript_review.docx`: editable body, 3 native tables, 59 OMML elements; 14 pages in Word.
- `supplementary.pdf`: 27 pages; full S1–S8 recipe/results.
- `main.tex`, `main_singlecolumn.tex`, `supplementary.tex`, `sections/`, `tables/`, `references.tex`: compilation sources.
- `figures/figure1.pptx`: native structural figure. Fig.2/3 scripts, PDF/SVG/PNG and print-size previews are included. Text editing for Fig.1 uses PPTX; path-based SVG glyphs are not represented as editable text.
- `inputs/`, `figure_table_sources.json`, `input_manifest.json`: pinned values and identities. Native scientific arrays and weights remain in PFS; display PNGs do not replace metric inputs.
- `validation/`: actual compile, data, Word and all-page visual review.
- `completion_matrix.csv`, `final_report_zh.md`: scope, results and limitations.

Build from this directory using `pdflatex main.tex` twice, similarly for `main_singlecolumn.tex` and `supplementary.tex`. Figure scripts use Python, NumPy, Pillow and Matplotlib. Word is built from the same LaTeX sections and table CSVs with `scripts/build_word.py`; its actual runtime and conversion receipt are recorded in validation. Reader-only CPU and asset verification entry points are described in `experiments/nestcom-v9-completion-20261101/README.md` at repository root.

Based on remote main `dde679637b30f1166bc540d4c7b6fa91fe457e8c`. V8 and all previous evidence are preserved. No training, new calibration, main-experiment replay, new timing or stopped branch was launched. Both required repository checks passed. Normal main publication was verified at artifact commit `c4e85c4d984e12b8c6e337f8cf97fd5cf8872b94`; validation/publication_receipt.json records it. A metadata-only follow-up records the receipt; the outer delivery receipt gives the final remote SHA.
