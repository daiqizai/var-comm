# NeST-Com: focused revision, 2026-10-29

Anonymous manuscript. This revision edits the narrative and figures using the completed experiments at base `2c303b615f73edd0a7a971308c0df8bd5bb885d0`. It adds no training, model inference, wireless frames, metric evaluation, or bootstrap draws.

- [Double-column paper: 5 pages, 10 pt](manuscript_doublecolumn.pdf)
- [Double-spaced single-column paper: 13 pages](manuscript_singlecolumn.pdf)
- [Editable Word: 13 pages](manuscript_review.docx) · [actual Word PDF preview](manuscript_word_preview.pdf)
- [Supplement: 8 pages](supplementary.pdf)
- [Configuration and evidence index](configuration_and_supplement_index.md)
- [Changes](CHANGELOG.md) · [claim/evidence map](claim_evidence_matrix.md) · [completion](completion_checklist.md)
- [Figure/table sources](figure_table_sources.json) · [captions](captions.md)
- [No-feasible-endpoint audit](implementation_status.md)
- [Actual all-page review](validation/artifact_review.json) · [five-page print preview](validation/print_preview/)

Fig. 1 preserves the native [PPTX](figures/figure1.pptx) and original three-panel arrangement. Fig. 2 is 7.16×2.40 in; its 36 means and 36 confidence intervals are unchanged. Fig. 3 is 6.0×3.35 in and uses the already displayed lamp, with a common 100×100 ROI and full-image PSNR/LPIPS values. Original two-case N512 and N1024 assets remain in `supplemental/`.

## Build from this directory

Use Python with Matplotlib, Pillow, NumPy, PyMuPDF, and python-docx; Word export also needs Pandoc and Poppler. All figures use local frozen inputs. Run `python scripts/draw_figure2.py`, `python scripts/draw_figure3_roi.py`, and `python scripts/build_supplement.py` for deterministic artifact generation. Compile `main.tex`, `main_singlecolumn.tex`, and `supplementary.tex` twice each with `pdflatex -interaction=nonstopmode -halt-on-error`. Rename the two main PDFs to the published manuscript filenames. `python scripts/build_word.py` reads the same main TeX, sections, figures, equations, references, and tables; set `PANDOC`, `PDFTOPPM`, and `NEST_DOC_WORK` if their programs/work directory are elsewhere. It does not compute science.

Final double-column floats are queued ahead of their textual subsections to place Fig. 2 on page 3 and Fig. 3 on page 4. Single-column and Word use normal reading order. Native Word body/tables and 102 OMML math elements remain editable; only figures are images. Fig. 1's PPTX is its editable text source; its SVG may contain outlines. Fig. 2/3 have editable SVG text plus Python sources.

Historical manuscript directories, raw evidence, and paused research branches are preserved. Arrays/checkpoints/full logs stay in their existing PFS locations.
