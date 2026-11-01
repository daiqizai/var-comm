# NeST-Com V8 — source-referenced perceptual reconstruction

This is an editorial revision of V7 at `231c3e4b5b8a975330a4baf6fd3ad6116c6d7623`, using its closed E1–E6 evidence. No experiments, model inference, timing or bootstrap statistics were rerun. The original V7 directory and frozen CSV bytes are preserved.

- [Double-column manuscript](manuscript_doublecolumn.pdf): 5 pages, 10 pt, anonymous, 99-word abstract.
- [Double-spaced single-column manuscript](manuscript_singlecolumn.pdf): 15 pages.
- [Editable Word](manuscript_review.docx): 14 pages in Microsoft Word; 3 native tables and 56 editable OMML expressions.
- [Supplement](supplementary.pdf): 16 pages, complete E1–E6 evidence and selected-source protocol.
- [Chinese change notes](CHANGELOG.md), [final report](final_report_zh.md), [claim mapping](claim_evidence_matrix.md), [figure/table provenance](figure_table_sources.json), [acceptance](validation/artifact_review.json).

All 50 final pages were rendered and actually viewed. Fig.1 remains native PPTX; PDF/SVG/PNG are exports (SVG labels are paths). Fig.2 is unchanged. Fig.3 uses exactly the V7 source84/source143 N1024/7dB first-repetition images, with the best distance values bold. The full same-source comparison stays in the supplement.

## Rebuild without experiments

Use the existing figure exports for document-only builds. Run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice, and likewise for `main_singlecolumn.tex` and `supplementary.tex`. The included IEEEtran class is unchanged. For Word, set `PANDOC`, `PDFTOPPM` and `NEST_DOC_WORK` to the installed executables and an external scratch directory, then run `python scripts/build_word.py`; Pandoc and python-docx are required. This rebuild replaces the Word output, so use a copy for independent reproduction. Native Word was used to render this delivered DOCX because the bundled renderer could not find LibreOffice.

`scripts/audit_v8_claims.py` checks the sealed CSVs and their V7 counterparts; it never calls models or resamples statistics. `scripts/build_source_index.py` creates the exact source map and captions. Figure editing/export scripts and supplemental builders are included; rerendering a figure may change PDF metadata without changing its data.

The repository is the PFS worktree on `main`; this Windows directory is a delivery mirror, not a second Git worktree. Publication and repository checks are recorded in `validation/publication_receipt.json` and `validation/repository_checks.json` when completed.
