# NeST-Com WCL: N512 revision

Anonymous manuscript revised from `43ccd51473f978510f7310342b7b513f70c3da5d`, incorporating the completed, independently calibrated N512 experiment.

- [Submission PDF](manuscript_doublecolumn.pdf): IEEEtran, 10 pt, 5 pages including references.
- [Single-column review PDF](manuscript_singlecolumn.pdf): 12 pt, double-spaced, 14 pages.
- [Editable Word](manuscript_review.docx): 14 pages in the native Word render, with the same text, equations and tables. Pagination differs from LaTeX.
- `main.tex` and `main_singlecolumn.tex` share the same section files. Run `pdflatex` twice on either source.
- Figures 1–3 and Tables I/II are byte-identical to the confirmed N1024 manuscript. Table III is generated from the full-precision paired N512 data.
- `captions.md`, `input_manifest.json`, `inputs/`, `tables/` and `validation/` preserve the data mapping and completed review. `scripts/build_word.py` exports editable equations and tables using Pandoc. `scripts/build_table.py` reproduces Table III without computing new statistics.
- [Complete scientific report](../../../results/nestcom_n512_20261028_v1/delivery/REPORT.md): all four metrics, failure counts, endpoints and new/reused computation.

The N512 discussion retains Ada's higher CLIP-I at all three SNRs and lower DreamSim at 19 dB. Independently calibrated budget comparisons do not isolate token ordering or N alone. Historical N2048 DISTS and SNR-mismatch rejection findings remain in Section III.E.

All 5 double-column pages, 14 single-column pages and 14 Word-rendered pages were visually reviewed. No overlapping or cropped content, undefined references or LaTeX overfull boxes remained. Word was rendered with the installed Microsoft Word application because the packaged renderer could not find LibreOffice. Anonymous author and document metadata are retained. Publication verification is separate from the original experiment receipts.
