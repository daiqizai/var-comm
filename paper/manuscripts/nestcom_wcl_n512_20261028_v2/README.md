# NeST-Com WCL: representation and fixed-frame narrative revision

Revised from `ccf4084165cc57e356157ae074c9bfdedb03d2fa` without new scientific computation. The title, anonymous authors, three figures, three tables and all frozen results are retained. The previous manuscript remains in `../nestcom_wcl_n512_20261028_v1/`.

The abstract contains 91 words after expanding nonbreaking spaces. Introduction and Section III.D now separate the practical value of the complete next-scale system from the shared-model gains of within-scale transmission. WHOLE−Ada differences are descriptive means; all significance language refers to existing direct source-paired intervals. See `validation/claim_audit.json` and `CHANGELOG.md`.

`main.tex` and `main_singlecolumn.tex` share the same sections. `scripts/build_word.py` exports the same manuscript to editable Word; `scripts/build_table.py --check` verifies unchanged Table III. Final compilation and visual review are recorded in `validation/`.

## Deliverables and review

- `manuscript_doublecolumn.pdf`: anonymous 10 pt, 5 pages.
- `manuscript_singlecolumn.pdf`: single-column double-spaced review version, 14 pages.
- `manuscript_review.docx`: editable Word with 3 native tables and 6 display equations; 15 pages in the verified Word rendering.
- `main.tex` / `main_singlecolumn.tex`: shared-section LaTeX sources.
- `results_outline_zh.md`: revised argument and evidence map.

All pages were rendered and visually inspected. No overfull boxes, undefined citations or references were reported. `validation/artifact_review.json` records final output hashes and checks. Word pagination differs from LaTeX; its textual content, figures, equations and tables derive from the same sources. Repository check receipts are in `validation/repository_checks/`.
