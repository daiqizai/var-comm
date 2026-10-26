# NeST-Com WCL: reviewed anonymous LaTeX manuscript

This is a full, editable LaTeX transcription and limited technical revision of
the user's `NeST_Com_WCL_Draft_20261022.pdf`, archived under
`paper/manuscripts/nestcom_wcl_draft_20261022/` at base commit
`ff1273b2e71e0db73a8a306ef871f9c501f89ad9`.
The original manuscript and final Fig.1 remain unchanged.

- `main.tex`: anonymous IEEEtran manuscript; sections and references are editable.
- `manuscript.pdf`: rendered five-page double-column manuscript.
- `REVIEW_zh.md`: technical findings and disposition of the five suggested fixes.
- `tables/`: Tables I--III in LaTeX and full-precision frozen CSV.
- `figures/`: the original vector figure PDFs, reused without redrawing.
- `figure2_data.csv`, `figure3_metrics.csv`: exact existing plotting/display data.
- `input_manifest.json`, `implementation_evidence.json`: input and code identities.
- `audit.json`, `pdf_stream_identity.json`, `visual_review.json`: targeted checks.

Compile from this directory with a standard TeX installation containing
IEEEtran, amsmath/amssymb, graphicx, booktabs, cite, url and hyperref:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The Codex editor's built-in compiler also successfully compiled the project.
No BibTeX run, new experiment, inference, image reconstruction, timing run or
bootstrap is required. Tables apply display rounding only. The fixed main-paper
decoder is the legacy Stage-A update-38000 model, not the later progressive-RD
pilot or scale-up checkpoints.

Anonymous authorship follows the user's explicit instruction. This is a review
artifact, not a journal submission. The official single-column submission copy
and final author metadata remain submission-stage tasks.
