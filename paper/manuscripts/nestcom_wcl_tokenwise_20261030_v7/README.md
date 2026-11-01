# NeST-Com V7 — complete artifacts

Main configuration: `original_receiver_plus_D_token`. Title: **NeST-Com: Fixed-Budget Next-Scale Token Communication**. Anonymous review version. Historical V6 and all original scientific records are preserved.

E1–E6 and figure selection are closed. See `final_report_zh.md` for findings, unfavorable results, actual computation/reuse, limitations and delivery paths. `validation/artifact_review.json` binds every actually viewed final page to the delivered document hashes; earlier draft reviews certify only their original files.

The complete entry points are `manuscript_doublecolumn.pdf`, `manuscript_singlecolumn.pdf`, `manuscript_review.docx`, and `supplementary.pdf`. Native LaTeX, PPTX, Matplotlib scripts, editable Word tables/equations, PDF/SVG/PNG figures and full-precision CSVs accompany them. Model weights, native arrays, caches and large logs stay on PFS.

## Rebuild documents only

Use the frozen delivered input CSVs. Run `scripts/build_evidence_tables.py`, `scripts/audit_closed_claims.py`, and `scripts/build_supplement.py`. Compile `main.tex`, `main_singlecolumn.tex` and `supplementary.tex` twice with pdfLaTeX. Set `PANDOC`, `PDFTOPPM` and optionally `NEST_DOC_WORK`, then run `scripts/build_word.py`. It reads the same TeX/CSVs and preserves native text, three tables and OMML equations. Run `scripts/build_source_index.py` after changes and repeat real full-page rendering/review; do not reuse old visual PASS.

Figure sources are in `scripts/` and native Fig.1 in `figures/figure1.pptx`. The fixed scientific figures need no model calls to redraw. Source/data conditions and hashes are in `figure_table_sources.json`; claim populations and limitations are in `claim_evidence_matrix.md`.

Experiment entry points are separately registered. Do not run launchers as part of manuscript compilation, and do not repeat completed timing or statistical stages. Final normal remote-only Git publication is verified in the external publication receipt and final user delivery; no local commit or force push.
