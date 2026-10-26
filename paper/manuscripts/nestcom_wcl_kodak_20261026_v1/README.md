# NeST-Com WCL: final current-model Kodak revision

This anonymous manuscript is based on commit
79ac43c647209a757ee0c97c83911f3da51f258b. It applies the user's C1-C9 edits
and integrates the actually completed current-model Kodak evaluation.

- `manuscript_double_column.pdf`: normal 10-point IEEEtran, 5 pages.
- `manuscript_single_column_double_spaced.pdf`: 12-point, one column,
  explicitly double-spaced body, 14 pages.
- `manuscript_review.docx`: synchronized editable equations/tables, 14-page
  Microsoft Word review export; figures retain their frozen content.
- `main.tex` / `main_singlecolumn.tex`: compile twice with pdfLaTeX.
- `latex_source.zip`: source-only delivery archive, kept locally because the
  repository excludes ZIPs; all unpacked sources and figures are tracked.

Fig.1-3 retain their exact previous PDF bytes. Fig.2 remains common500;
Fig.3 remains the selected 10 dB Pastries/Dog realizations. Table I has
separate common500 and Kodak blocks. Table II remains N2048 matched100,
and Table III remains the original 10 dB software latency measurement.
The current Stage-A update38000 decoder is used; progressive-RD is separate.

Kodak input/results: `results/kodak_current_20261026_v1/delivery/`.
Full four-metric tables and 60 source-paired contrasts are provided there.
The 19 dB Kodak PARTIAL-WHOLE DreamSim interval crosses zero; the manuscript
does not generalize favorable mean rankings into universal superiority.
Preserved engineering exceptions are detailed in the independent report.

Publication/check receipts live under the corresponding PFS runtime root;
the final source commit and remote SHA are reported after the normal push.
