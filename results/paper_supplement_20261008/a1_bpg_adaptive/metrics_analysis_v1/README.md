# Completed adaptive BPG result analysis

Inputs: the actually completed `../metrics_v1`, its owner wait-zero receipt, and the hash-bound original common500 and native256 BPG statistics retained under `.research/`. All numerical extraction uses CSV/JSON precision, never copied report numbers.

- `validation.json`: local availability and integrity scope, exact mean/direction checks, complete grid counts, unchanged old values, and zero new bootstrap/model/PHY work.
- `adaptive_summary_original_precision.csv`: the actual24 adaptive result rows, retaining original numeric strings.
- `paired_interpretation.csv`: the actual96 paired rows without sign changes, with explicitly separate interpretation and percentage-point display columns.
- `paired_results.md`: rounded six-SNR reading tables for all four references.
- `selected_resolution_counts.csv`: source-level frozen encoder resolution choices; each source counted once per SNR.
- `manuscript_results.tex`: concise English results text that retains adaptive BPG's PSNR advantages and the Swin19-dB limitation.
- `结果与边界.md`: Chinese conclusions, scientific limits, cost caveat, and local-cache scope.

Reproduce the read-only checks from the repository with:

```text
python experiments/paper_supplement_20261008/a1_bpg_adaptive/metrics/analyze_completed_metrics.py --workspace C:/Users/11946/Documents/ChatGPT/comm
```

This script verifies existing means and interval ordering but never regenerates bootstrap draws or confidence intervals. The original scientific files are read-only. Its text-reading tables do not replace their source CSVs. 507 downloaded scientific outputs were hash-verified locally;1000 declared source-feature cache files were not downloaded or locally verified. The documented remote full-output audit is separate.
