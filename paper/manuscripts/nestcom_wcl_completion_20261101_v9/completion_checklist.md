## 2026-11-02 图版接续

本轮 Fig. 2 横排与全格式交付已完成实际编译和 34 页目视检查（5/15/14 页）。详细变更与来源见 [修订说明](FIG2_1X4_REVISION_20261102.md)。以下保留原 V9 科学交付记录；本轮未重跑实验。

# V9 completion checklist

- **P1: done** — Integer CDF/AC/PHY recipe, 1241 candidate rows, 24 LDPC layouts, model files/recipes/relocatable verification, 6 CPU cases and 2 actual TX/RX vectors. Evidence: `inputs/reproduction/; ../../../../experiments/nestcom-v9-completion-20261101/README.md`.
- **P2: done** — Current N1024 paid-NULL: 33 actual checks, 12 MCS paths, original records/waveforms preserved. Evidence: `../../../../results/nestcom_v9_completion_20261101_v1/execution_receipts/paid_null/`.
- **P3: done** — 216 Token logical frames, all 864 four-arm outputs, five metrics, 24-source paired intervals. Evidence: `inputs/kodak_v9/`.
- **P4: done** — Closed Ada sampling/argmax table; 165 descriptive rows recomputed; fixed median-rank/worst examples. Evidence: `inputs/v9_package/; figures/supp_examples_distribution.pdf`.
- **P5: done** — Matched control100 quality and timing32 TX costs; balanced complete-system evidence and ResARC positioning. Evidence: `tables/table_i.csv; sections/results.tex; references.tex`.
- **P6: done** — 5-page main, 15-page double-spaced single, 14-page native Word, 27-page supplement; 61 pages reviewed. Evidence: `validation/artifact_review.json`.
- **optional: not-run-optional** — Perceptual-Swin training, ResARC wireless baseline, extra data/SNR/parameter searches. Evidence: `Outside this work order`.

No scientific task is blocked. Both required repository checks passed and normal main publication was verified at artifact commit `c4e85c4d984e12b8c6e337f8cf97fd5cf8872b94`; see validation/publication_receipt.json. Private adapted weights remain in PFS with file identities and training recipes; they are not represented as public downloads.
