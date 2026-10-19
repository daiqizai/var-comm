# N2048 two-method validation: final delivery

The fixed 100-source, two-method validation is complete. The reviewed final
figure is `figures_v2/n2048_full_comparison.pdf` (editable SVG and 600 dpi PNG
beside it). `figures_v1` is retained as the first export with its recorded title
clipping; v2 changes title layout only. The earlier report's v1 figure pointer
is superseded by this final index. All scientific numbers are unchanged.

- Chinese scientific report: `delivery/REPORT_zh.md`.
- Exact source lengths, overhead and actual energy: `delivery/layout_and_payload.csv`.
- Reception counts: `delivery/link_delivery.csv`.
- Complete seven-metric scores and direct paired intervals: `metrics_v1/`.
- Qualification image review: `qualification_review.json`.
- Final visual review: `figures_v2/actual_visual_review.json`.
- Actual child waits and byte-preserved failure lineage: `execution_receipts/`.
- Explicit four-repeat authorization: `rx_import_continuation_authorization.json`.
- Exact total calls: `delivery/actual_counts.json` (3648 subframe transmissions,
  7300 header/body decoder attempts, including four approved repeated decodes).

At 4 and 10 dB both methods reject all 300 objects. At 19 dB both deliver all
300 correctly. The fixed shared 64-QAM, rate-5/6 layout fits both complete
representations, but does not provide low-SNR reliability. At 19 dB NeST-Com
has favorable paired intervals for PSNR, LPIPS, DreamSim and MS-SSIM; ARPC has
a favorable DISTS interval. DINO-L and source-prediction agreement intervals
cross zero. These are pointwise intervals, without multiplicity adjustment.

The existing N1024 comparison is preserved. No new training, TX probability
passes, tokenizer calls, bandwidth sweep or further scientific stage follows.
Large restoration backups remain deferred at the user's request. The original
completion receipt's publication-pending field records its creation time;
the final normal publication receipt is delivered separately after pushing.
