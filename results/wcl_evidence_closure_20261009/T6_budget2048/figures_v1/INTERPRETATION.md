# Actual cross-budget figure interpretation

These figures compare partial-scale minus complete-scale raw transmission with VAR completion within each budget. N1024 uses the original 500-source holdout and six SNRs; N2048 uses the separately registered 100-source confirmation set and 4, 10 and 19 dB. Each source has three noise realizations. All intervals below are copied from the completed paired CSV; the figure performed no new bootstrap.

At N2048 and 4 dB, all four paired intervals favor partial-scale transmission. At 10 dB, the PSNR and LPIPS intervals favor partial-scale transmission, while the DINOv2-L interval crosses zero and the agreement interval includes zero. At 19 dB, every paired mean and interval is exactly zero.

The zero at 19 dB is consistent with the frozen configurations: both raw arms select m = 10, K = 0, profile 218. All 300 partial-scale frames use that complete-scale configuration. Partial-scale transmission is allowed to select K = 0; the experiment does not force a partial transmission at every SNR.

| N2048 SNR (dB) | Metric | Paired mean | Existing 95% CI |
| --- | --- | --- | --- |
| 4 | PSNR (dB) | 1.2204509606122578 | [1.0895560375439335, 1.362022210942058] |
| 4 | LPIPS | -0.03887644816190004 | [-0.04195227733254433, -0.0358388987807557] |
| 4 | DINOv2-L cosine | 0.08972028307616711 | [0.06851394494436681, 0.11233250287733973] |
| 4 | ConvNeXt prediction agreement (percentage points) | 12.00 | [5.00, 19.00] |
| 10 | PSNR (dB) | 0.059652106203638644 | [0.0495370173750523, 0.07097129846950857] |
| 10 | LPIPS | -0.001003079190850258 | [-0.0012835438763722778, -0.0007282481556758286] |
| 10 | DINOv2-L cosine | 0.0017591583728790283 | [-0.0010513990223407746, 0.004739294335246084] |
| 10 | ConvNeXt prediction agreement (percentage points) | 2.00 | [0.0, 5.00] |
| 19 | PSNR (dB) | 0.0 | [0.0, 0.0] |
| 19 | LPIPS | 0.0 | [0.0, 0.0] |
| 19 | DINOv2-L cosine | 0.0 | [0.0, 0.0] |
| 19 | ConvNeXt prediction agreement (percentage points) | 0.0 | [0.0, 0.0] |

Positive PSNR, DINOv2-L and agreement differences favor partial-scale transmission; negative LPIPS differences favor it. Agreement is prediction agreement, not classification accuracy, and its delta is in percentage points.

The two budgets use different source populations. Their within-budget differences are descriptive mechanism comparisons, not a causal estimate of changing bandwidth on the same images. No equal-quality bandwidth saving, interpolated crossing, or percentage saving is inferred. The plotted pointwise intervals have no multiple-comparison adjustment.

Input provenance and exact SHA-256 bindings are recorded in completion.json and visual_QA.json. The 19 dB configuration explanation is checked against the separately sealed T6 final selected_configurations.csv and mechanism_and_failure_summary.csv.
