# Editorial revision of the completed N512 manuscript

Base: `ccf4084165cc57e356157ae074c9bfdedb03d2fa`. Scope: wording, organization and document synchronization only.

1. Abstract now includes the existing shared-model PSNR gains at both 1024 and 512 symbols, alongside the unchanged utilization and external-comparison findings.
2. Four introduction paragraphs connect the full-grid coarse description, residual refinement, endpoint/capacity mismatch and the complete transmit/receive design. All existing related-work citations remain.
3. Section II.B explicitly connects retained tokens to actual-length feasibility. Section II.C states the two roles of received boundary tokens: residual correction and finer-scale conditioning.
4. Section III.D separates descriptive WHOLE−Ada mean PSNR differences from the existing source-paired NeST−WHOLE evidence. It adds the shared 10 dB target/MCS and the 79.30%→93.46% utilization example.
5. The conclusion returns to the method and evidence rather than repeating every experiment. Adverse CLIP-I and DreamSim results, failure denominators and cross-budget interpretation limits remain.
6. Figures 1–3, Tables I–III, metric inputs and historical experiments remain byte-identical. No new confidence interval, significance test, inference, channel event or timing was run.
