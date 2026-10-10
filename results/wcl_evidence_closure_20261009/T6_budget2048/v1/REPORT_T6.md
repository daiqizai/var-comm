# T6: N2048 digital-branch confirmation

This is the actually completed second-budget experiment: 100 pre-fixed sources, three noise realizations, three SNRs and three frozen methods, totaling 2,700 method-frame conditions. Both calibration branches finished before the confirmation source pipeline. No failures or zero increments are removed.

## Scope and source audit

The single entropy family is EC_VAR_WHOLE, chosen using N1024 calibration only. The original1000 calibration selected all nine N2048 method/SNR policies; confirmation images and scores did not select policies. The content audit scope is `all_available_audited_source_content_hashes`: available_reference_hash_count=23500, unavailable_reference_hash_count=0. Missing hash domains are retained in content_duplicate_check.json. No source was replaced after content inspection. This establishes exclusion within the audited coverage, not that these images were never encountered anywhere.

## Complete-scale boundaries and receiver outcomes

| Method | SNR | m10, K=0 frames | K=0 frames | Gray frames | Body CRC rejects | Partial policy identical to WHOLE |
|---|---:|---:|---:|---:|---:|---|
| Complete-scale raw transmission with VAR completion | 4 | 0/300 | 300/300 | 0/300 | 0/300 | False |
| Complete-scale raw transmission with VAR completion | 10 | 0/300 | 300/300 | 0/300 | 0/300 | False |
| Complete-scale raw transmission with VAR completion | 19 | 300/300 | 300/300 | 0/300 | 0/300 | False |
| Partial-scale raw transmission with VAR completion (NeST-Com) | 4 | 0/300 | 0/300 | 0/300 | 0/300 | False |
| Partial-scale raw transmission with VAR completion (NeST-Com) | 10 | 0/300 | 0/300 | 0/300 | 0/300 | False |
| Partial-scale raw transmission with VAR completion (NeST-Com) | 19 | 300/300 | 300/300 | 0/300 | 0/300 | True |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion | 4 | 0/300 | 300/300 | 0/300 | 0/300 | False |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion | 10 | 24/300 | 300/300 | 0/300 | 0/300 | False |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion | 19 | 300/300 | 300/300 | 0/300 | 0/300 | False |

Raw CRC rejection and gray fallback are different events under KEEP. The mechanism table separately preserves header, CRC, framing, entropy canonical rejection and accepted token-error diagnostics. Failure counts can overlap; all2700 scored rows remain in per_frame.csv. Entropy source-length fallback and raw PARTIAL choosing K=0 are distinct mechanisms.

If PARTIAL selects the same full-scale physical action as WHOLE, its exact state, image and metrics must match; zero paired increments are valid boundary results. Complete m10 transmission is reported only where the actually transmitted frame has m=10,K=0. It is never inferred from an average compressed length.

## Source-paired comparisons: all four metrics

| Method minus reference | SNR | Metric | Mean difference | 95% interval | Interpretation |
|---|---:|---|---:|---|---|
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 4 | psnr_db | +1.22045096 | [1.08955604, 1.36202221] | method_better |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 4 | lpips_alex | -0.03887645 | [-0.04195228, -0.03583890] | method_better |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 4 | dinov2_vitl14_cosine | +0.08972028 | [0.06851394, 0.11233250] | method_better |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 4 | convnext_top1_source_prediction (percentage points) | +12.00000000 | [5.00000000, 19.00000000] | method_better |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 10 | psnr_db | +0.05965211 | [0.04953702, 0.07097130] | method_better |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 10 | lpips_alex | -0.00100308 | [-0.00128354, -0.00072825] | method_better |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 10 | dinov2_vitl14_cosine | +0.00175916 | [-0.00105140, 0.00473929] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 10 | convnext_top1_source_prediction (percentage points) | +2.00000000 | [0.00000000, 5.00000000] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 19 | psnr_db | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 19 | lpips_alex | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 19 | dinov2_vitl14_cosine | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus Complete-scale raw transmission with VAR completion | 19 | convnext_top1_source_prediction (percentage points) | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 4 | psnr_db | +1.28450609 | [1.07715888, 1.49751418] | method_better |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 4 | lpips_alex | -0.03556161 | [-0.04081326, -0.03030630] | method_better |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 4 | dinov2_vitl14_cosine | +0.07144501 | [0.05289289, 0.09152009] | method_better |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 4 | convnext_top1_source_prediction (percentage points) | +8.00000000 | [1.00000000, 15.00000000] | method_better |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 10 | psnr_db | +0.12900334 | [0.04833106, 0.22628915] | method_better |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 10 | lpips_alex | -0.00109072 | [-0.00199599, -0.00036665] | method_better |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 10 | dinov2_vitl14_cosine | +0.00342785 | [0.00026235, 0.00771128] | method_better |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 10 | convnext_top1_source_prediction (percentage points) | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 19 | psnr_db | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 19 | lpips_alex | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 19 | dinov2_vitl14_cosine | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| VAR-conditional entropy-coded complete-scale transmission with VAR completion minus Complete-scale raw transmission with VAR completion | 19 | convnext_top1_source_prediction (percentage points) | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 4 | psnr_db | -0.06405513 | [-0.23222642, 0.10108877] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 4 | lpips_alex | -0.00331484 | [-0.00910316, 0.00215063] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 4 | dinov2_vitl14_cosine | +0.01827527 | [-0.00120572, 0.03938026] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 4 | convnext_top1_source_prediction (percentage points) | +4.00000000 | [1.00000000, 8.00000000] | method_better |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 10 | psnr_db | -0.06935123 | [-0.16816043, 0.01391479] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 10 | lpips_alex | +0.00008764 | [-0.00070332, 0.00106480] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 10 | dinov2_vitl14_cosine | -0.00166869 | [-0.00650700, 0.00264595] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 10 | convnext_top1_source_prediction (percentage points) | +2.00000000 | [0.00000000, 5.00000000] | interval_includes_zero |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 19 | psnr_db | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 19 | lpips_alex | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 19 | dinov2_vitl14_cosine | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |
| Partial-scale raw transmission with VAR completion (NeST-Com) minus VAR-conditional entropy-coded complete-scale transmission with VAR completion | 19 | convnext_top1_source_prediction (percentage points) | +0.00000000 | [0.00000000, 0.00000000] | exact_zero_difference |

Deltas remain method minus reference. Negative LPIPS is better; ConvNeXt is source-prediction agreement, not classification accuracy. Canonical CSV agreement values are proportions; displayed paired agreement values are percentage points. Three noise results are averaged within each source before the existing100-source bootstrap (10,000 replicates, seed2026100701). Intervals are pointwise and unadjusted for multiple comparisons. Inclusion of zero is not evidence of equivalence. This export performs no new bootstrap.

## Computation and interpretation

calibration_execution.csv records actual pilot/full logical frames and packet callbacks against their finite preregistered caps. Qualification failures unsupported by the real constructor are documented, not silently replaced. Selection within the pilot/full unions is not a claim of global optimality.

Actual confirmation packet callbacks: 4800 of maximum5400; unresolved=0. New VAR renders: 573 plus 2 exact replay qualification calls. New entropy source decodes: 0. Actual score accounting: {"new_quality_calls":573,"new_reference_preparations":100,"old_quality_cache_reads":0,"old_reference_cache_reads":0,"within_run_exact_image_reuses":2127}.

N1024 used the previously published common500 source population; N2048 uses the separately registered confirmation100. Their means cannot be treated as a paired cross-budget trajectory. Two budget points do not establish a precise bandwidth saving, an equal-quality crossing or an interpolated optimum. Any m10/full-scale fallback, reduced partial advantage, negative difference or zero difference in this experiment limits the claim rather than motivating another search.

Only the supplied completed artifacts are exported. The source parent, rendering/calibration owners and score parent were checked for actual waits and successful exits. No model, channel, image rendering or resampling is run by this command.
