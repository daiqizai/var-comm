# User-authorized revision, 2026-09-29

The original execution sheet and failed clean-prior preflight remain immutable evidence. This revision replaces section 3.2 and engineering check 4.2 only. All original oracle formulas, four A1 difficulty targets and M1-M3 thresholds remain unchanged.

The fixed likelihood variance is noise variance alone, eta squared times the calibration-estimated variance of the downsampled unit noise. Its beta=0, lambda=1 version is the untuned reference. At exactly zero noise the fixed MAP uses the nearest-codeword limit.

The calibrated variance adds beta times s_k squared. A2 and V independently select from exactly beta in [0, .05, .1, .2, .5] and lambda in [.5, .75, 1, 1.5], using the same first 200 calibration sources, seeds 4101/4102/4103, and true-prefix mean true-token log probability at scales 7-10. Each scale has equal positive weight; within a scale every token has equal weight. Scores then average seeds and sources equally. Ascending beta then lambda breaks exact ties. A1 uses the same grid for reporting log probability only; positive scalar variance does not change its nearest-codeword MAP. Calibrated results determine M1-M3; fixed results are also retained.

Before development is read, freeze all four noise levels and every selected parameter pair in frozen_config.json. The difficulty targets still use A1 TF accuracy, equally averaged over scales 8-10. A deterministic coarse 1 dB search from -20 to 60 dB is refined at 0.1 dB within one dB of each closest target; report actual achieved accuracies and target errors.

The hard V consistency check is preregistered at **30 dB**, without increasing SNR after seeing outcomes. Both TF and CL must reach 99 percent for the fixed version and separately for every calibrated parameter pair selected at the four difficulty levels. Reuse the original four engineering calibration sources (indices 0/100/200/300) with the same three calibration seeds. Check pooled token agreement for each version/mode; also retain per-source and per-scale results. All sources remain calibration sources, and the scoring subset is unchanged. Failure stops before development; it is not permission to select parameters against this check or discard failing pairs.

The project AWGN actually uses real-coordinate standard deviation 10^(-SNR_dB/20), so real variance is 1/SNR_linear. No extra 3 dB adjustment is appropriate. O1 uses this same convention. The 4096-complex-use standardized latent experiment is only approximately comparable to P4084; no claim of exact N/energy matching is added.

All new code and results are versioned independently. Original source, failed results, the safe 29001 checkpoint, original locks, and hardware settings are preserved.
