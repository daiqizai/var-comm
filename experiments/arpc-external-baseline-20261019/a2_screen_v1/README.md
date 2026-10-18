# A2a finite screening

This CPU-only calculation evaluates all 84 permitted strategies using the saved
100-source lengths and corresponding actual fallback DINO-L quality. PSNR and
LPIPS are diagnostics. No new model, channel, recovery or bootstrap call occurs.

Historical joint header/body exact-success evidence is a proxy: original NeST
payloads and catalogue, same PHY settings, 100 distinct sources per each of 36
MCS/SNR cells. Its Clopper–Pearson Bonferroni 95% simultaneous intervals express
binomial uncertainty, not a theorem transferring reliability to arbitrary ARPC
payloads. All-zero-error cells retain nonzero uncertainty.

Before opening the new quality scores, the screening rule is fixed as follows.
Within each MCS, retain the target with highest mean actual fallback DINO-L;
equal means prefer the larger target. This is dominance within the common-p
proxy only. Keep all 84 strategies in the report. Per SNR propose the highest
point estimate and the different MCS leader with largest upper bound. Any
additional leader whose upper bound reaches the largest lower bound is an
uncovered possible competitor and triggers a budget gate. Do not discard it,
alter intervals after seeing results, or launch additional channel trials.

The output is a nomination, never a measured ARPC wireless result. Entering A2b
still requires real link qualification and a finite, frozen registration. The
original two-candidate/SNR and 18,000-logical-frame ceilings remain unchanged.
