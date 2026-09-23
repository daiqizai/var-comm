# Digital m8 base plus additional continuous latent enhancement

- This is a newly authorized experiment, not a restart/relabeling of the closed fixed-m7 RGB residual hybrid.
- Preserve the existing full m8 N3060 waveform, header/FEC/modulation/power/noise and receive/failure rules. Additional budgets are512/1024 only, giving N3572/E7144 and N4084/E8168. Never renormalize the concatenated waveform or subtract from the base allocation.
- Freeze official encoder, quant_conv, quantizer/codebook/scale mappings and VAR. Keep original image decoder D0 unchanged; phaseA trains only an independent copy of post_quant_conv plus decoder Dc.
- Inspect actual latent shapes. F is prequantization, Fq all true scales, Fb_TX true m8 plus original deterministic completion, Fb_RX actual received prefix plus same completion. Residual is F-Fb_TX; correction is applied only after final VAR latent completion, with no requantization or history injection.
- RX accepts only enhancement observation, actual Fb_RX, shared SNR and protocol-observable status. No GT F/e/Fb_TX, true BER/error positions, sender per-image norm or free feedback.
- PhaseA uses fixed50/25/25 continuous/quantized/interpolated inputs, MSE[0,1]+0.1LPIPS[-1,1]. Record branchwise PSNR/LPIPS. Do not mechanically start phaseB if no useful continuous-latent improvement is demonstrated.
- PhaseB freezes selected Dc while retaining input gradients; train two small E/Ds and a receiver-only refiner control. Use image loss+0.01 mean normalized latent error; sF comes from training only. Actual digital errors, all header/body failures, and independent reproducible added noise are mandatory.
- All inference returns original D0 when enhancement is disabled. Header rejection uses original gray fallback, with all resources charged. Body CRC failure retains the originally allowed candidate. No GT output selection.
- Original20k train/1k calibration/100 development and noises only. DINO is report-only, no new holdout. Preserve raw/arithmetic digital competitors at exact total budgets, same-Dc controls, oldm9/N3060, R3/Deep/external natural-rate references.
- GPU0 only when available. Do not stop other authorized tasks, alter shared environments, overwrite checkpoints/results or modify old experiments. Pause this task's HiFi workers/resumers with provenance. Every authorized Git commit must be pushed and verified.

## User-authorized post-review research (2026-09-23)

`research/` adds a separate pure continuous N4084/E8168 control and a matched light-TX versus full-TX pair. These are explicit exceptions to the original enhancement-budget-only scope above; original A/B models and rules remain unchanged. See research/config.json and reports/research_20260923_protocol.md. Do not reinterpret old checkpoints as trained at the new pure budget. Commit phase1 repair/results first; commit phase2 source/results separately after actual acceptance.
