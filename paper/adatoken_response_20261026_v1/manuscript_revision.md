# Completed evidence and manuscript changes

Keep the title **NeST-Com: Fixed-Budget Next-Scale Token Communication**.
The released earlier manuscript is historical; these replacement paragraphs and
the six-system Fig. 2 are the current A/B/C evidence package. No new study is pending.

## Attribution paragraph for related work

Ada-TokenCom combines prefix entropy coding, receiver-side tail generation and
joint prefix/MCS selection under a long-term average resource constraint. Our
comparison adapts this class-conditioned approach to a fixed per-image frame:
a frozen sender classifier predicts the class, its 10-bit ID is transmitted and
protected within the budget, and an independently terminated prefix is chosen
to fit the available payload. We retain the official LlamaGen-L/VQ16 components
and report the adaptation separately from the original controller and ARQ setting.

The multiscale next-scale representation and known-token overwrite mechanism
come from VAR. NeST-Com studies the granularity of within-scale endpoints,
actual terminated-length fitting and prefix-constrained reconstruction together
under a per-image hard symbol budget. WHOLE/PARTIAL and the controlled nested
endpoint experiment isolate internal design effects; the external comparison
compares entire systems with different tokenizers, priors and image decoders.

## III.B replacement result paragraph

The six-system comparison shows a quality tradeoff rather than uniform
dominance. Relative to the predicted-class Ada fixed-frame adaptation,
NeST-Com improves common500 PSNR by 4.116, 0.455 and 1.226 dB at 4, 10 and
19 dB, respectively. At 4 dB, NeST-Com also has lower LPIPS and DreamSim,
whereas Ada has higher CLIP-I. At 10 dB, Ada has lower LPIPS and DreamSim and
higher CLIP-I despite its lower PSNR. At 19 dB, NeST-Com retains higher PSNR
and CLIP-I and lower LPIPS, while Ada has lower DreamSim. The corresponding
direct source-paired intervals are reported without inferring differences
from overlap between marginal intervals.

On Kodak, NeST-Com improves all four metrics at 4 dB and PSNR at 10 and 19 dB.
Its LPIPS advantage at 19 dB also has a nonzero paired interval. The other
perceptual/CLIP contrasts at 10 or 19 dB span zero; this is uncertainty, not
evidence of equivalence. The two populations are reported separately.

## III.C mechanism paragraph

With fixed models, MCS, maximum endpoint and fitting rule, the quarter grid
improves 10-dB PSNR by 0.660 dB over whole-scale endpoints. Refining to eighths
adds 0.123 dB, but the DreamSim and CLIP-I increments span zero. We therefore
retain the frozen main grid and do not claim quarter-scale endpoints are
globally optimal. The receiver ablation separates direct residual correction
from propagation through finer scales. At the final scale, the latter term is
exactly zero, a boundary case included in the reported results.

## Exact claim and figure edits

- Replace “five methods” with “six systems” only for the new N1024 quantitative
  comparison; the confirmed qualitative figure still shows its original five.
- Replace Fig.2 with figures_v1/four_metrics_common500.pdf. Use the separate Kodak
  figure as supplementary evidence. Keep ARPC and the original five curves.
- Remove unrestricted “lowest DreamSim/highest CLIP-I at all SNRs.” At common500
  10/19 dB Ada has lower DreamSim; at 4/10 dB it has higher CLIP-I.
- Keep NeST's non-leading PSNR versus Swin/BPG, Latent's lower 4-dB LPIPS,
  and ARPC's favorable N2048 19-dB DISTS. No N2048 experiment or timing was added.
- Do not attribute the CLIP gap to class conditioning without an ablation.
  Do not interpret four metrics as four independent confirmations.
- Preserve jointly calibrated WHOLE/PARTIAL and utilization evidence. The new
  controlled A experiment is a separate existing100-source diagnostic, not a new blind test.

## Sources

- Ada-TokenCom: https://arxiv.org/html/2608.28086v1
- Official LlamaGen: commit ce98ec41803a74a90ce68c40ababa9eaeffeb4ec;
  sampling defaults explicitly adapted because the Ada paper does not disclose
  the full generation configuration.
- Experiment plan: experiments/adatoken-response-20261026/C_PREDICTED_CLASS_PLAN.md.
- Exact current numbers: inputs/{common500,kodak24}/paired_differences.csv,
  combined_summary.csv and original provenance.json; no new statistics in this document.
