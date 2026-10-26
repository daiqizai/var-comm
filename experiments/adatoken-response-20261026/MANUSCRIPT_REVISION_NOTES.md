# Manuscript revision tasks after the Ada concern

This is a bounded revision checklist, not a new full manuscript or a claim
that the Ada formal evaluation has finished.

## Attribution and method scope

1. Attribute prefix entropy coding, receiver tail generation and joint
   prefix/MCS selection to the related token-communication literature,
   including [Ada-TokenCom](https://arxiv.org/html/2608.28086v1).
2. Attribute the next-scale multiscale representation and conditional
   known-token overwrite to VAR. Describe NeST-Com's implemented contribution
   as within-scale endpoint granularity coupled to actual terminated-length
   fitting under a per-image hard symbol budget and its receiver integration.
3. Distinguish Ada's long-term average resource/ARQ setting from this fixed
   per-image N1024 adaptation. The class comes from a fixed sender classifier
   and costs10 transmitted bits, protected and included in the body length.
   Do not call it free class information or ImageNet ground truth.

## Evidence and placement

- Keep the existing jointly calibrated WHOLE/PARTIAL evidence. Its endpoint
  families, targets and fallback were separately calibrated; it is not the
  same experiment as the new controlled nested grids.
- A fixes the models, MCS, maximum endpoint and fitting rule. Report all four
  grids and all three SNRs, including the10dB eighth-minus-quarter DreamSim
  and CLIP-I intervals that cross zero. Do not claim quarter is globally
  optimal or replace the frozen main setting from these observed scores.
- B is secondary explanatory evidence. Separate direct correction from
  subsequent propagation. Preserve the19dB zero propagation increment and
  the same-device numerical amendment; no general representation-order
  causal claim follows from the ablation.
- C supplies the nearest external fixed-frame reference after formal
  completion. Retain ARPC. Add the qualified C results to Fig.2 only after
  calibration freezes the strategy and formal common500/Kodak results close.
  Do not substitute qualification images or calibration scores.
- Do not automatically widen Fig.3. Preserve its confirmed examples and
  realization-specific scores.

## Claims requiring recheck after C

Recheck the method count, every best/lowest/highest claim, the abstract and
the discussion of pixel/perceptual/semantic tradeoffs. Report all predefined
outcomes whether favorable, unfavorable or uncertain. Use direct source-
paired intervals, not overlap between marginal intervals. System differences
include tokenizer/prior/decoder differences and cannot all be credited to
within-scale endpoints. The existing100-source diagnostic is not a new blind
test; common500 and Kodak must remain separate populations.

The official LlamaGen generation configuration is frozen from
[`sample_c2i.py`](https://github.com/FoundationVision/LlamaGen/blob/ce98ec41803a74a90ce68c40ababa9eaeffeb4ec/autoregressive/sample/sample_c2i.py).
It is an explicitly documented adaptation because the Ada paper does not
specify a complete sampling configuration. Source CDFs use the full-vocabulary
conditional branch; tail sampling alone uses the frozen CFG/top-k route.
