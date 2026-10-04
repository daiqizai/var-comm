# Prior-aware grouped MCS, V1

This study runs alongside the existing M1 N2048 task. It does not train a model.
The user-supplied execution plan is preserved in `EXECUTION_PLAN.md`; the seven
subsequent user amendments are registered in `protocol.json` and
`model_validation_rules.json`. Current user authorization includes normal Git
publication and overrides the older attachment's publication-approval sentence.

## Experiment contract

- Main study: N1024. N2048 enumeration is resource-only until the dual-metric
  replication gate passes. This is independent of the previously authorized M1
  N2048 experiment, which continues unchanged.
- Frozen unconditional class1000 VAR, deterministic greedy argmax, same Encoder,
  VQ, latent F and Dc. No generation sampling or true class side information.
- 68-use QPSK header carries a paid 12-bit final public profile ID. Each body
  group uses CRC16 and a real Sionna 2.2.0 5G LDPC codeblock, with supported actual
  rate matching. Unsupported configurations remain explicitly excluded.
- One or two groups; the receiver uses only the contiguous CRC-accepted prefix.
  Wrong accepted bits are actually rendered. Rejected hard decisions are discarded.
- B0–B4 plus matched (m,K,j) controls, strongest baseline and a frozen distinct
  validation candidate. B0 includes the same source and full-budget-fill grid.
- Select with DINOv2 ViT-L/14 on calibration only. ConvNeXt-Tiny V1 source
  prediction agreement is an independent, development-only validation metric.
- Actual link: original100 development sources × seeds2001/2002/2003 ×
  4/7/10/13dB. Add frozen P1024 inference at10dB. No added samples or holdout.

## Execution and scheduling

The isolated `outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment` is Python3.11
with Sionna-no-rt2.2.0/PyTorch2.11.0. Original visual inference uses the existing
Python3.10 unified-metrics environment. Interchange is through sealed JSON/NPZ.

`qualify_phy.py` checks actual LDPC/CRC/header/constellation roundtrips.
`profiles.py` enumerates actual encodable resources. `phy_lookup.py` partitions
the N1024-only physical table into eight deterministic shards; `coarse_dispatch.py`
runs four CPU workers at a time at reduced scheduling priority, with checkpoint
continuation. The briefly measured eight-worker setup slowed the older GPU task,
so it was not retained. Every CPU worker has CUDA visibility disabled.

`study_owner.py` waits for the existing M1 N2048 publication receipt before GPU
qualification and a two-calibration-source source-Q cost pilot. It then chooses
the preregistered full1000 search or, if estimated cost exceeds48h, first300
screening followed by full1000 verification of a frozen shortlist. The latter
does not establish a full-grid1000 optimum. Quality and coarse PHY work overlap;
finalist refinement and actual development wait for both.

Actual encoding, received-state rendering, image metrics, P controls, uncached
receiver timing, model validation, figures and publication have separate receipts.
A source checkpoint permits exact resume; unknown failures stop for review.
An engineering test pass is not a completed image-quality experiment.

## Validation and stopping

Independent source bootstrap uses100 original sources after averaging three
noise realizations,10,000 draws and seed20261002. ConvNeXt never selects or
refines a policy. Model validation uses frozen diagnostic clean states of the
same100 development sources, never calibration means masquerading as matched
development predictions. Its engineering equivalence bands are stated explicitly.

N2048 UEP replication is eligible only after nondegenerate B3−B0 is positive in
both DINO-L and independent ConvNeXt paired intervals at at least two SNRs, and
registered model validation passes. An uncertain or negative result is delivered
without additional samples. No same-scale adapter, M2 change, loss change or old
queue is launched by this study.

Reports distinguish cached offline throughput from independently measured
uncached receiver latency. QPSK has exact frame E=2N. Fixed 16QAM shares average
symbol power, and its actual frame energy is reported separately.
