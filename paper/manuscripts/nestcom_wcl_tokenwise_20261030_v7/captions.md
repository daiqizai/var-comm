# Captions — V7

Extracted from the current TeX. Figure numbering and supplement section numbering are distinct. Artifact inspection is recorded separately.

## Fig. 1

NeST-Com. (a) The protected profile specifies the endpoint and MCS; both terminals reconstruct coding probabilities from the shared frozen VAR and known coarse tokens. (b) Independently terminated token-wise prefixes fit the body capacity; bars are schematic. (c) Received boundary tokens overwrite predicted positions. The merged map updates the residual latent and finer-scale context. Unreceived boundary positions retain their coarse-conditioned predictions.

## Fig. 2

Six-system reconstruction on common500 at $N=1024$. Points and bars are equal-source means and pointwise 95\% source-bootstrap intervals after averaging three channel realizations per source, including failures. Ada is the predicted-class, fixed-frame adaptation. Lower DISTS and DreamSim are favorable.

## Fig. 3

Selected qualitative examples at $N=1024$ and 7 dB. Columns show source, WHOLE, fixed-frame Ada, and NeST-Com. These examples illustrate visible advantages rather than a random sample. Each uses the fixed first channel realization for that method. Values are DISTS$\downarrow$ / DreamSim$\downarrow$ for the displayed full-field output.

## Table I

Within-scale transmission evidence. (A) NeST minus WHOLE under the frozen family policies. (B) Only endpoint granularity changes under matched model, MCS, ceiling, and noise. (C) Ordered receiver interventions on the actual NeST prefixes. Brackets are source-paired 95\% intervals; each block uses its stated population.

## Table II

Auxiliary quality and processing costs at $N=1024$, 10 dB. Quality uses common500; timing uses 32 sources, one RTX PRO 6000, two CPU cores, batch one, and complete TX/RX processing boundaries. BPG uses CPU only.

## Table III

Reduced-budget reconstruction on common500 at $N=512$. Means include all failures; PSNR differences use direct source-paired 95\% intervals. All five metrics and paired comparisons are in Supplement~S2.
