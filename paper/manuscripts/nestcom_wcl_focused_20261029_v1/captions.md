# Captions (canonical LaTeX wording)

## Fig. 1

NeST-Com. (a) The protected profile specifies the endpoint and MCS; both terminals reconstruct coding probabilities from the shared frozen VAR and their known coarse tokens. (b) Independently terminated within-scale prefixes fit the body capacity; bars are schematic. (c) Received boundary tokens overwrite predicted positions. The merged map updates the residual latent and the context for finer-scale prediction. Unreceived boundary positions retain their coarse-conditioned predictions.

## Fig. 2

Performance on 500 ImageNet validation images at $N=1024$. Error bars show 95\% source-bootstrap confidence intervals. Ada denotes the fixed-frame adaptation.

## Fig. 3

Example reconstructions at $N=512$ and 10~dB. The top row shows full images and the bottom row enlarges the same marked region. Values denote full-image PSNR$\uparrow$ / LPIPS$\downarrow$. Ada denotes the fixed-frame adaptation.

## Table I

Within-scale endpoint and receiver evidence. (A) Main family-specific policies; differences are PARTIAL minus WHOLE. (B) Controlled granularity on 100 sources with fixed model, MCS, maximum endpoint and fitting rule. (C) Receiver interventions on the same 100 sources and actual prefixes. Brackets give source-paired 95\% CIs; populations and conditions are as stated.

## Table II

Perceptual distance and image-feature similarity on common500 at $N=1024$. Each cell is LPIPS-Alex$\downarrow$ / CLIP-I$\uparrow$. Ada denotes the fixed-frame adaptation. Values are source means.

## Table III

Reduced-budget comparison on common500 at $N=512$. Differences are NeST-Com minus the fixed-frame Ada-TokenCom adaptation; brackets give source-paired 95\% CIs. Positive PSNR and negative DreamSim favor NeST-Com.
