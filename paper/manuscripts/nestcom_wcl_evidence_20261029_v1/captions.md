# Final captions (same source as LaTeX/Word)

## Fig. 1

NeST-Com. (a) The protected profile specifies the endpoint and MCS; both terminals reconstruct coding probabilities from the shared frozen VAR and their known coarse tokens. (b) Independently terminated within-scale prefixes fit the body capacity; bars are schematic. (c) Received boundary tokens overwrite predicted positions. The merged map updates the residual latent and the context for finer-scale prediction. Unreceived boundary positions retain their coarse-conditioned predictions.

## Fig. 2

Six-system performance on common500 at $N=1024$. Points and error bars are equal-source means and pointwise 95\% source-bootstrap confidence intervals after averaging three channel realizations per source. Ada denotes the predicted-class, fixed-frame adaptation. Higher PSNR and lower DreamSim are favorable.

## Fig. 3

Source reconstructions at $N=512$ and 10~dB for two images selected by the preregistered source-ID rule. Columns show the original, WHOLE, NeST-Com, and the fixed-frame Ada-TokenCom adaptation. Each reconstruction uses the first registered channel repetition. Values are LPIPS-Alex$\downarrow$ / DreamSim$\downarrow$ for the displayed realization.

## Table I

Within-scale endpoint and receiver evidence. (A) Main family-specific policies; differences are PARTIAL minus WHOLE. (B) Controlled granularity on 100 sources with fixed model, MCS, maximum endpoint and fitting rule. (C) Receiver interventions on the same 100 sources and actual prefixes. Brackets give source-paired 95\% CIs; populations and conditions are as stated.

## Table II

Perceptual distance and image-feature similarity on common500 at $N=1024$. Each cell is LPIPS-Alex$\downarrow$ / CLIP-I$\uparrow$. Ada denotes the fixed-frame adaptation. Values are source means.

## Table III

Reduced-budget comparison on common500 at $N=512$. Differences are NeST-Com minus the fixed-frame Ada-TokenCom adaptation; brackets give source-paired 95\% CIs. Positive PSNR and negative DreamSim favor NeST-Com.
