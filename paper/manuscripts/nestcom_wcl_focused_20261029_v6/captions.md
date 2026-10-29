# Captions — synchronized with current V6 TeX

## Fig. 1

NeST-Com: (a) The protected profile and body share the fixed noisy frame; both terminals reconstruct coding probabilities from the shared frozen VAR and known coarse tokens. (b) Candidate prefixes share the coding traversal and are finalized independently for actual-length fitting. Bars are schematic for the same MCS. (c) Received tokens overwrite boundary predictions; the merged map updates the residual latent and finer-scale context.

## Fig. 2

Source-referenced perceptual distances on common500 at $N=1024$. Points and error bars are equal-source means and pointwise 95\% source-bootstrap intervals after averaging three channel outcomes per source. Lower LPIPS-Alex and DreamSim are favorable. Ada denotes the predicted-class, fixed-frame adaptation.

## Fig. 3

Qualitative reconstructions of two fixed development examples at $N=1024$ and 4~dB. Values denote LPIPS-Alex$\downarrow$ / DreamSim$\downarrow$ for each displayed realization. Ada denotes the predicted-class, fixed-frame adaptation.

## Table I

Within-scale endpoint and receiver evidence. (A) Main common500 policies, PARTIAL minus WHOLE. (B) Fixed 100-source model/MCS/maximum-endpoint control. (C) Ordered interventions on the same 100 sources and actual prefixes. Brackets give source-paired 95\% CIs.

## Table II

Pixel fidelity and image-feature similarity on common500 at $N=1024$. Each cell is PSNR (dB)$\uparrow$ / CLIP-I$\uparrow$. Ada denotes the fixed-frame adaptation. Values are source means.

## Table III

Reduced-budget perceptual differences on common500 at $N=512$. (A) NeST-Com minus WHOLE with shared models. (B) NeST-Com minus the fixed-frame Ada-TokenCom adaptation. Brackets give direct source-paired 95\% CIs; negative values favor NeST-Com.
