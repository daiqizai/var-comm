# Captions — synchronized with the final V3 TeX

## Fig. 1
NeST-Com: (a) The protected profile and body share the fixed noisy frame; both terminals reconstruct coding probabilities from the shared frozen VAR and known coarse tokens. (b) Independently terminated within-scale prefixes fit the body information capacity; bars are schematic for the same MCS. (c) Received tokens overwrite boundary predictions, and the merged map updates the residual latent and finer-scale context while unreceived positions retain their coarse-conditioned predictions.

## Fig. 2
Source-referenced perceptual distances on common500 at $N=1024$. Points and error bars are equal-source means and pointwise 95\% source-bootstrap intervals after averaging three channel outcomes per source. Lower LPIPS-Alex and DreamSim are favorable. Ada denotes the predicted-class, fixed-frame adaptation.

## Fig. 3
Qualitative reconstructions of two development examples at $N=1024$ and 10~dB. Values denote LPIPS-Alex$\downarrow$ / DreamSim$\downarrow$. Ada denotes the fixed-frame adaptation.

## Table I
Within-scale endpoint and receiver evidence. (A) Main family-specific policies, PARTIAL minus WHOLE. (B) Fixed 100-source model/MCS/maximum-endpoint control. (C) Ordered interventions on the same 100 sources and actual prefixes. Brackets give source-paired 95\% CIs.

## Table II
Pixel fidelity and image-feature similarity on common500 at $N=1024$. Each cell is PSNR (dB)$\uparrow$ / CLIP-I$\uparrow$. Ada denotes the fixed-frame adaptation. Values are source means.

## Table III
Reduced-budget comparison on common500 at $N=512$. Differences are NeST-Com minus the fixed-frame Ada-TokenCom adaptation; brackets give source-paired 95\% CIs. Negative LPIPS and DreamSim favor NeST-Com.
