# Reselected reconstruction examples

Development examples at N=1024 and SNR=13 dB. Display rows use source indices0,21,25,50. Source21 is retained in the second row at the user request; the other rows take the first three entries of the other available, previously verified four-source display bundle. This is a user-requested display revision, not an independently preregistered qualitative sample. No reconstruction-quality ranking or best-noise selection is used. All methods use the same source pixels and operating point. The fixed first noise is2001 for latent/Swin/HiFi and6201 for digital methods; cross-family draws differ. Swin and HiFi use the same received waveform. No inference, channel simulation or statistics were repeated. The latent and Swin checkpoints are historical40k and80k adaptations, not the final100k/200k baselines of the common500 performance plot. HiFi uses the existing unconditional diffusion-prior recovery at13 dB. These examples are development illustrations, not common500 holdout results.

Development examples at N=1024 and SNR=13 dB. Display rows use source indices0,21,25,50. Source21 is retained in the second row at the user request; the other rows take the first three entries of the other available, previously verified four-source display bundle. This is a user-requested display revision, not an independently preregistered qualitative sample. No reconstruction-quality ranking or best-noise selection is used. All methods use the same source pixels and operating point. The fixed first noise is2001 for latent/Swin/HiFi and6201 for digital methods; cross-family draws differ. Swin and HiFi use the same received waveform. No inference, channel simulation or statistics were repeated. Complete scales means the selected complete-scale prefix (m=8,K=0), not every source scale. Partial transmission uses m=8,K=9. The last two columns use identical actual received tokens and the same frozen decoder. Direct decoding sets absent residual contributions to zero, does not use codebook index0 as an unknown token, and receives no extra source truth.

Style follows the installed figures4papers scientific-figure-making skill. Original panels are retained. Both panels have PDF/SVG and600dpi PNG. Photo pixels are unchanged; labels are vector. The28 individual PNGs are retained in the local delivery package.

Reproduce with the two original verified display bundles (no models):

```sh
python scripts/plot_paper_reselected_examples.py --group1 PATH_TO_FIRST_GROUP --group2 PATH_TO_SECOND_GROUP
```
