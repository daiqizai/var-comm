# User-selected reconstruction panels

User-selected qualitative examples from the historical development set at N=1024 and SNR=13 dB. Rows show the white shaggy dog, broccoli, pastries, and shopping cart (source indices21,92,95,75; candidate numbers06,15,16,04). The user selected these illustrative examples after viewing the 16-source gallery; this is not a preregistered or representative quantitative sample. All methods use the same source image and operating point. The fixed first noise is2001 for latent/Swin/HiFi and6201 for digital methods; cross-family noise draws differ. Swin and HiFi share the same received waveform. No per-method best-noise selection or new scientific calls were made. The latent40k and adapted Swin80k checkpoints are historical baselines, not the final100k/200k models of the common500 performance figure. HiFi uses the existing unconditional diffusion prior and full posterior schedule. No official-optimal or convergence claim is made.

User-selected qualitative examples from the historical development set at N=1024 and SNR=13 dB. Rows show the white shaggy dog, broccoli, pastries, and shopping cart (source indices21,92,95,75; candidate numbers06,15,16,04). The user selected these illustrative examples after viewing the 16-source gallery; this is not a preregistered or representative quantitative sample. All methods use the same source image and operating point. The fixed first noise is2001 for latent/Swin/HiFi and6201 for digital methods; cross-family noise draws differ. Swin and HiFi share the same received waveform. No per-method best-noise selection or new scientific calls were made. Complete-scale transmission uses the selected prefix(m=8,K=0), not every source scale. Partial transmission uses m=8,K=9. Both configurations have the same modulation and coding permissions. The last two columns share actual received tokens and the frozen decoder Dc. Direct decoding sets missing residual contributions to zero; unknown tokens are not codebook index0, and no additional source-image ground truth is supplied.

Both panels have PDF/SVG/600dpi PNG; image pixels are unchanged and text remains vector. The28 native PNGs are in the local delivery. Internal method mapping and hashes remain in the script and provenance. Figures4papers typography is reused.

Reproduce with the verified input bundles:

```sh
python scripts/plot_paper_user_selected_examples.py --group1 PATH_TO_GROUP1 --group2 PATH_TO_GROUP2 --group4 PATH_TO_GROUP4
```
