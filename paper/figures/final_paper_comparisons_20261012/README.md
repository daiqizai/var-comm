# Main-paper comparison figures

The new main figure compares the proposed raw partial-scale method with selected latent continuous JSCC (100k), adapted SwinJSCC (200k), and adaptive-resolution BPG + LDPC. It uses only the original common500 cohort at N1024, six numeric SNRs, four metrics, and existing source-level 95% intervals. No training, inference, channel simulation, policy selection, or bootstrap was performed. Single-column versions use separate layouts.

Original main results: commit 252176e041758ecb2d3e81b6fde5b587e7e17bb7; final selected-baseline supplement: 3ccfe6e9fb2c914f1ac999733e61ccd3d0493015. Exact CSV paths, hashes, identity mappings and unrounded numbers are in plot_data.csv. Proposed values were also checked against the unchanged anchors in the adaptive-BPG comparison. New100 results are not mixed into this figure. The adaptation and range changes are specified in captions.tex.

Reproduce from the repository root:

```sh
python scripts/plot_paper_final_comparison.py
```

PNG previews are 600 dpi; PDF curves are vector; SVG text remains editable.

## Qualitative panels for the paper

The existing five-column and four-column panels are at ../hifi_development_N1024_13dB_group02/. These are fixed development sources 4,21,24,29 at N1024/13dB, not common500. They use historical latent 40k and Swin80k models, including the HiFi reconstruction conditioned on that same Swin reception. They must not be presented as images from the final 100k/200k baselines. Use as explicitly historical adapted-baseline illustrations or supplementary material; they are not a matched qualitative validation of the final main curves. No newer matched HiFi panel is asserted here.

All methods share source pixels and working point. The fixed first noise is used: 2001 for latent/Swin/HiFi and 6201 for digital methods; cross-family draws are not identical. No per-method best-noise selection. Mechanism panels retain the same received partial tokens and decoder, zero absent residual contribution (not codebook index 0), and no extra source truth. Original provenance, captions and image hashes remain alongside those files. Existing mainraw64_holdout500 fig03/fig04 are the unchanged paired mechanism ablations.
