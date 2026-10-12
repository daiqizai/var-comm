# figures4papers style trial

Uses the scientific-figure-making skill from https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making (SKILL.md, design-theory.md, api.md and common-patterns.md). Adapted conventions: FigureStyle, apply_publication_style, finalize_figure, deep-blue key method, warm-red/teal/violet comparators, clean spines, dedicated legend area, consistent line weights and vector text. The user's 7-inch double-column and 8–9pt final-size typography requirements take precedence over the skill's large-canvas defaults. Different markers/line patterns retain grayscale readability. Shaded bands use published intervals, not estimated deviations.

This is a separate layout trial, preserving all original figure files. plot_data.csv is an exact byte copy of the preceding96-row common500 main-figure CSV. No new scientific calls, statistics, point selection, cropping, smoothing or image enhancement. Four independent quantitative panels accompany the2x2 composite. The qualitative panels retain all four fixed sources and the same256x256 display pixels. Only layout and public labels change.

IMPORTANT: performance curves use final latent100k/Swin200k, while the HiFi-matched qualitative panel is historical development with latent40k/Swin80k. It is explicitly labeled and cannot be described as final-baseline holdout reconstruction. Digital mechanism examples are also development. Fixed first noise2001 for latent/Swin/HiFi and6201 for digital; cross-family noise draws differ. Same-SNR HiFi only; no substitutions. Original captions and source/noise/receiver proofs accompany this trial.

Reproduce from the repository root with matplotlib, NumPy and Pillow:

```sh
python scripts/plot_paper_figures4papers.py --individual-dir PATH_TO_EXISTING_GROUP02_INDIVIDUAL_PNGS
```

Default individual PNG path: paper/figures/hifi_development_N1024_13dB_group02/individual. These display assets remain in the local figure backups. No inference is required. PDF/SVG quantitative curves are vector; photographs are embedded rasters, with vector labels. PNG previews are600dpi.
