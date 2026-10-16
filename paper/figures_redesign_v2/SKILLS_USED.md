# Skills actually read and applied

The installed figures4papers entry was read in full at
`C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md`, together with
`references/design-theory.md`, `references/common-patterns.md` and `references/demos.md`.

Canonical examples actually read:

- [Ophthalmology trends](https://github.com/ChenLiu-1996/figures4papers/blob/main/figure_ophthal_review/plot_trend.py): restrained axes and consistent trend encodings.
- [VIGIL post-training](https://github.com/ChenLiu-1996/figures4papers/blob/main/figure_VIGIL/plot_posttraining.py): compact comparison hierarchy and shared visual grammar.
- [RNAGenScape sweep](https://github.com/ChenLiu-1996/figures4papers/blob/main/figure_RNAGenScape/plot_sweep.py): aligned multi-panel trend comparisons.
- [Cflows ablation](https://github.com/ChenLiu-1996/figures4papers/blob/main/figure_Cflows/plot_comparison_Ablation.py): grouped mechanism comparisons and interval/legend treatment. This bar example informed comparison structure, not the geometry of Figure 3.
- [Official Matplotlib asymmetric error bars](https://matplotlib.org/stable/gallery/statistics/errorbar_features.html): point estimates with asymmetric intervals; documentation/code read without executing example data.

The examples' data, oversized canvases, large presentation fonts, shading and numerical annotations were not copied. The user's v2 palette, exact manuscript widths and requested point-interval representation take precedence.

K-Dense's [scientific-visualization skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-visualization/SKILL.md) was read from the previously downloaded local copy:
`C:/Users/11946/Documents/ChatGPT/comm/.research/figures_redesign_v1/references/kd_SKILL.md`.
Its local `kd_publication_guidelines.md` and `kd_color_palettes.md` were also read. No installed copy was found in the local skill/plugin roots. All actual paths and file hashes are recorded in `skill_read_manifest.json`.

Applied review: population and checkpoint separation, direct paired intervals, precise dimensions, readable final-size typography, redundant line/marker encodings, opaque baselines, grayscale inspection, editable SVG text, searchable PDF text, no embedded raster plots, and real 600 dpi rendering. The exported PDF/PNG and browser-rendered SVG results were opened; automated checks alone were not treated as visual review. No dependencies, model environments or font bundles were installed; no external image-generation service was called.

The software citation requested by the skill is recorded here, without changing the scientific manuscript: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065). The current arXiv author/title record was checked (v2); the DOI is version independent.

The manuscript uses IEEEtran. [IEEE's graphics guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/) informed redundant encodings and grayscale review. Exact current text/column widths are measured in the included template record and rechecked in the actual-size LaTeX preview. This package awaits the user's style confirmation.
