# Existing cal100 analysis and proposed N2048 validation

This directory is CPU-only analysis / plotting source and input evidence, not a
scientific execution entry point. Current user authorization ends at the plan.
Do not run channel, neural, scoring, training or bootstrap work from this package.
Preserve the completed N1024 comparison at basis HEAD
`fbc17ae300e46bfdab56c22f989752545a5c9b3c`.

`collect.py` reads closed metadata and hashes existing complete source objects
and selected image assets. `analyze.py` joins the original 700 cal100 scores and
lengths and performs integer capacity accounting. `plot.py` draws the original
descriptive means, not new scores or intervals. The 96-bit reassembly record is
a conservative proposed protocol, not an already implemented/qualified codec.

Reproduction (output must be a new directory):

```text
python -B analyze.py --inputs inputs --output NEW_OUTPUT
python -B plot.py NEW_OUTPUT
```

Plot requires an already available matplotlib runtime. No environment install
or upgrade is needed. The existing remote frozen_ldpc_py311 runtime was used;
CUDA is hidden and no scientific package/model is imported.

Actual skill paths: `C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md`,
its `references/design-theory.md`, `common-patterns.md`, `demos.md`, `tutorials.md`;
existing local canonical demo copy
`.research/figures_redesign_v1/references/figure_ophthal_review__plot_trend.py`.
Borrowed aligned trend panels, restrained spines and readable vector labels;
scaled to 180 x 67 mm and the user's blue palette rather than the demo's canvas.

The input audit had two read-only harness failures (encoding spelling, then
an incorrectly named metadata key), both preserved locally. Corrected audit
completed; no science was attempted by any audit invocation. Local plotting
stopped before plotting because the local runtime lacked matplotlib. It was
then run with the already installed remote runtime, without installation.

Results: `results/arpc_n2048_feasibility_20261019_v1/PLAN_zh.md`, curves and CSVs.
The proposed_scope explicitly forbids automatic science/next stage execution.
