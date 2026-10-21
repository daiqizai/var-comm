## Fig.3 selected sample update — 2026-10-21

The second row now uses the original white long-haired dog21 at N1024,
10 dB, timing_case s10_i0021_r3. Pastries95 remains first. Both rows retain
complete views, original fixed outputs and LPIPS-Alex values. Five dog
DreamSim distances were scored with the same frozen protocol; all five
Pastries distances are reused. Labels remain LPIPS-Alex / DreamSim, both
lower-is-better, three decimals. Current full-precision source:
`results/figure3_dog_20261021_v1/delivery/figure3_metrics.csv`.
The shopping-cart results remain preserved as historical candidate material.
Word, LaTeX, captions and the current outline refer to the same two rows.

The following metric-supplement record describes the preceding sample set.

## Fig.3 metric-label update — 2026-10-21

Fig.3 retains the fixed 10-dB two-row full-view pixels and r3 reception records.
Labels now show LPIPS-Alex / DreamSim (both lower-is-better), three decimals.
The ten new DreamSim distances are scored from hash-verified native float arrays
with the main evaluation’s frozen ensemble, FP32 and 224-pixel bicubic-antialias
preprocessing. Existing LPIPS values are copied verbatim. Full precision and
file/array identities: `figure3_metrics.csv`; authoritative scored delivery:
`results/figure3_dreamsim_20261021_v1/delivery/`. No reconstruction, channel,
other metric or bootstrap was repeated. Fig.1, Fig.2 and Tables I–III are unchanged.

# WCL revision outline and figures

Updated from confirmed parent `46ce02f136362709dfe033f2e09bfad0fb72cc7d`.
Frozen quantitative evidence and completed ARPC sensitivity are unchanged.

- Fig.1 simplifies the user's original PPTX, retaining its schematic bird
  images and multiscale grids. Two panels show system flow and boundary-scale
  prediction/overwrite/continuation. Utilization remains in Table I only.
- Fig.2 remains the frozen 1×4 five-method main comparison; all 60 means/CIs
  are unchanged.
- Fig.3 is 2×6 full-view 10-dB Pastries95 / White long-haired dog21, 7.16×3.00 inches.
  No ROI boxes or crop rows. Original image bytes and whole-image scores remain.
- Word, LaTeX and captions now reference the same Fig.1 and Fig.3 exports.
- Table I–III and completed ARPC sensitivity remain unchanged.

The new four-source 13-dB check is separate:
`../figures/selected_examples_13db_20261021_v1/` and
`../../results/selected_examples_13db_20261021_v1/delivery/`.
Its lower-grid digital policy is declared before execution; it does not select
a new common500 policy or silently replace the 10-dB paper examples.

Editable sources: `figure1.pptx`, `scripts/figure1.mjs`, `draw.py`, `style.json`.
The diagram PDF has vector modules/text/grids and two original schematic raster
thumbnails. Fig.3 has unmodified raster photos with vector text. The user-supplied
PPTX and PDF remain preserved under inputs/ and latex/images/ respectively.
Previous outputs remain in Git history and the task's before-edit snapshot.
See visual_review.json, data_validation.json and input_manifest.json.
