# Figure 2 style previews — awaiting style choice

Only Figure 2 A/B is rendered in this batch. Historical figures and experimental
outputs are untouched. The basis is main `9e59f62e6b803b79564d57beb905d59dbe1e1ece`.

| Version | Physical size | Layout |
|---|---|---|
| A | 181.353 × 124 mm | Compact 2 × 2; shared two-row legend above |
| B | 181.353 × 141 mm | Taller panels; shared two-row legend below |
| Individual panels | 88.568 × 64 mm | One column; use the shared method legend |

The exact manuscript uses `\documentclass[journal]{IEEEtran}`, with no width
override. A class-only TeX measurement gives textwidth 516 pt, columnwidth 252 pt
and column separation 12 pt (TeX points). See `template_geometry.json`.
The supplied manuscript PDF was not rebuilt. An initial missing-class attempt
is preserved in the runtime directory; the class was downloaded only into that
temporary directory. No training environment was installed or upgraded.

Both alternatives use identical data, axis limits, ticks, 95% intervals, font
sizes, colors, line styles, markers and display transformations. Only panel size,
spacing and shared-legend position change. Font: DejaVu Sans; labels 9 pt,
ticks/legend 8 pt, panel letters 9.5 pt, curves 1.5–1.7 pt. All methods are opaque.
The x coordinates are exactly 4, 10 and 19 dB; straight segments only.

## Deliverables

- `previews/fig2_A.*`, `previews/fig2_B.*`: vector PDF, editable SVG, 600 dpi PNG.
- `previews/fig2_{A,B}_{psnr,lpips,dreamsim,prediction_agreement}.*`: individual
  panels in the same three formats. Their standalone layout is identical; no
  duplicate statistics were calculated.
- `plot_inputs/fig2_plot_data.csv`: the 60 exact summary positions used in both versions.
- `plot_inputs/common500_all_metrics_versioned.csv`: existing metrics with both entropy
  families corrected at 19 dB, including the unplotted whole-scale comparator.
- `plot_inputs/t1c_replacement_log.csv`: all 14 replaced metric positions and old bounds.
- `input_manifest.json`, `DATA_VERSIONS.md`: file hashes and scope/version mapping.
- `methods.json`, `style.json`, `prepare_data.py`, `plot_figure2.py`: reproducible
  plotting and normalization only; no experiment/statistics modules imported.
- `caption.tex`, `captions.md`, `review/`: caption and actual visual/container checks.
- `SKILLS_USED.md`, `skill_read_manifest.json`: actual skill paths, hashes and uses.

PDF/SVG contain paths and text with zero raster images. SVG text remains editable
and uses an installed font; no font files are distributed. PNGs are rasterized
from the vector artists at 600 dpi, not upscaled from earlier figures. Separate
96 dpi and screen views in `review/` are inspection aids, not high-resolution
deliverables or a claim about any monitor's physical calibration.

## Reproduction

Use an existing Python environment with Matplotlib (rendered with 3.10.8).
No dependency installation is required for this delivery. Set `MPLCONFIGDIR`
outside the source tree and hide CUDA. `prepare_data.py --root REPO --geometry
GEOMETRY_JSON` verifies the closed tables and creates the plotted table;
`plot_figure2.py --output NEW_OUTPUT_DIRECTORY` renders to a new directory.
The output directory must not already exist. `audit_exports.py` uses existing
Pillow/pypdf packages for container inspection and separate downsampled review
views. A deterministic render's initial JSON metadata serialization failure was
fixed by converting tick scalars to native floats; its prior outputs remain in
the runtime directory. Neither that failure nor the redraw invoked science.

No training, channel simulation, VAR/diffusion recovery, scoring, bootstrap,
policy/checkpoint selection or timing was run. Scientific backups stay deferred.
Figure 1, Figure 3, Tables I/II and supplementary figures await the user's style
choice. This preview does not assert final journal submission compliance.
