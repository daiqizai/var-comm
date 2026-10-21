# Deterministic authoring

`../draw.py` recreates Figures 2/3 and Tables I–III from pinned inputs.
Its Figure 3 uses 12 full views, with no ROI boxes or cropped rows.
`build_outline.py` assembles Word from saved outline sections, tables and PNGs;
the current editable Word file was updated in place to preserve existing work.
`write_materials.py` refreshes the Markdown outline from `outline_sections.json`.
Captions and the existing LaTeX source are edited directly, not overwritten by
an obsolete whole-document generator.

For the diagram, set NEST_FIGURE_WORK_DIR to a separate scratch directory and
run figure1.mjs with the installed artifact-tool Node runtime, then
normalize_diagram.py. It imports inputs/system_diagram_v3_entropy.pptx and
reuses the original diagram_thumbnail.png. Native grids, modules and text stay
editable. Validate the resulting PPTX package/layout before copying it to
figure1.pptx. Export PDF with native PowerPoint SaveAs format 32, then render
600-dpi PNG with Poppler. This exact route produced the delivered figure.
Final dimensions are 7.16 × 3.1683 inches; auxiliary type is at least 7.26 pt.
The two original bird thumbnails are raster; modules, grids, arrows and text
are vectors. No bitmap-only PDF is represented as a vector diagram.

The scientific results and figure assets are pinned in input_manifest.json,
data_validation.json and delivery_manifest.json. These authoring scripts do
not launch science or resample statistical intervals.
