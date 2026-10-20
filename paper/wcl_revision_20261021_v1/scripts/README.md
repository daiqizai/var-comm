# Deterministic authoring

`../draw.py` recreates Figures 2/3 and all three tables from the pinned inputs.
`build_outline.py` recreates Word from the saved outline sections and figure PNGs.
The authoring runtime is the Codex bundled Python with python-docx.

For the editable system diagram, set NEST_FIGURE_WORK_DIR to a writable scratch
directory outside this deliverable, run figure1.mjs with the bundled artifact-tool
Node runtime, then normalize_diagram.py. The native PowerPoint objects, text and
connectors remain editable. Copy the reviewed normalized PPTX to figure1.pptx and
export as PDF using native PowerPoint (SaveAs format 32). Render that PDF at
600 dpi with Poppler. This path was used after the bundled artifact PDF exporter
failed on a missing standard-font alias; no runtime package was modified.

The input PPTX is preserved under inputs/. Output dimensions are 7.16 by 3.915 in.
Run the presentation package/layout validators before delivering a changed PPTX.
The final font/geometry check is summarized in visual_review.json; the native
validator receipt remains in the task's private review directory.

These scripts do not launch or rerun any scientific experiment.
