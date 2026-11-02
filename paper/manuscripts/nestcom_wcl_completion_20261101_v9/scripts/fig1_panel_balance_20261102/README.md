# Lower-panel balance recipe

Copy `../../figures/figure1_thin_v6_clean.pptx` to a scratch folder as
`source_clean.pptx`, and copy this folder's merge script, layout plan and two
native component PPTX files there. Run `python merge_layout.py` with lxml.
The source hash is enforced. The output is `balanced_candidate01.pptx`.
ZIP timestamps may differ, but its package-part content is reproducible.

`author_layout.mjs` records the first-party artifact-tool import and native
position edits. Its runtime path refers to the actual authoring environment.
The exported draft supplies shape placements; the original package retains
font/image parts, text properties, custom vector geometry and group math.
The arrows component supplies native route shapes, completed with the same
compact attached-head geometry as the preceding cleaned variant.

Native PowerPoint exports the exact final candidate. SVG text is outlined;
use the PPTX for text editing. Panels (b)/(c) are editable groups. This recipe
does not change manuscript references, run inference, or rebuild the paper.
