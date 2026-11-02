# Fig.1 supplied-PPTX cleanup recipe

The user-edited source is `../../inputs/figure1_thin_v5_user_20261102.pptx`.
Work in a separate scratch directory: copy that source as `user_edited_source.pptx`
and copy these scripts, `route_plan.json`, and `arrows_native.pptx` there.
Run `python patch_native.py` with lxml installed. It produces `clean_candidate02.pptx`
and a structural preservation audit. Zip timestamps may differ; compare package
parts for deterministic reproduction.

`arrows_native.pptx` is the small editable arrow component deck actually authored
with the first-party artifact-tool. `build_routes.mjs` records the authoring call
and its original local runtime path; it is not necessary to rerun it to apply the
patch. For another environment, point its createRequire path at the installed
artifact-tool runtime. No image-model or scientific computation is involved.

The final deck keeps all original package parts except slide1.xml. Text, group
transforms, images, and embedded fonts retain the user's source identity. Each
replacement arrow contains a single shaft path and a filled head path inside one
native shape. The PDF was exported from the exact validated candidate using
PowerPoint 16. SVG text is outlined; edit text in PPTX. Do not change the source
positions or substitute this editing variant into a manuscript without updating
its figure references and rebuilding the documents.
