# Closed A/B/C evidence and six-system figure package

Read REPORT_zh.md for results, compute/reuse counts and limitations;
manuscript_revision.md supplies attribution and result replacement paragraphs.
captions.md contains the English captions. Common500 and Kodak stay separate.

The exact original delivery is results/adatoken_response_20261026_v1/
c_predclass_formal_delivery_v1/. inputs/ contains hash-verified copies and original
remote descriptors. Old baseline CSV values were retained without recomputation.
The extraction commit is recorded in inputs/figure_inputs.json; the new publication
commit is separately verified after repository checks.

Reproduce only the deterministic figures in an existing Matplotlib/Pillow environment:
`python -B draw.py --manifest inputs/figure_inputs.json --output figures_reproduced`
This loads only closed summaries, never models, images, PHY data or bootstrap banks.
The bundled Windows runtime lacked Matplotlib; the actual successful render used
the existing remote frozen_visual_py310 environment without installing dependencies.

figures_v1 includes vector PDF, editable-text SVG, 600 dpi PNG, exact plotted CSVs,
physical-size and grayscale previews. render_validation.json is an immutable
pre-review record; actual_visual_review.json records the subsequent real inspection.
Do not overwrite historical figures or use calibration/qualification as formal results.
