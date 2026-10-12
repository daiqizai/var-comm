# NeST-Com paper figure redraw v1

All numerical figures read full-precision published CSV values and existing intervals. No new training, model selection, scoring or bootstrap is performed by this script. Every figure folder contains PDF, editable-text SVG, 600dpi PNG, grayscale proof, plot_data.csv, caption.tex and provenance.json.

M04 separately binds eight actual final-checkpoint inference frames, preserving the chosen sources and first-noise rule. Old HiFi examples remain historical and are not relabeled as final baselines.

MT01 and S14 require actual same-environment timing under the original exclusive-resource gate; no publication-ready numerical table is fabricated while waiting. The review PDF contains figures at their specified physical width; it is a figure proof, not a compiled five-page manuscript.

Reproduce from the repository root:

```sh
python scripts/plot_paper_redraw_v1.py
```
