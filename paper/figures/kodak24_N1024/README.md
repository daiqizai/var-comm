# Kodak24 paper figures

These figures read the completed 48 summary rows and 36 source-paired differences from `results/generalization_kodak_20261009/analysis_v1`. No bootstrap, model call, channel draw, policy selection, or change to original results occurs. Both figures preserve Swin's 19 dB point using a hollow marker and a dashed 10-to-19 dB segment because that point is outside its training/calibration range.

All 24 predetermined center crops, three noises per source, and failure outputs are included. DINO-L uses dinov2_vitl14_cosine. ConvNeXt is source-prediction agreement, displayed as percent; paired increments are percentage points. LPIPS differences remain proposed minus reference. The small crop benchmark and pointwise intervals do not establish universal superiority or training-data non-overlap.

Files: main and paired 2x2 figures, each vector PDF/editable-text SVG/600-dpi PNG, plus captions and actual plotting rows. The original analysis exports and completion are preserved. Figures still require actual PNG visual inspection, recorded separately.

Reproduce into a new directory:

`python experiments/generalization_kodak_20261009/plot_only.py --data results/generalization_kodak_20261009/analysis_v1 --out NEW_FIGURE_DIRECTORY`
