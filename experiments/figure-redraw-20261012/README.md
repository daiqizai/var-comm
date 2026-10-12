# NeST-Com figure redraw, 2026-10-12

The user supplied the redraw specification and registry and then authorized
execution, including missing data. Historical results and images stay read-only.
The main data plot uses published common500 values, the two-interface paired
comparison uses published new100 intervals, and each supplemental population is
kept separate. No training, model selection, metric scoring, or bootstrap was
performed for this redraw.

## Small missing-image evaluation

`render_final_examples.py` is a one-attempt, locked entry for precisely eight
frames: sources21,92,95,75, final latent100k and Swin200k, N1024,13dB, first
continuous noise2001. The fixed source pixels are checked against the previously
selected gallery. It uses fresh source encoding and independently generated TX
waveforms/RX observations for each final model, not historical received y.
The existing frozen receiver functions only receive the actual observations and
public N/SNR. Native reconstructions, transmitted frames and observations are
saved in the user's tmp root; receipts and hashes are in
`results/figure_redraw_20261012/final_examples_v1/`.

All eight frames actually completed with actual exit0. Unchanged digital images
reuse the verified historical seed6201 observations. This is a user-selected
development illustration, not a new quantitative holdout or random example set.
No HiFi inference was needed; the old Swin80k/HiFi gallery remains explicitly
historical. Do not rerun the one-attempt entry to reproduce figure layout.

## Render and check

```sh
python scripts/plot_paper_redraw_v1.py
python experiments/figure-redraw-20261012/validate_delivery.py
```

Output: `paper/figures/redraw_v1/`. Every delivered group includes plotted rows,
English caption, provenance, vector PDF, editable-text SVG and600dpi PNG.
Native photographs remain256x256; export DPI does not imply improved resolution.

MT01 and S14 remain dependent on actual six-method timing. The original admission
requires all GPUs free of other compute jobs and other repository work stopped.
The owner again returned WAIT without scientific calls. No foreign process was
stopped and no global GPU settings were changed. Historical device timings are
not substituted. The current source tree has tables/captions but no full paper
manuscript; the included physical-size review PDF is not claimed to be a compiled
five-page submission.

S01 and S17 are optional historical compatibility checks, not silent additions
to the main experimental budget. Old SourceA DINO is not renamed DINOv2-L.
The completed metric reference suite supplies LPIPS/DINO-L but no ConvNeXt;
S13 displays only compatible existing metrics.
