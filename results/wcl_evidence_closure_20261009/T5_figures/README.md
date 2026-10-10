# WCL T5: existing evidence, new presentation

Original common500 statistics: commit `252176e041758ecb2d3e81b6fde5b587e7e17bb7`. Supplemental cached images come from the science snapshot `14b09ecd72984fb39c683d1a62bcd3221c69142f`. No original scientific file is changed; all curves use original CSV precision and existing intervals.

`fig_t5_same_prior_N1024` explicitly shows the complete-scale and partial-scale raw+VAR curves. `fig_t5_partial_minus_complete` shows their existing paired differences. Each has four independent single-column panels. PDF/SVG data plots contain vector lines, bands, markers and text; SVG text remains editable. PNG files are 600dpi.

Mechanism samples cover1/10/19dB, with16 fixed development sources in four four-row pages perSNR. External five-column samples cover13dB on all16 of the same sources. A combined `all16.pdf` accompanies each set. Source photographs in PDF/SVG remain embedded raster images with vector titles. Main-text examples use the first four entries in the historical list, without outcome selection; full pages preserve all16. These are development examples, not holdout images. Internal method IDs are kept only in metadata/code.

`plot_data.csv` preserves raw input numbers and explicit display conversions. ConvNeXt prediction agreement is displayed as percent; differences are percentage points. LPIPS difference is never sign-flipped. `sample_manifest.csv` binds original order, sourceID, method, working point, seed, cached image and actual-state identity. Public labels contain no internal P/H/MAIN/WHOLE shorthand.

The complete external10/19dB fixed16 views are not generated because matching Swin/adaptiveBPG caches were not established locally. No13dB image substitutes for them. T1 entropy-coded examples await actual new output. Swin19dB remains out of training/calibration range wherever shown in the pre-existing external curves; no19dB Swin image is shown here.

Reproduce from the repository root into a new output directory:

```text
python experiments/wcl-evidence-closure-20261009/scripts/plot_existing_t5.py --output-dir results/wcl_evidence_closure_20261009/T5_figures_replot
```

The script reads standard repository/cache locations on the original host and known local synchronized locations on this workstation. All source bindings and outputs are in `completion.json`. It imports only numpy, matplotlib and Pillow; no model, PHY or bootstrap module is called.
