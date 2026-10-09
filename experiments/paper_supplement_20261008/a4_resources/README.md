# A4 read-only export

`export_resources.py` reads the hash-pinned final common500 r6 completion and
source checkpoints, actual receiver metadata, existing four core metrics, the
published CSV and the already verified selected-configuration table. It reads
original token arrays only for offline correctness diagnostics. There are no
model, training, channel, PHY, bootstrap, or random-number calls.

Run in the existing unified metrics environment, which has NumPy/matplotlib:

```bash
outputs/UNIFIED-METRICS-20261002/environment/bin/python experiments/paper_supplement_20261008/a4_resources/export_resources.py --root /home/liulu/projects/VAR_COMM --output /home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a4_resources/v1
```

The independent output directory contains four figures in PDF/SVG/600dpi PNG,
captions, a report, full compact frame and quality rows, resource/energy tables,
failure-state counts, token-error quantiles, state-conditioned quality, reused
published intervals, provenance, and completion hashes. Existing science files
are never changed.

The all-frame means must reproduce every corresponding published four-metric
mean within absolute 1e-10 before completion is emitted. Six SNRs, three noise
realizations and all500 sources are required. The shared 7 dB physical RAW
frame belongs to both frozen families; it is not charged or inferred twice.

CRC rejection under KEEP is distinct from gray output. Wrong accepted headers
and bodies are offline truth diagnostics. A token-error rate over compared
positions is separate from a rate that counts absent planned tokens; missing
positions are not silently converted into successful delivery. Zero-count
states stay explicit. Conditional means are descriptive, have no new confidence
intervals, and are not used to modify policies.

The source-bits/quality plot is descriptive: SNR changes along with the selected
configuration. It does not isolate the causal effect of source-bit allocation.
Swin19 dB remains outside its training/calibration range. Continuous symbols
have no raw-token bit-efficiency number and are marked not applicable.

Large compact CSVs and binary figures should follow the existing repository
publication filters; this exporter does not alter those filters.

## Rendering revision 2

To revise only the layout from the completed v2 tables, use:

```bash
outputs/UNIFIED-METRICS-20261002/environment/bin/python experiments/paper_supplement_20261008/a4_resources/export_resources.py --render-existing results/paper_supplement_20261008/a4_resources/v2
```

This checks the four existing CSVs against their original completion hashes and
creates `v2/figures_r2/` without changing any original table or figure. No source
checkpoints are read and no statistics are recalculated. It removes overlapping
10/19 dB point labels, preserves exact point mappings in the original CSV,
copies existing conditional subset counts, and calls out the single 7 dB
CRC-rejected KEEP frame. A fresh `--output` can be supplied for another render.
Closed output manifests exclude runtime logs and launcher metadata, which can
change after the result is emitted.
