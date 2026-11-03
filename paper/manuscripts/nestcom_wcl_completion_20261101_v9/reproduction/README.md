# Reproduction records

Start with [the manuscript companion checklist](../REPRODUCTION_CHECKLIST.md).

`phy_layouts.csv`, the pilot and confirmation score CSVs, and their two verbatim LaTeX table sources retain the numerical detail moved from S1. `environment_device_records.json` groups exact existing fields without replacing the original records. The inventory covers every included file except itself and the ZIP container.

All other metadata keep their original `inputs/` or `reproduction_logs/` paths. The local/PFS delivery includes a ZIP; Git stores the individual files and [the deterministic builder](../scripts/build_reproduction_companion.py), because archives are excluded by repository policy. Weights, datasets and native scientific image caches are not included.
