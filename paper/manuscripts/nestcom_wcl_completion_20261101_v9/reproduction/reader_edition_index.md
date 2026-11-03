# Reading edition and complete evidence

The current supplement is a concise technical reading edition. The main manuscript and every existing scientific input are unchanged. No model inference, channel simulation, metric scoring, resampling, timing or training is performed.

The preceding full technical text is preserved byte-for-byte as [technical_record_20261103_full.tex](technical_record_20261103_full.tex). Its section numbering and run-history prose describe the preceding edition. All full-precision result files remain at their original paths and are included in the downloadable companion. Table compression rounds presentation only; original values and confidence intervals are never overwritten. Nonzero interval endpoints are not rounded to zero.

## Evidence map

### S1

**Reading PDF:** Algorithm, coder-specific proof, CDF/numerical route, protected-frame fields, calibration and configuration index.

**Companion:** Environment/path setup, full hashes, verification commands, frame/decode counts and calibration execution totals.

**Presentation rule:** All methodological conditions retained; implementation/run details indexed outside the reading PDF.

- [model_manifest.json](../inputs/reproduction/model_manifest.json)
- [reproduction_config.json](../inputs/reproduction/reproduction_config.json)
- [experiment_index.csv](../inputs/policy_index_20261103/experiment_index.csv)
- [policy_sets.csv](../inputs/policy_index_20261103/policy_sets.csv)
- [directory_versions.json](../inputs/policy_index_20261103/directory_versions.json)
- [pilot_candidate_scores.csv](../reproduction/pilot_candidate_scores.csv)
- [confirmation_candidate_scores.csv](../reproduction/confirmation_candidate_scores.csv)

### S2

**Reading PDF:** All 40 system/workpoint five-metric means, main family contrasts, ARPC adaptation contrasts and failure denominators.

**Companion:** Every marginal CI, remaining external paired contrasts and source-level better/tie/worse counts.

**Presentation rule:** N1024: all NeST-WHOLE, NeST-Ada, WHOLE-Ada and NeST-ARPC points. N512: NeST-WHOLE, NeST-Ada and WHOLE-Ada at all SNRs; Quarter means retained. No choice based on sign.

- [summary.csv](../inputs/unified_calibration_20261103/external_join/summary.csv)
- [paired_deltas.csv](../inputs/unified_calibration_20261103/external_join/paired_deltas.csv)
- [per_source_improvement.csv](../inputs/unified_calibration_20261103/external_join/per_source_improvement.csv)
- [capacity_summary.csv](../inputs/unified_calibration_20261103/formal/capacity_summary.csv)
- [summary.csv](../inputs/e1/summary.csv)
- [paired_deltas.csv](../inputs/e1/paired_deltas.csv)

### S3

**Reading PDF:** Both matched grids, all their five-metric paired contrasts, ordered A0/B0/C0 contrasts and four-metric Ada sensitivity.

**Companion:** New/reused execution counts and historical counter-field definitions.

**Presentation rule:** All original control comparisons retained; DISTS is not invented for the earlier four-metric Ada diagnostic.

- [summary.csv](../inputs/e3/summary.csv)
- [paired_deltas.csv](../inputs/e3/paired_deltas.csv)
- [efficiency.csv](../inputs/e3/efficiency.csv)
- [paired_deltas.csv](../inputs/e4/paired_deltas.csv)
- [summary.csv](../inputs/unified_calibration_20261103/low_budget/summary.csv)
- [paired_deltas.csv](../inputs/unified_calibration_20261103/low_budget/paired_deltas.csv)
- [ada_generation_sensitivity.csv](../inputs/v9_package/inputs/ada_generation_sensitivity.csv)

### S4

**Reading PDF:** Six-budget distance figure, all NeST-WHOLE/Ada five-metric intervals, overhead and saturation explanation.

**Companion:** Full means/intervals, WHOLE-Ada contrasts, source/padding-bit table and execution/reuse counts.

**Presentation rule:** All six budgets and favorable, unfavorable and zero-crossing contrasts retained.

- [summary.csv](../inputs/e2/summary.csv)
- [paired_deltas.csv](../inputs/e2/paired_deltas.csv)
- [representation_and_capacity.csv](../inputs/e2/representation_and_capacity.csv)
- [source_means.csv](../inputs/e2/source_means.csv)

### S5

**Reading PDF:** Fixed-policy threshold figure with intervals, every rejection count, source-cluster denominator and limits.

**Companion:** Duplicated per-metric tables and metric-cache/reconstruction execution counts.

**Presentation rule:** All five actual SNRs and all three methods retained; success-conditional and unconditional outputs remain distinct.

- [summary.csv](../inputs/e6/summary.csv)
- [source_means.csv](../inputs/e6/source_means.csv)

### S6

**Reading PDF:** Main selected-example rule plus the fixed median-rank and minimum DISTS-gain examples.

**Companion:** Full five-metric per-image listing and additional 7-dB advantage gallery.

**Presentation rule:** No reselection: identities/repetitions remain fixed; counterexample retained. Historical and current policies explicitly distinguished.

- [display_metrics.csv](../inputs/fig3_unified_20261103/display_metrics.csv)
- [fig3_selected_sources.csv](../inputs/f1/fig3_selected_sources.csv)
- [fig3_metrics.csv](../inputs/f1/fig3_metrics.csv)
- [e2_quantile_example_ids.csv](../inputs/v9_package/derived/e2_quantile_example_ids.csv)
- [figure3_all_methods_supplement.pdf](../figures/figure3_all_methods_supplement.pdf)
- [supp_examples_distribution.pdf](../figures/supp_examples_distribution.pdf)

### S7

**Reading PDF:** Main physical-core TX/RX timings, matched-grid costs, compiled Ada latency and its paired five-metric output quality.

**Companion:** Historical SMT tables, two-source prefill match counts, counter corrections, resource windows and run logs.

**Presentation rule:** Main/SMT protocols never pooled; actual timed compiled quality stays paired with its own latency, not common500 quality.

- [timing_summary.csv](../inputs/timing_fairness_20261102/timing_summary.csv)
- [granularity_summary.csv](../inputs/timing_fairness_20261102/granularity_summary.csv)
- [summary.csv](../inputs/ada_compiled_prefill_20261103/summary.csv)
- [paired_deltas.csv](../inputs/unified_calibration_20261103/compiled32/paired_deltas.csv)
- [summary.csv](../inputs/unified_calibration_20261103/compiled32/summary.csv)
- [timing_summary.csv](../inputs/e5/timing_summary.csv)
- [source_fitting_accounting.csv](../inputs/e5/source_fitting_accounting.csv)
- [model_call_accounting.csv](../inputs/e5/model_call_accounting.csv)
- [timing_execution_history_20261103.md](../reproduction_logs/timing_execution_history_20261103.md)

### S8

**Reading PDF:** All three SNRs/four-method means and all NeST-WHOLE/Quarter/Ada five-metric paired intervals.

**Companion:** Marginal intervals, other contrasts, source-level win counts and new/reused frame/render totals.

**Presentation rule:** All SNRs and adverse/mixed outcomes retained; historical 19-dB Ada policy explicitly identified.

- [summary.csv](../inputs/kodak_v9/summary.csv)
- [paired_deltas.csv](../inputs/kodak_v9/paired_deltas.csv)
- [per_source_improvement.csv](../inputs/kodak_v9/per_source_improvement.csv)
- [source_means.csv](../inputs/kodak_v9/source_means.csv)
- [capacity_summary.csv](../inputs/kodak_v9/capacity_summary.csv)

## Implementation and access

The [reproduction checklist](../REPRODUCTION_CHECKLIST.md) supplies commands, numerical environments, public and project-stored model identities, and directory mappings. Historical execution notes remain in [the reproduction log](../reproduction_logs/timing_execution_history_20261103.md). Raw pixel caches, training weights and datasets remain in project storage; the companion does not assert public access to private PFS paths.

The [rendered-cell ledger](reader_table_cells.json) records input-file hashes and CSV row/field pointers for every reformatted quality cell. Full CI files allow both favorable and unfavorable results to be checked independently of the reading edition's table selection.
