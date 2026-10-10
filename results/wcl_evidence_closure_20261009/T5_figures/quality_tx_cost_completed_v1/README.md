# Actual quality and TX cost

Six completed2x2 figure groups: mean TX(primary) and median TX(companion), each at4/10/19dB. PDF/SVG contain vector points/error bars/text; PNG600dpi. Logarithmic time axis is explicitly labelled. Quality uses500 sources x3 noises; TX uses16 development sources x3 measured repeats. Warm-ups are excluded. No cross-population per-image correlation or paired interval is constructed. Vertical intervals are existing quality CIs. Timing has no inferred CI.

The actual1152-frame timing owner completed with observed exit0;576 measured rows and576 warm-ups are sealed. The plot validates all-frame TX means and medians against timing_per_call.csv and verifies the original raw quality values unchanged. All four methods, three registered SNRs and four metrics are retained, including VAR-conditional entropy at19dB with higher quality and greater TX time than NeST-Com.

Reproduce with Python, numpy and matplotlib into a fresh output directory:

```text
python experiments/wcl-evidence-closure-20261009/scripts/plot_t1_quality_tx_cost.py --statistics ".research\wcl_evidence_closure_20261009\t1_actual_statistics" --original-summary ".research\main_raw64_20261007\take_over_v1\final_publication_r6\actual_staging_r6\results\main_raw64_20261007\final_common500_r6\summary.csv" --freeze ".research\wcl_evidence_closure_20261009\t1_frozen_inputs\frozen_policy.json" --timing ".research\wcl_evidence_closure_20261009\t4_actual_timing_v3" --out "<fresh-output-directory>"
```
