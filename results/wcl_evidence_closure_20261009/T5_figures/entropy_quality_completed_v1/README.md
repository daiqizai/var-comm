# Completed entropy-code quality figures

N1024 only; SNR4/10/19. Post-hoc same-source comparison on the original500 sources,3 noises each. This plotting run performs zero model, PHY and bootstrap calls. The two figure families have four independently laid-out single-column panels each, exported as vector PDF, editable-text SVG and600dpi PNG.

Inputs are authenticated by the completed T1 statistics SHA and output seals. The original raw partial means/CIs additionally match the published252176e041758ecb2d3e81b6fde5b587e7e17bb7 CSV. Only the displayed agreement values are multiplied by100. LPIPS signs are unchanged. All three registered SNRs and all selected rows are retained. The source CSV precision is preserved in plot_data.csv; no values are copied from prose.

Method mapping and appearance are in method_mapping.json; captions.tex explains pairing and the statistical/failure-rule limitations. The complete-scale static and VAR-conditional entropy curves are separate methods; no result is selected per source or noise.

Reproduce into a fresh output directory with Python, numpy and matplotlib:

```text
python experiments/wcl-evidence-closure-20261009/scripts/plot_t5_entropy_quality.py --statistics ".research\wcl_evidence_closure_20261009\t1_actual_statistics" --original-summary ".research\main_raw64_20261007\take_over_v1\final_publication_r6\actual_staging_r6\results\main_raw64_20261007\final_common500_r6\summary.csv" --freeze ".research\wcl_evidence_closure_20261009\t1_frozen_inputs\frozen_policy.json" --out "<fresh-output-directory>"
```
