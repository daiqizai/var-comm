# Cross-budget within-budget mechanism comparison

This is a rendering of existing paired intervals, not new inference, PHY, scoring or bootstrap. N1024 uses the original500 sources at all six SNRs; N2048 uses the registered new100 confirmation sources at4/10/19dB. Both contrasts are partial-scale raw minus complete-scale raw, including all failure outcomes. The two curves do not report raw absolute quality or a same-source causal budget effect. LPIPS signs are preserved; agreement deltas are percentage points.

All5 figures have vector PDF/editable SVG and600dpi PNG exports. The source CSV precision is preserved in plot_data.csv. No zeros, negative means or confidence limits are discarded; real numeric SNR spacing is used. Only two budgets were evaluated; no bandwidth-saving percentage or equal-quality crossing is inferred. New statistics admission requires the explicit actual completion SHA and verifies its sealed paired/summary/point files,100-source/3-noise/2700-frame closure, frozen evaluator and selection flags.

Reproduce into a fresh directory:

```text
python experiments/wcl-evidence-closure-20261009/scripts/plot_t6_cross_budget.py --original-paired ".research\main_raw64_20261007\take_over_v1\final_publication_r6\actual_staging_r6\results\main_raw64_20261007\final_common500_r6\paired.csv" --original-pairs ".research\main_raw64_20261007\take_over_v1\final_publication_r6\actual_staging_r6\results\main_raw64_20261007\final_common500_r6\pairs.json" --new-completion ".research\wcl_evidence_closure_20261009\t6_actual\statistics_v1\completion.json" --new-completion-sha 765185997ed066bf5d27f09580ba57b1e622eb3c72a35bd55b1468e1a0057ad5 --out "<fresh-output>"
```
