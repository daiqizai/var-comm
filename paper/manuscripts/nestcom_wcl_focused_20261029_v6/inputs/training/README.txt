SwinJSCC training and checkpoint evidence for NeST-Com V5

The five original CSV/JSON files are copied byte-for-byte. source_manifest.json
maps each copy to its original repository path and SHA-256. Historical stopping
labels are preserved; no training, image evaluation, or source metric was rerun.
The shared original_training_plot_data.csv includes both P1024 and SwinJSCC.

Computation in swin_training_audit.json:
1. Select branch "SwinJSCC (adapted)" and series
   "training_loss_500_update_block_mean" from original_training_plot_data.csv.
2. For each (start, end] global-step window, weight each block mean by its count.
   Every block contains 500 updates. All blocks in the windows are included.
   Means for 140k-160k / 160k-180k / 180k-200k are respectively
   0.006115509365885373 / 0.006115821430805226 / 0.006086149529604154.
3. Calibration change uses 100 * (objective_before - objective_after) /
   objective_before. At 180k and 200k the original objectives are
   0.006579187295491768 and 0.006566026536579253, a 0.20003624036562137%
   reduction (rounded to 0.20% in the supplement). At 190k-200k it is 0.077376%.
4. Selection minimizes mean RGB MSE over 12 equally weighted budget/SNR cells:
   1,000 calibration sources, two budgets, six SNRs, one noise per cell.
   The 200k checkpoint has the lowest objective among all 25 calibration checks.
   No test or holdout score enters selection.

Interpretation: trained for 200k updates, calibration-selected checkpoint, and
largely stable late-stage loss are supported. The original stringent two-window,
all-cell 0.2% stability rule was not passed; its original records are retained.
This descriptive plateau summary does not change the historical stopping rule.
The frozen parent inputs/six_system_configuration.csv is unchanged.
