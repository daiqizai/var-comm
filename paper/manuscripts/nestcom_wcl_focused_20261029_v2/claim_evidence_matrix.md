# Claim–evidence matrix

| Claim | Evidence/population | Boundary |
|---|---|---|
| Within-scale endpoints improve utilization and PSNR | Table I(A), common500/Kodak24 N1024; Table III context/N512 paired files | Shared model components; independent budget-specific calibration |
| Finer fallback preserves more source information | Current N512/10 dB 500 sources ×3: both target fallback99.6%; NeST within89%;91.256→107.896tokens | Same target/MCS; does not mean fewer fallback events |
| Residual and conditioning roles | Table I(C), fixed100-source A0/B0/C0 direct paired intervals | Ordered interventions, A0 still uses complete coarse scales |
| Average energy difference is small in these formal records | `results/nestcom_focused_review_20261029_v2/{energy_summary,paired_energy}.csv`,22500logical/20613physical | Finite QAM variation retained; no exact all-frame equal-energy or general reliability claim |
| Rejections do not explain N1024 internal gains | `failure_summary.csv`, each family/SNR1500parsed | Observed batch only; all gray outputs included at N512 |
| Completion changes reconstruction within fixed adapted receiver | E2 fixed100 source pairs; Supplement S4 | Decoder training did not explicitly cover coarse-only branch; not a fairly trained decoder comparison |
| System comparisons have metric-dependent tradeoffs | Fig.2/TableII/N512TableIII and frozen four-metric CSV | Ada predicted-class fixed-frame; no universal SOTA |
| NULL extension handles no-fit | New version18actualCPUframes36packetdecodes | Historical N1024 results unchanged; targeted engineering checks, no noisy reliability experiment |
| Fig3 illustrates system outputs | Two historical development sources,8native metric pairs | Not common500, not random/representative, method-specific noise |
