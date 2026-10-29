# 主张与证据

| 主张 | 证据 | 样本与范围 |
|---|---|---|
| LPIPS/DreamSim improve at six common500 budget/SNR points | `inputs/internal_common500_paired.csv`, `inputs/paired_deltas.csv` | 500 sources per budget, three outcomes/source; shared models; 12 original paired upper bounds below zero; no 500-source CLIP-I inference for N1024 |
| 13.7% LPIPS / 18.3% DreamSim relative mean reduction at N1024 10 dB | `inputs/internal_common500_paired.csv`, `inputs/common500_summary.csv` | common500; metric-mean percentages, not human preference improvement; original WHOLE means verified in prior evidence |
| Residual correction and later conditioning both reduce perceptual distances | `inputs/b_paired.csv` | fixed 100-source subset, N1024 10 dB; ordered B0-A0/C0-B0; A0 still has generation; final-scale propagation zero |
| Eighth adds LPIPS and PSNR gain over Quarter | `inputs/a_paired.csv`, `inputs/a_efficiency.csv` | fixed 100 sources, 10 dB, fixed maximum endpoint/MCS/model; DreamSim interval includes zero; not equivalence, saturation or optimality; timing blocked |
| Six-system perceptual and pixel-fidelity tradeoffs | `inputs/common500_summary.csv`, `figures/figure2_data.csv`, `tables/table_ii.csv` | N1024 common500; all 72 four-metric means retained; Ada/Latent advantages explicit; no overall SOTA or isolated representation-order causal claim |
| N512 joint-success directions retained | `inputs/reception/joint_success_paired_quality.csv` | 500 retained sources, 1491–1500 actual joint events depending on comparison/SNR; 24 descriptive source-equal conditional means; no new CI; failure-inclusive main estimates retained |
| New TX timing is incomplete | `inputs/timing/actual_compute_and_reuse.json` | 88 complete source pairs at first repeat only; 176 validated measured calls; source 89 tokenizer identity mismatch; no full-benchmark or speed-order claim |
