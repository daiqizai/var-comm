# Claim–evidence matrix

| Claim | Population | Scope | Evidence |
|---|---|---|---|
| Retain scale-wise prediction while selecting within-scale transmission endpoints by actual terminated length | Frozen VAR-d16 representation and the implemented digital frame | Defines the fixed-frame design; does not claim a new probability model, arithmetic coder or VAR recurrence | sections/method.tex; inputs/reproduction/; original VAR attribution in references.tex |
| ARPC establishes the closest next-scale progressive source-coding precedent | Original ARPC publication and codec | Related source-coding work; NeST is not an implementation of ARPC's codec. Our ARPC+LDPC adaptation remains a supplementary comparison | references.tex; supplementary.tex S1/S2/S7; figures/figure2_complete_with_arpc.pdf |
| Token versus WHOLE, all five paired metrics at seven budget/SNR points | common500; 3 events/source | shared components, independently calibrated budget policies | inputs/e1/ |
| 83.4% DISTS, 83.0% DreamSim, 72.2% both improved | common500 N1024/10, source means | descriptive; numerical tie tolerance 1e-10; includes failures | inputs/v9_package/derived/e1_per_source_improvement.csv |
| Eighth ~92% of Whole-to-Token gain; Token +31.1ms TX | control100 quality, timing32 cost | PSNR/DISTS/DreamSim only; shared ceiling (8,126); no universal optimality | inputs/v9_package/derived/granularity_quality_cost.csv |
| Ordered direct-residual and future-conditioning gains | control100 received Token prefixes | Mechanism evidence within VAR's inherited recurrence. A0 retains prior completion; these interventions do not establish a new receiver algorithm or superiority to every valid prefix implementation | inputs/e4/ |
| Lower DISTS/DreamSim than SwinJSCC, Latent JSCC and BPG at all four N1024 SNRs | common500; 3 events/source | 24 direct paired intervals exclude zero; full-system comparison, not isolated endpoint attribution. Higher comparator PSNR remains reported | inputs/e1/paired_deltas.csv; validation/positioning_20261102/data_checks.json |
| Ada has better distances at high equal-bit budgets | common500, six bit budgets | full-system differences; retains tokenizer/prior/decoder/class differences | inputs/e2/ |
| Argmax worsens Ada DreamSim at 512/4,512/10,1024/10 | fixed100, original three events | closed sensitivity, original sampling curve retained; no DISTS invented | inputs/v9_package/inputs/ada_generation_sensitivity.csv |
| Kodak Token improves PSNR/DISTS/DreamSim over WHOLE; Ada better DISTS at 10/19 | 24 center crops,3 SNR,3 original events | no recalibration; 24 independent sources, all864 events included | inputs/kodak_v9/paired_deltas.csv |
| 0.660s vs 15.422s; JSCC faster | timing32, matched hardware/quota | software TX+RX processing, excludes loading/I/O/scoring | inputs/e5/timing_summary.csv |
| Finite fixed-policy rejection transition | 100 sources×20 events per point | 8–10dB fixed-policy scan; no broad channel-robustness claim | inputs/e6/ |
| Paid NULL current catalogue works without altering normal outputs | 33 CPU checks / all12 MCS | protocol regression, not new quality or reliability result | ../../../../results/nestcom_v9_completion_20261101_v1/execution_receipts/paid_null/worker_completion.json |


## Fig.3 selected examples, 2026-11-02

On selected sources 135 and 201 at N1024/10 dB, NeST has lower DISTS and DreamSim than WHOLE, SwinJSCC and Ada. Individual perceptual advantage examples; not random, representative, all-metric or aggregate superiority. Swin PSNR and the stated Ada per-image advantages are retained. Evidence: `inputs/fig3_selected_20261102/all_five_metrics.csv`; full source/noise/native array identities in `inputs/fig3_selected_20261102/native_mapping.json`. Population: Two user-selected first-repeat examples after candidate inspection.
