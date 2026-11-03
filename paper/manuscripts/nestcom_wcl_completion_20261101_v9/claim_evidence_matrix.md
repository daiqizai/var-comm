# Claim–evidence matrix — system framing, 2026-11-03

All paths below are relative to this manuscript. Historical policy evidence retains its original population and cannot be pooled with the unified-policy results. `inputs/unified_calibration_20261103/` is abbreviated U.

| Claim | Population / conditions | Scope and limits | Evidence |
|---|---|---|---|
| Fixed-budget digital communication framework based on next-scale residual descriptions | Source-prefix selection, protected digital frame and prefix-constrained reconstruction | Overall system design; actual-length fitting supplies the specific technical mechanism | Abstract; introduction; II; Fig.1; TablesI–IV |
| Actual independently terminated within-scale prefixes fit a protected digital frame | Frozen VAR-d16, 68-symbol header, actual body capacity | Scale-wise probabilities with within-scale independent termination; inherited model/coder/receiver components are identified in II | sections/method.tex; U/public_directories.json; U/protocol.json |
| ARPC is the closest next-scale progressive source-coding precedent | Original source-coding work | Progressive source-coding precedent; specified ARPC digital-link adaptation and full results retained in S1/S2 | introduction; references.tex; complete Fig.S1 |
| Common12MCS pilot100/full1000 calibration | Three token systems, four SNRs, DINO objective including gray failure | 25 finite NeST ceilings with651 token fallbacks; not exhaustive651-ceiling optimization | U/pilot_calibration/candidate_scores.csv; U/full_calibration/candidate_scores.csv; U/frozen_policies.json |
| NeST improves all five paired quality metrics over WHOLE | N1024 common500,4/7/10/19,3shared-noise events/source | Independently calibrated system policies, not a pure-granularity causal contrast | U/formal/paired_deltas.csv |
| N1024/10 utilization77.70→99.79%, DISTS−8.7%, DreamSim−20.5% | same common500 | Source-bit capacity utilization, not total air-interface efficiency | U/formal/capacity_summary.csv; U/formal/summary.csv |
| DISTS412/500, DreamSim409/500, both361/500 improve | N1024/10 source means | Descriptive82.4/81.8/72.2%; includes failures, not per-image monotonicity | U/external_join/per_source_improvement.csv; U/external_join/joint_perceptual_improvement.csv |
| N512 Token−Whole+.533655 dB; Eighth−Whole+.463624 dB | control100×3, identical model/MCS/ceiling/noise/RX rule | Matched termination grid only; Eighth≈87% mean PSNR gain; Token−Eighth DISTS/CLIP intervals cross0 | U/low_budget/summary.csv; U/low_budget/paired_deltas.csv |
| Historical N1024 matched-grid quality/cost | control100 quality; timing32 cost, common ceiling(8,126) | Distinct from newly calibrated main policies; comparisons are not forward-call counts | inputs/v9_package/derived/granularity_quality_cost.csv; inputs/timing_fairness_20261102/granularity_summary.csv |
| Ordered residual and future-context receiver effects | control100 same actual received Token prefixes | Mechanism within inherited VAR recurrence; A0 still completes; not new receiver-algorithm priority | inputs/e4/ |
| NeST lower DISTS/DreamSim than Swin/Latent/BPG at all4SNR | common500,3events/source | 24 direct paired intervals; historical external noise retained, full-system contrast; Swin/BPG PSNR higher | U/external_join/paired_deltas.csv; U/external_join/identity.json |
| Ada perceptual advantages and mixed results retained | common500 unified N1024 policies | Ada DISTS/DS/CLIP better at10; DISTS better at19, DS inconclusive at7/19; not uniform dominance | U/formal/paired_deltas.csv |
| N512 main and equal-bit findings retained | Existing independently calibrated common500 policies; six bit budgets | Separate policies and target pools; not budget-only or representation-order causality | inputs/e1/; inputs/e2/; tables/table_iii.csv; U/historical_n512_policies.json |
| Existing Ada argmax, Kodak and rejection-transition evidence | Original fixed100 / Kodak24 / original fixed-policy scan | Historical policies, unchanged; no invented DISTS or updated robustness inference | inputs/v9_package/inputs/ada_generation_sensitivity.csv; inputs/kodak_v9/; inputs/e6/ |
| Compiled Ada quality paired with its own3.0643s cost | same32timing sources, actual saved events and generation seeds | All5 timed-vs-original CIs cross0; fixed old seed32/32 pixel equal. Diagnostic only; no common500 substitution or population equivalence | U/compiled32/quality_latency.csv; paired_deltas.csv; pixel_identity_audit.json |
| Exact termination length b=E+P+2 and nondecreasing lengths | 512 integer-prefix checks plus976500 saved streams/975000 adjacencies | Nonmutating finish; currently all-candidate scan. No optimized-timing claim | U/encoder_audit/integer_codec_checks.json; implementation_identity.json; REPORT_zh.md |
| Selected Fig.3 perceptual advantage examples | sources135/201 chosen from the previous gallery for favorable perceptual examples and fixed before unified calibration, N1024/10, original first repeat identity | No reselection; not representative/all-metric superiority. SwinPSNR and Ada individual advantages retained | inputs/fig3_unified_20261103/selection.json; native_mapping.json; display_metrics.csv |

## Wording and policy-context audit — 2026-11-03

| Clarification | Evidence and scope |
|---|---|
| MCS and current profile directory are preconfigured; the paid profile identifies the endpoint | sections/method.tex; Supplement S1.3; inputs/policy_index_20261103/directory_audit.json; no current per-frame directory/MCS switch claimed |
| Current U, earlier H, N512 L and matched C policy families are distinct | S1.3 policy/experiment index; 13 policy sets, 12 experiment groups, full directory hash pins |
| Main-method processing costs are explicit | Earlier H10, 32 sources, same RTX PRO 6000 and two physical CPU cores; inputs/timing_fairness_20261102/timing_summary.csv; complete TX/RX spans, no new timings |
| Operational history is retained outside the paper | reproduction_logs/timing_execution_history_20261103.md; final valid configurations and negative evidence remain in S7 |

## Contribution narrative — 2026-11-03

| Technical statement | Evidence and conditions |
|---|---|
| Probability-evaluation unit and transmission endpoint can be separated | Fixed within-scale conditional probabilities; II-A/II-B; actual independent terminated streams |
| Fixed-MCS feasible endpoints form an initial segment | b=E+P+2 under the specified traversal/finish; S1.5; U/encoder_audit; full-scan timing remains current |
| Eighth and Token offer different quality/cost operating points | N1024/10 matched control100 quality, timing32 cost; Token extra30.7ms/14.1%; N512 control is independently identified |
| Shared-model and matched-grid results justify the design; receiver interventions explain token use | TableI(A/B/C), N512 matched control; VAR overwrite/recurrence attributed in II-C |
| H10 timing is relevant to equal MCS/ceiling U10, with distinct directory identity | S1.3 policy index; historical timing result files; no new timing claim |
