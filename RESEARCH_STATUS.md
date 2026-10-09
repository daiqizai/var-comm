## Latest completed supplement (2026-10-09)

[Paper supplement and result index](results/paper_supplement_20261008/PUBLICATION.md): the A-class first round is complete, including the later-added adaptive BPG + LDPC baseline (500 sources x 6 SNRs x 3 noises), four metrics and source-paired comparisons, native Swin checks, five-method single-output timings, resource/failure tables and paper figures. The original frozen common500 results at commit `252176e` remain unchanged.

Adaptive BPG has higher PSNR at every tested SNR; the proposed method has better LPIPS, DINOv2-L and source-prediction agreement. All points and negative findings are retained. Swin at 19 dB is outside its training/calibration range; native256 BPG timings do not measure the adaptive source search. External native replication and B/C extensions remain conditional and were not started. No training or new scientific work is triggered by this publication; the monitor remains paused.

## M2 scale-causal study delivered 2026-10-02

Zero neural training updates. Frozen calibration choices, original100 development sources and failure-inclusive metrics. See [complete report](reports/scale_causal_m2_20261002.md); historical models and queues are preserved. The two-method authorized pipeline stops after this verified publication; actual-link branch status is explicit in the report.

## M1 scale-causal study delivered 2026-10-02

Zero neural training updates. Frozen calibration choices, original100 development sources and failure-inclusive metrics. See [complete report](reports/scale_causal_m1_20261002.md); historical models and queues are preserved. Method2 proceeds only after the verified method1 publication.

## N1024 extreme bandwidth study completed 2026-10-01

One fresh P1024 completed 40000 updates and selected step 40000 using complete1000-source calibration at1/4/7/13/19dB. Digital raw N1024 policies use paid68-symbol headers and were independently frozen on1000 calibration sources; P1024 receiver policies use200 calibration sources. All systems then used the same100 development sources and three noise repeats. P1024 budget_truncated=True; no convergence claim is made. QPSK/continuous have actual per-frameE2048;16QAM retains original fixed constellation scaling and actual energy. H_D/H_R decisions, paid-class conditions, distortion, source specificity, failures and computation are reported. Development is reused and the intervals do not cover training-seed variance. This authorized N1024 stage is complete. N512 remains frozen at its 40000-update budget cap; further experiments await a separate decision.

[Scientific report](reports/extreme_bandwidth_probe_20261001_R1_N1024.md).

## N512 extreme bandwidth study completed 2026-09-30

One fresh P512 completed 40000 updates and selected step 40000 using complete1000-source calibration at1/4/7/13/19dB. Digital raw N512 policies use paid68-symbol headers and were independently frozen on1000 calibration sources; P512 receiver policies use200 calibration sources. All systems then used the same100 development sources and three noise repeats. QPSK/continuous have actual per-frameE1024;16QAM retains original fixed constellation scaling and actual energy. H_D/H_R decisions, paid-class conditions, distortion, source specificity, failures and computation are reported. Development is reused and the intervals do not cover training-seed variance. N1024 remains a later decision.

[Scientific report](reports/extreme_bandwidth_probe_20260930_R1_N512.md).

## Step2 B completed 2026-09-30

The authorized single P_low arm continued selected27500 with the original populated AdamW, order and RNG. It completed 20000 added updates and selected added step 20000 only by full1000 calibration utility. The frozen evaluation then recalibrated receiver policies on200 calibration sources and evaluated100 development sources at -5/-2/1/4/13dB. G2: -5.0dB BYPASS_SELECTED, -2.0dB BYPASS_SELECTED, 1.0dB BYPASS_SELECTED. The13dB result is an out-of-training-range side-effect diagnostic.

[Report and full tables](results/rx_posterior_step2_B_20260930_R1/README.md). The development set is reused; this is not a new holdout or a training-seed replication.

## Step2 B started 2026-09-30

The user explicitly authorized B. A single P_low arm resumes selected27500 with original AdamW moments, source order and RNG. Train SNRs are -5/-2/1/4dB; full1000 calibration selects checkpoints every2500 added updates. The first milestone is10000 added updates; calibration-only extensions are capped at30000. Real-GPU populated-optimizer bitwise resume, frozen Dc/LPIPS gradients, and N4084/E8168 checks passed. The detached pipeline will evaluate the selected model and publish results after checks. Training/evaluation are in progress; no B scientific conclusion is available yet. Original A/P assets and paused old queues are preserved.

> 2026-09-30 Step2 A COMPLETE: frozen P4084 selected27500, zero training updates. Policies bypass at 1/4/7/13 dB. Out-of-training-range -5/-2 dB exceed B2/A1/A2 with shared-weight content support, at about 9.7x receive time. High-SNR protection is bypass; raw fusion worsens LPIPS. B/P_low and old queues remain paused pending user decision. See [reviewed report](results/rx_posterior_step2_A_20260930_R1/report_reviewed.md) and [complete result archive](results/rx_posterior_step2_A_20260930_R1/README.md).

> 2026-09-24 Swin/ADJSCC:9000 actual paid-frame PHY receipts/noise hashes/resources verified. Qualification/development adapter revision differs; not a confirmed quality error, but fresh native parity and uniform timing remain required. Pinned torch1.12.1 runtime is available. See reports/token_efficiency_author_phy_audit_20260924.md.

> 2026-09-24 N4084 actual replay continuation implemented:6 legacy methods, real24-check gate before9000 frames/600 timings, serially after the existing reference metric worker. GPU acceptance/results are NOT_RUN until actual receipts. See reports/token_efficiency_n4084_replay_queue_20260924.md.

> 2026-09-24 historical N4084 provenance:7 methods/10500 original rows and300 original timing consistency records verified; source pixels, selected hashes and frozen-policy lineage checked. Current full PHY/metric/timing compatibility remains pending. See reports/token_efficiency_n4084_reference_audit_20260924.md.

> 2026-09-24 P2048: real initial20k complete, all135000 calibration rows and selected SHA verified. Both final calibration intervals support a10k extension after the existing P3060 initial run. Development/timing and full merged delivery remain pending. See reports/token_efficiency_P2048_20k_20260924.md.

> 2026-09-23 scheduler recovery: original C controller is RETIRED_VERIFIED after a blocked m6 launch, with failure evidence preserved and no m6 training. Detached coordinator_v2 resumed P2048 from verified step5967 and advanced beyond6000. Never SIGCONT the old PID. See reports/token_efficiency_scheduler_recovery_20260923.md. A is complete; B1/B2/C remain pending.

> 2026-09-23 source A COMPLETE: actual source streams/roundtrips and20 representation paths for1000 calibration+100 original development sources, calibrated quality targets frozen before development. Full source report: reports/token_channel_efficiency_20260923_source_A.md. P2048/P3060 real-device qualification passed; P2048 is training. B1/B2 communication matrices and merged short-prefix C remain incomplete; no new holdout.

> 2026-09-23 supplement accepted: A actual source representation/bitstreams, B1 P2048/P3060 and QPSK resource curves, B2 real16QAM, then merged short-prefix C. New holdout and content selectors are deferred. Code/CPU qualification is not real-result completion. See experiments/token_channel_efficiency_20260923/README.md and outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/status.json. Existing m6 cache continues unchanged; its controller is intentionally held before training to honor A/B priority.

> 2026-09-23 short-prefix study: user-authorized new plan; v1 protocol, five-arm interface,4500 real-PHY preflight and real-GPU qualification implemented. First20k matrix is being launched; new quality/development,N3060,seed repeats,policy and new500 test are NOT_RUN. See reports/short_prefix_20260923_launch.md and experiments/var-short-prefix-hybrid-20260923/README.md.

> 2026-09-23 phase2 COMPLETE: registered bounded training, full calibration, selected-model quality/timing, strict-precision digital requalification, frozen resource lookup and PCA comparison are complete. See reports/review_20260923_phase2.md and results/review_20260923_phase2/. Original A/B and historical artifacts are preserved; no new holdout or convergence claim.

> 2026-09-23 phase1: local review repairs and affected real-weight reevaluation completed; see reports/review_20260923_phase1.md. Phase2 is separately authorized and not included in this completion.

> 2026-09-23 repository migration: the actual VAR_COMM root is the sole Git worktree. Historical research claims below are not newly certified by this migration. No A/B retraining or GPU quality validation was performed. See reports/repository_migration_20260923.md.

# Research status after consolidated repair (2026-09-22)

Source fixes and focused validation are recorded in `reports/repair_consolidated_20260922.md`. Historical quality conclusions remain valid only where their original receipts and scope are complete. See `results/repair_consolidated/issue_status.json` for per-issue status; blocked and partial entries are not presented as completed scientific results.

2026-09-24: P3060 initial20k and135,000 calibration rows verified; selected20k. Both budget milestones are complete; the existing delivery_chain has begun P2048 calibration-driven30k extension. P3060 also satisfies the initial extension criterion but awaits its serialized decision. Full evaluation/delivery remains pending. See [P3060 milestone](reports/token_efficiency_P3060_20k_20260924.md).

2026-09-24: P2048 completed30k and stopped by the registered calibration rule (last interval0.186047%<0.2%); selected30k. All195,000 calibration rows checked,60,000 new rows published. The existing scheduler has started P3060 until30k. Development/quality/timing and whole-study delivery remain pending. See [30k milestone](reports/token_efficiency_P2048_30k_20260924.md).

2026-09-24: P3060 completed 30k; selected 30k and 195,000 calibration records verified, with 60,000 new rows published. Last intervals improved 0.650284% / 1.597301%; the existing scheduler continues to 40k under the registered rule. P2048 remains finalized at 30k. Full development/evaluation/delivery remains pending. See [P3060 30k milestone](reports/token_efficiency_P3060_30k_20260924.md).

2026-09-24: P3060 finalized after 40k; selected 37.5k. Last interval changes +0.080443% / -0.470516% do not support extension. All 255,000 calibration records verified. P2048 remains finalized at 30k. Shared real-weight interface acceptance passed (selected trained-model replay remains pending); the existing scheduler is executing QPSK full calibration. See [training stop and acceptance](reports/token_efficiency_P3060_40k_and_shared_acceptance_20260924.md).


2026-10-04: Registered prior-aware UEP N1024 with real Sionna5G LDPC, continuous accepted-prefix RX, independent ConvNeXt validation and conditional N2048 replication. CPU BLER shards running; GPU source-Q waits for existing M1 N2048 delivery. No new quality result or training. See [Stage A](reports/prior_aware_uep_stage_a_20261004.md).


2026-10-04: Qualified M1 N2048 source-parallel GPU execution on the same six completed calibration sources with exact full-row parity (image metrics, link decisions and waveform/observation hashes; no reconstructed-pixel hashes were recorded). Selected 4 workers at 2.021x measured source throughput; preserved 286 prior source checkpoints. Original science and delivery paths continue. See [execution benchmark](reports/m1_gpu_workers_20261004.md).


2026-10-04: Prior-aware UEP N1024 actual-link evaluation, independent ConvNeXt validation, matched P1024 10 dB, fixed examples and cost measurements completed. DELIVER_AND_STOP_NO_EXTENSION. See [UEP report](reports/uep_prior_aware_20261004.md). No new model training or holdout access.
