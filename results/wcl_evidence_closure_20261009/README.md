# Completed NeST-Com WCL supplementary evidence

T0–T6 are complete. Start with the [final report](FINAL_REPORT.md), its [input bindings](FINAL_REPORT_evidence_v1.json), and the [212 completed checks](FINAL_REPORT_checks_v1.json). The original common500 release is `252176e041758ecb2d3e81b6fde5b587e7e17bb7`; the scientific code snapshot preceding this publication is `14b09ecd72984fb39c683d1a62bcd3221c69142f`.

## Results and figures

| Task | Completed evidence |
|---|---|
| T0: reuse and scope | [Inventory](T0_inventory/inventory.md), [execution plan](T0_inventory/RUN_PLAN.md) |
| T1: same-prior whole-scale entropy coding | [Report](T1_entropy_whole/v3/REPORT_T1.md), [summary](T1_entropy_whole/v3/summary.csv), [paired comparisons](T1_entropy_whole/v3/paired.csv), [source lengths](T1_entropy_whole/v3/source_lengths.csv) |
| T2: whole-scale candidate coverage | [Report](T2_calibration_audit/v1/REPORT_T2.md), [coverage](T2_calibration_audit/v1/candidate_coverage.csv), [full calibration rankings](T2_calibration_audit/v1/full_calibration_rankings.csv) |
| T3: Swin adaptation and paid side information | [Field audit](T3_swin_sideinfo/side_information_audit.md), [training scope](T3_swin_sideinfo/training_scope.md) |
| T4: actual resources, energy and software time | [Report](T4_resources_cost/summary_completed_v2/REPORT_T4.md), [timings](T4_resources_cost/summary_completed_v2/timing_summary.csv), [actual energy](T4_resources_cost/summary_completed_v2/energy_summary.csv) |
| T5: original500 curves and fixed examples | [Complete/partial curves](T5_figures/fig_t5_same_prior_N1024.pdf), [paired increments](T5_figures/fig_t5_partial_minus_complete.pdf), [entropy curves](T5_figures/entropy_quality_completed_v1/fig_t5_entropy_quality_N1024.pdf), [external examples](T5_figures/external_completed_v1/), [accepted mechanism examples v2](T5_figures/mechanism_entropy_completed_v2/), [quality versus TX time](T5_figures/quality_tx_cost_completed_v1/) |
| T6: N2048 confirmation | [Report](T6_budget2048/v1/REPORT_T6.md), [protocol](T6_budget2048/v1/N2048_protocol.md), [summary](T6_budget2048/v1/summary.csv), [paired comparisons](T6_budget2048/v1/paired.csv), [configurations](T6_budget2048/v1/selected_configurations.csv) |
| Cross-budget mechanism figure | [PDF](T6_budget2048/figures_v1/fig_t6_cross_budget_partial_minus_complete.pdf), [SVG](T6_budget2048/figures_v1/fig_t6_cross_budget_partial_minus_complete.svg), [PNG](T6_budget2048/figures_v1/fig_t6_cross_budget_partial_minus_complete.png), [captions and reproduction](T6_budget2048/figures_v1/README.md) |

T2 retained the original N1024 whole-scale winners at 10/19 dB after the finite expanded calibration review. At N1024/19 dB, VAR-conditioned entropy coding makes whole-scale m9 feasible for all 500 sources and exceeds raw partial-scale PSNR by 0.257819 dB, with greater TX cost (155.844 versus 52.731 ms, measured on 16 fixed development sources with three measured repeats per source). The quality and timing populations differ. Neither universal quality superiority nor global search optimality is claimed.

The independent N2048 confirmation comprises 100 registered sources, three SNRs (4/10/19 dB), three noise repeats and three digital methods: 2,700 method frames. Raw partial minus raw whole PSNR is +1.220451, +0.059652 and exactly zero dB, respectively. At 19 dB all three digital methods transmit the full ten scales and produce identical outputs. The full report retains uncertainty in the other metrics and comparisons.

The N1024 and N2048 populations differ (500 versus 100 sources). Each paired difference is computed within its own budget; their absolute means are not a source-matched budget experiment. LPIPS retains method-minus-reference signs; ConvNeXt measures source-prediction agreement, with differences in percentage points. Swin 19 dB remains outside its training/calibration range.

## Publication and restoration boundaries

This publication preserves the completed scientific CSV/JSON, scripts, protocols and figure bytes. Earlier plans, failed attempts and intermediate indices remain historical snapshots. In particular, the older N1024 index/template still says T6 is pending and contains workstation links; use this relative-link entry and `FINAL_REPORT.md` for the completed T0–T6 status. The rejected mechanism layout v1 is historical; the accepted figure set is v2.

The original document disabled automatic pushes. The user explicitly requested publication on 2026-10-10; this commit follows that later instruction. Publication itself performs no training, image inference, channel simulation, scoring or bootstrap. Repository CPU checks are synthetic engineering checks.

Git contains reviewed source, reports, figures and lightweight results under the existing filters. Model weights, source datasets, tensors, environments, raw run caches and recovery archives remain external. Absolute paths and hashes in sealed provenance identify their original execution locations; a Git checkout alone does not supply those assets. Figure reproduction commands require their documented inputs and a fresh output directory.

The separate Lu_Data restoration transfer and its byte-verification receipts determine backup completion. This Git publication does not claim that transfer, a fresh staging restoration, or a new server runtime has passed. The final report's backup-status sentence records its creation time and is not rewritten by publication.
