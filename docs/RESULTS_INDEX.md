## 20260923 review and research update

- Current bounded-study results: ../results/review_20260923_phase2/, with complete source/SNR/seed pairing, selected model/Decoder/numerical-protocol identities and actual GPU receipts.
- Phase1 source repairs, predictor/linear requalification and timing remain documented at ../reports/review_20260923_phase1.md.
- Current N4084/Dc digital calibration and adaptive comparison use the explicit strict-precision version. Earlier digital adaptive development values were numerically reproduced; they are not broadly discarded. Historical TF32 calibration is preserved with its original scope.
- Folded512 is freshly re-evaluated at strict precision; older allocation_v1 remains historical evidence.
- Historical adapted-Decoder rows are separate context, not newly qualified same-Dc controls. New holdout, m10 and full adapted-Decoder digital adaptation are NOT_RUN in this bounded study.
- Exact historical impact: ../results/review_20260923_phase2/historical_impact.json. Per-item completion and exclusions: ../results/review_20260923_phase2/issue_status.json.

# 结果、图表与原始产物映射

## 最新：固定混合三权重收尾

目录`results/hybrid_weight_closure/`对应原`outputs/HYBRID-WEIGHT-CLOSURE-20260916/`的精选文件。

| 文件 | 内容 |
|---|---|
| `development.csv` | 六实际工作点、9000条传输；含全部失败与指标 |
| `references.csv` | 六冻结系统、9000条匹配参考 |
| `frozen_comparisons.json` | 校准冻结的checkpoint、真实点匹配及比例；不按development重选 |
| `quality_summary.csv` | 主/机制/高SNR及逐SNR均值 |
| `paired_intervals.csv` | 1856项源图级配对区间 |
| `expected_time_sharing.csv` | 9000条**期望**参考，不是部署传输 |
| `matched_actual_points.*` | 12项预登记匹配判断，含覆盖不足 |
| `full_calibration_all_metrics.csv` | 完整校准节点、六臂各指标与加权选模目标 |
| `training_curves_100update_windows.csv` | 真实训练曲线的100步窗口均值 |
| `final_calibration_trends.csv` | 末段仍改善的证据，不伪称封顶即收敛 |
| `failure_coverage.csv` | 各SNR所有失败计入的核对 |
| `independent_audit.json` | 1500 PHY / 9000图像及统计复核回执 |
| `figures/` | 取舍图、校准曲线、训练曲线、逐SNR图，PNG＋PDF |

图例中的条件/控制、实际点/期望参考、校准/development必须区分。图像数据集的照片与重建样本阵列没有上传。

## 其他必要对照

- `results/digital_adaptation/`：完整raw与实际整帧算术码＋FEC、校准策略、development逐帧结果，以及**早已使用**的历史holdout汇总。
- `results/online_timing/`：已审计冻结系统同端点在线处理时间。
- `results/hybrid_base_conditioning/`：原0.01基图条件实验；保留原负结论。
- `results/hybrid_source_correction/`：原三通道gain等源纠偏负结果。
- `results/learned_prefix_reference/`：parallel / next-scale学习链及强参照汇总。
- `results/progressive_digital/`：固定预算数字逐尺度系统参考。

上述都是选择性公开结果，不是完整历史输出目录。每个文件的原始相对路径与原始SHA可在`release_manifest.json`追溯。历史报告中的其他绝对路径仅是审计线索，不保证该资产已上传。
# External baseline update

[External results](../results/external_baselines/README.md): 25,500 completed development/protocol/reference rows, 18,000 calibration candidate rows, 119 summaries, 1,624 paired metric intervals and 1,920 complete TX/RX timing pairs. Author-assumed and paid-information protocols remain separate.

The user-requested curated figures include 120 PNG files: 24 comparison panels and 96 individual reconstructions; 12 panels use exactly N4498. Standalone source images, raw datasets and pixel arrays are excluded. HiFi is not ranked before completion. Large CSVs are partitioned without changing scientific cells; provenance is in the release manifest.

The index below preserves the original historical release.

- 2026-09-24 P2048 initial20k: `results/token_channel_efficiency_20260923/budget_milestones/P2048_seed2026092304_20k/`; real calibration/selected lineage, not convergence or completed development. Report: `reports/token_efficiency_P2048_20k_20260924.md`.

- 2026-09-24 N4084 historical provenance: `results/token_channel_efficiency_20260923/n4084_reference_audit_v1/audit.json`; seven methods,10500 real rows and300 original timing checks. Current full PHY/metric/timing compatibility remains pending. See `reports/token_efficiency_n4084_reference_audit_20260924.md`.
