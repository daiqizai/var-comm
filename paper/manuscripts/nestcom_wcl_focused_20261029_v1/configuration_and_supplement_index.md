# 配置、训练与补充证据索引

[补充PDF](supplementary.pdf) 的 S1–S6 给出可直接阅读的配置、全部诊断、失败和旧案例。本轮只重新排版，不覆盖既有记录。以下链接固定到核验基准提交。

| 内容 | 固定来源 | 范围 |
|---|---|---|
| N1024同模型主对照 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/n1024_consistency_20261019_v1/repair_v1/delivery/paired_common500.csv) | common500，PARTIAL−WHOLE来源配对 |
| Kodak主对照 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/kodak_current_20261026_v1/delivery/paired_differences.csv) | 24源；独立于common500人口 |
| 固定粒度 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/adatoken_response_20261026_v1/a_delivery/REPORT.md) | 100源、固定模型/MCS/最大端点 |
| 接收干预 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/adatoken_response_20261026_v1/b_delivery/REPORT.md) | 同100源实际前缀；A0/B0/C0 |
| Ada完整适配 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/experiments/adatoken-response-20261026/C_PREDICTED_CLASS_PLAN.md) | 预测类别付费、固定帧，与原平均预算/ARQ的差异 |
| N512协议 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/nestcom_n512_20261028_v1/protocol.json) | 独立校准、paidNULL、预算及候选 |
| N512冻结策略 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/nestcom_n512_20261028_v1/frozen_policies.json) | 各SNR/MCS/端点 |
| N512全部结果 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/nestcom_n512_20261028_v1/delivery/REPORT.md) | 四指标、容量、失败、统计身份 |
| E1敏感性 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/nestcom_evidence_20261029_v1/e1/REPORT_zh.md) | 固定100源、3工作点、原sampling和新argmax |
| E2补全 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/nestcom_evidence_20261029_v1/e2/REPORT_zh.md) | 相同WHOLE实收前缀，coarse/completed |
| E1/E2重采样身份 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/nestcom_evidence_20261029_v1/statistics_identity.json) | 保留原100源bank；本轮无新抽样 |
| Swin选中checkpoint | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/baseline_strength_20261011/Swin_1to19/selected_checkpoint.json) | step200000，SHA及选取规则 |
| Swin训练状态 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/baseline_strength_20261011/Swin_1to19/stopping_checks.json) | BUDGET_CAP_WHILE_IMPROVING，非充分收敛声明 |
| Swin验证历史 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/baseline_strength_20261011/Swin_1to19/calibration_history.json) | 全程及尾段验证 |
| 训练曲线 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/baseline_strength_20261011/training_figures_v1/training_calibration_curves.pdf) | Latent/Swin原始训练/校准曲线 |
| 训练曲线原数据 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/baseline_strength_20261011/training_figures_v1/plot_data.csv) | 可追溯数值 |
| N2048 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/arpc_n2048_validation_20261019_v1/metrics_v1/paired_differences.csv) | 保留DISTS不利结果 |
| 失配统计更正说明 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/wcl_revision_20261027_v1/p1_bootstrap_clarification_v1.md) | 仅复用重采样索引；本次PARTIAL−WHOLE共享噪声 |
| 处理时间 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/results/five_method_timing_20261020_v1/delivery/summary_ms_per_image.csv) | 16开发源、原硬件协议；未给Ada计时 |
| 旧完整稿与配置来源 | [记录](https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/paper/manuscripts/nestcom_wcl_evidence_20261029_v1/figure_table_sources.json) | 原数据路径、commit和hash |

## 本地可复核副本

`inputs/` 保留与旧稿逐字一致的输入表、策略和身份。`figure_table_sources.json` 逐项给出路径与 SHA256，`validation/data_checks.json` 记录本轮不变性核验。N512 拒收次数（NeST 4/1/0；WHOLE 7/0/0；Ada 5/0/0，均各1500）及零 TX_NO_FIT 见 `inputs/failure_summary.csv`；全部计入原均值。N1024 的历史噪声身份、100源诊断范围、来源bank、训练/生成路由及新增/复用计数仍由原实验记录定义。

主文取消重复限制段并不改变协议。补充保留：ARPC同属next-scale；Ada为predicted-class固定帧适配；每个预算独立校准；源码容量利用率不同于总空口效率；固定10dB策略在实际8dB的300次/方法全拒收；计时未覆盖Ada；Quarter不是经证明的最优点。
