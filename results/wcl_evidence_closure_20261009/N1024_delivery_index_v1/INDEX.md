# N1024 已完成交付索引（独立版本）

更新时间（UTC）：2026-10-09T11:40:14.102700+00:00。N1024 T1–T5 已完成的本地交付经过本轮逐文件 SHA 复核；**T6 N2048 运行中，不计为完成**。本目录仅导航、原精度摘录及报告模板，0 模型、0 PHY、0 bootstrap、0 新计时；不覆盖旧封印或发布过滤器。所有数据图 PDF/SVG 为矢量线条与文字；重建样例的照片保留真实栅格像素，标题为矢量。PNG 为 600 dpi。

## 数据及报告

| 内容 | 当前入口 | 口径与边界 |
|---|---|---|
| 总报告草稿 | [FINAL_REPORT_TEMPLATE.md](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/N1024_delivery_index_v1/FINAL_REPORT_TEMPLATE.md>) | 已回答五个科学问题；T6 留待实测，不是全任务完成报告 |
| T0 盘点 | [inventory.md](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T0_inventory/inventory.md>)，[provenance.json](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T0_inventory/provenance.json>) | 历史盘点；后续完成状况以本索引和各新版本为准 |
| T1 整尺度熵码 | [REPORT_T1.md](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/REPORT_T1.md>)，[completion.json](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/completion.json>) | v3 含真实在线 TX/RX 成本；两家族各 500×3SNR×3噪声，事后同源比较 |
| T1 原精度质量及诊断 | [summary.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/summary.csv>)，[paired.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/paired.csv>)，[per_frame.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/per_frame.csv>)，[raw_crc_drop_per_frame.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/raw_crc_drop_per_frame.csv>) | 原 KEEP 不变，CRC-DROP 仅同 RX 诊断；禁止把两种 CI 相减 |
| T1 长度/冻结 | [source_lengths.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/source_lengths.csv>)，[m9_sendable_summary.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/m9_sendable_summary.csv>)，[frozen_policy.json](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/frozen_policy.json>)，[roundtrip_validation.json](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/roundtrip_validation.json>) | 实际纯熵码、m4 下界按长度回退、source-only 字段不代替 PHY 资格 |
| T2 候选审查 | [REPORT_T2.md](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T2_calibration_audit/v1/REPORT_T2.md>)，[candidate_coverage.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T2_calibration_audit/v1/candidate_coverage.csv>)，[full_calibration_rankings.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T2_calibration_audit/v1/full_calibration_rankings.csv>) | 10/19 dB 两原整尺度赢家不变；未重选原部分尺度，未新跑 holdout |
| T3 Swin 审计 | [side_information_audit.md](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T3_swin_sideinfo/side_information_audit.md>)，[training_scope.md](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T3_swin_sideinfo/training_scope.md>)，[adapter_validation.json](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T3_swin_sideinfo/adapter_validation.json>)，[header_budget.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T3_swin_sideinfo/header_budget.csv>) | 原合法付费适配保留；未训练 C7、不删内容相关 mask/power，19 dB 范围外 |
| T4 当前汇总 | [REPORT_T4.md](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2/REPORT_T4.md>)，[resource_summary.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2/resource_summary.csv>)，[energy_summary.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2/energy_summary.csv>)，[failure_breakdown.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2/failure_breakdown.csv>) | 实际资源和能量；SOURCE_UNFIT 能量 NA；未改旧图质量 |
| T4 新四臂计时 | [timing_per_call.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2/timing_per_call.csv>)，[timing_summary.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2/timing_summary.csv>) | 4/10/19 dB；development16×3 实测重复；576 实测/576 预热，真实 exit0 |
| T4 旧外部计时 | [README.md](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T4_resources_cost/external_timing_reuse_v1/README.md>)，[external_timing_single_output.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T4_resources_cost/external_timing_reuse_v1/external_timing_single_output.csv>) | 7/13/19 dB，P/Swin/native256 BPG；642 原输出 SHA 验证；与新计时分列、不补缺点 |
| T6 | [review.json](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T6_budget2048/consumer_review_v1/review.json>) | 仅独立消费者工程审查，不是实际科学完成；执行与最终结果由根任务补入 |

## 实际图片入口

每个下列 stem 对应 `.pdf`、`.svg`、`.png`，完整逐文件绝对路径与 SHA 见 [delivery_files.csv](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/N1024_delivery_index_v1/delivery_files.csv>)。独立子图与所有全样例页都在各组目录。

| 图组 | 入口 | 范围 |
|---|---|---|
| 原完整尺度与部分尺度四指标曲线 | [PNG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/fig_t5_same_prior_N1024.png>) / [PDF](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/fig_t5_same_prior_N1024.pdf>) / [SVG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/fig_t5_same_prior_N1024.svg>) | 原500、六SNR；2×2与四独立子图 |
| 原 partial−complete 配对增量 | [PNG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/fig_t5_partial_minus_complete.png>) / [PDF](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/fig_t5_partial_minus_complete.pdf>) / [SVG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/fig_t5_partial_minus_complete.svg>) | 原500、六SNR；7 dB零及13 dB小增量保留 |
| 完整外部五列固定样例 | [10 dB PNG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/external_completed_v1/fig_t5_external_completed_N1024_10dB_page01.png>) / [10 dB全16 PDF](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/external_completed_v1/fig_t5_external_completed_N1024_10dB_all16.pdf>) | 1/10/19 dB、全部固定16、每SNR四页；共12页，BPG/Swin实际缺项已补完 |
| 第五熵码列机制样例 v2 | [19 dB PNG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/mechanism_entropy_completed_v2/fig_t5_mechanism_entropy_N1024_19dB_page01.png>) / [19 dB全16 PDF](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/mechanism_entropy_completed_v2/fig_t5_mechanism_entropy_N1024_19dB_all16.pdf>) | 4/10/19 dB、固定16、共12页，EC_VAR由校准预选；不得使用排版拒收v1 |
| 两熵码与原部分尺度曲线 | [PNG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/entropy_quality_completed_v1/fig_t5_entropy_quality_N1024.png>) / [PDF](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/entropy_quality_completed_v1/fig_t5_entropy_quality_N1024.pdf>) / [SVG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/entropy_quality_completed_v1/fig_t5_entropy_quality_N1024.svg>) | 4/10/19 dB，500×3；2×2与四独立子图 |
| EC−raw partial 配对增量 | [PNG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/entropy_quality_completed_v1/fig_t5_entropy_minus_partial_N1024.png>) / [PDF](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/entropy_quality_completed_v1/fig_t5_entropy_minus_partial_N1024.pdf>) / [SVG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/entropy_quality_completed_v1/fig_t5_entropy_minus_partial_N1024.svg>) | 方向未翻转，19 dB熵码优势及零一致率保留 |
| 四方案质量与 TX 代价 | [19 dB mean PNG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/quality_tx_cost_completed_v1/fig_t1_quality_vs_tx_mean_N1024_19dB.png>) / [PDF](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/quality_tx_cost_completed_v1/fig_t1_quality_vs_tx_mean_N1024_19dB.pdf>) / [SVG](<C:/Users/11946/Documents/ChatGPT/comm/results/wcl_evidence_closure_20261009/T5_figures/quality_tx_cost_completed_v1/fig_t1_quality_vs_tx_mean_N1024_19dB.svg>) | 4/10/19 dB，各mean与median；共六组2×2，TX轴为log(ms)；质量500与计时16不同总体 |

固定源顺序为 `[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]`。全部为 development 样例，不伪称 500 图随机展示。page01 用清单前四个源；所有16源全部交付，不按各方法挑噪声。Direct 和 partial+VAR 共享原实际接收 token 与 Dc，Direct 未传残差置零、不是未知 token 填索引0。不同方法仍保留各自预定 noise namespace；同源不意味着同 RX。

原根目录 `T5_figures` 的 external13dB 五列、1/10/19 四列机制图继续有效，原 README 中缺图说明是生成时状态；后续补齐以本索引的新子目录为准，不修改旧 README/seal。`mechanism_entropy_completed_v1` 因标题裁切被拒收，仅 v2 交付。synthetic N2048 排版测试不得纳入论文数据图。

已存在的完整外部曲线：[fig_adaptive_bpg_main_N1024.png](<C:/Users/11946/Documents/ChatGPT/comm/paper/figures/adaptive_bpg_holdout500/fig_adaptive_bpg_main_N1024.png>)。HiFi 仅在真实共同 N1024/13 dB 既有子集单独展示：[hifi_development_N1024_13dB_group02](<C:/Users/11946/Documents/ChatGPT/comm/paper/figures/hifi_development_N1024_13dB_group02>)；没有把其他 SNR 图片当成缺失工作点，也没有新增 HiFi 采样。

所有当前 T5 图组已有实际 PNG 打开检查记录，见各目录 `visual_QA.json`。本次只复核这些已完成文件的 SHA，未冒称重新逐张打开。重建页的照片不是矢量重建，但标题可编辑；数据曲线 PDF/SVG 为真正矢量。

## 可重现脚本与版本差异

本地脚本根目录：`C:/Users/11946/Documents/ChatGPT/comm/experiments/wcl-evidence-closure-20261009/scripts`。远端仓库：`/home/liulu/projects/VAR_COMM`。根任务已报告远端交付 SHA 验证完成，本次整理未使用 SSH。远端归并凭据：`outputs/WCL-EVIDENCE-CLOSURE-20261009/N1024_final_small_deliveries_v1_remote_verification.json`；它是根任务提供的路径，本索引不冒称独立读取远端。

- 原六点图：`plot_existing_t5.py`。
- 外部 fixed16 已封印生产脚本 SHA `9cfd10f7b84ec2fa0f6ded2c14f6a1db7042337c74f05ac0190466651fefb3b4`，保留在**远端** `experiments/wcl-evidence-closure-20261009/scripts/plot_t5_completed_examples.py`。
- 第五列机制 v2：SHA `279a229c7c50b5d17c7f3215aad4f743fe943f3c16241a07da2016e52f74a5f4`。本地为 `plot_t5_completed_examples.py`，**远端实际安装名称为 `plot_t5_completed_examples_delivery_279a229c.py`**。原 README 的脚本名不修改；重现 v2 时请用对应内容哈希，不能误用远端旧同名脚本。新版也支持外部模式，但不是旧外部 production seal 的原脚本身份。
- 熵码曲线：`plot_t5_entropy_quality.py`，SHA `ed901d0b6049bc57afb66c0c45b0e71fb46133a4bbc7721c57a0dcd6b9981491`。
- 质量/代价图：`plot_t1_quality_tx_cost.py`，SHA `a6798d9a4172a5cff92a49a5f5741bd150754adec18df674dc2852b9c4b1e47e`。

重现所有图时使用新空输出目录，输入真实完成资产和冻结文件。机制 v2 远端入口示例（路径参数由已封印 manifest 绑定）：

```text
python experiments/wcl-evidence-closure-20261009/scripts/plot_t5_completed_examples_delivery_279a229c.py --mode mechanism --root . --entropy-manifest outputs/WCL-EVIDENCE-CLOSURE-20261009/T5_examples/entropy_fixed16_v1/display_manifest.json --entropy-assets outputs/WCL-EVIDENCE-CLOSURE-20261009/T5_examples/entropy_fixed16_v1 --entropy-freeze outputs/WCL-EVIDENCE-CLOSURE-20261009/T1_entropy_whole/selected_entropy_family.json --out results/wcl_evidence_closure_20261009/T5_figures/mechanism_entropy_REPLOT
```

旧和新版输入命令更多选项见各组 README；原结果不覆盖。`delivery_files.csv` 列每个本地文件的准确绝对路径、仓库相对路径与对应远端仓库路径，不声称每个二进制都已 git 跟踪。发布继续遵循已有过滤规则，不为了 PDF/PNG 改过滤器。

## 仍未关闭的任务

T6：预定第二预算 N2048 的实际测试、评分、源配对统计和跨预算机制图等待执行闭环。这里不发布任何 N2048 科学均值。完成后根任务应依据真实结果填入总报告的第4问及对第5问的影响，保留零/负结果与不同源总体边界。
