# VAR_COMM：A 类补充成果索引

本轮首轮实验、核心 holdout 评分和图表已实际完成。原主结果版本为 `252176e041758ecb2d3e81b6fde5b587e7e17bb7`：N1024、六档 SNR、500 来源、每来源三次噪声；原结果和区间保持不变。
新增自适应降采样 BPG 是后来加入的冻结基线评测，完成 9000 帧实际接收、四指标均值及配对统计。条件性的 Swin 配置扩展和 A6 外部性能实测仍保留各自的 NOT_RUN/UNSUPPORTED 状态。
完整执行说明见[交付报告](../../reports/paper_supplement_20261009.md)，文件身份见[交付索引](DELIVERY_INDEX_20261009.json)。
交付报告和交付索引中的未提交状态记录初次本机交付时刻；Git 发布状态以仓库提交历史为准，原科学完成凭证保持不变。

## 成果地图

| 内容 | 直接入口 | 口径 |
|---|---|---|
| A0 复用与实现核验 | [覆盖表](a0_reuse/coverage.csv)、[检查凭证](a0_reuse/implementation_checks.json) | 核心必需缺项与条件性缺项分开记录 |
| A1 自适应 BPG 正式结果 | [均值与区间](a1_bpg_adaptive/metrics_v1/summary.csv)、[配对差值](a1_bpg_adaptive/metrics_v1/paired.csv)、[英文稿件](a1_bpg_adaptive/metrics_analysis_v1/manuscript_results.tex) | 500 来源 × 六 SNR × 三噪声；新比较单独统计 |
| A1 资源与失败 | [资源表](a1_bpg_adaptive/a1b_adaptive_v1/resources_v1/resource_summary.csv)、[失败分解](a1_bpg_adaptive/a1b_adaptive_v1/resources_v1/failure_counts.csv)、[实际发现](a1_bpg_adaptive/a1b_adaptive_v1/analysis_v1/ACTUAL_RESOURCE_FINDINGS.md) | 完整 BPG 文件计费；固定源码分辨率/QP 规则 |
| native256 BPG | [原结果](bpg_common500/summary.csv)、[新增配对](native_bpg_paired_v1/paired.csv) | 与自适应降采样版本分开保留 |
| A2 Swin | [实现与支持范围](a2_swin/REPORT.md)、[20 张实际对齐](a2_swin/native_v1/native_parity.csv) | 固定 80k 检查点；同观测实现诊断 |
| A3 单输出成本 | [时延 CSV](a3_timing/final_v1/timing_single_output.csv)、[存储 CSV](a3_timing/final_v1/model_storage.csv)、[测量说明](a3_timing/final_v1/README.md) | 五方法、固定 16 张 development 图、7/13/19 dB |
| A4 配置与接收机制 | [配置资源](a4_resources/v2/resource_summary.csv)、[状态质量](a4_resources/v2/state_quality.csv)、[稿件分析](a4_resources/analysis_v1/MANUSCRIPT_ANALYSIS.md) | 原 500 图接收记录导出；无重推断或新 bootstrap |
| A6 最近邻 | [机制矩阵](a6_neighbors/comparison.csv)、[可运行性边界](a6_neighbors/REPORT.md) | 文献与作者代码核对完成；外部 native 性能未测 |

图组入口包含图注、作图数据和复现方式；PDF/SVG/PNG 的实际保存位置由各目录说明给出。二进制文件是否入库沿用现有发布过滤规则，不能把本机存在当成已上传。

| 论文图表 | 入口 |
|---|---|
| 原 N1024 主曲线、部分尺度消融、VAR 补全消融 | [图组说明](../../paper/figures/mainraw64_holdout500/README.md)、[实际作图行](../../paper/figures/mainraw64_holdout500/plot_data.csv) |
| 加入自适应 BPG 的五方法曲线及 adaptive−native 增量 | [图组说明](../../paper/figures/adaptive_bpg_holdout500/README.md)、[图注](../../paper/figures/adaptive_bpg_holdout500/captions.tex) |
| 13 dB 六列固定样例：native256 / adaptive 两版 | [native256 版](../../paper/figures/a5_six_methods_N1024_13dB_group02/README.md)、[adaptive 版](../../paper/figures/a5_adaptive_six_methods_N1024_13dB_group02/README.md) |
| 10/19 dB 四列机制图 | [固定样例与身份说明](../../paper/figures/a5_mechanism_N1024_10_19dB_group02/README.md) |
| A4 预算、失败和状态质量图 | [修订排版说明](a4_resources/v2/figures_r2/README.md)、[图注](a4_resources/v2/figures_r2/captions.tex) |
| 可直接插入稿件的成本与配置表 | [时延/存储 LaTeX](../../paper/tables/paper_supplement_20261009/timing_single_output.tex)、[数字配置 LaTeX](../../paper/tables/paper_supplement_20261009/selected_digital_configs.tex) |

## 必须保留的结果与边界

- 自适应 BPG 六档均有 500/500 来源可编码；native256 的低码率不可行不能概括传统数字方案。仅 7 dB 有 15/1500 帧正文 CRC 拒绝并按冻结规则输出灰图。1 dB 有 82.4% 来源使用 32×32 输入；19 dB 仍有 54.8% 使用低于 256×256 的输入。
- 自适应 BPG 在六档的 PSNR 均高于本文方法，源级配对 95% 区间排除零；本文的 LPIPS、DINOv2-L 和原图预测一致率更好。应描述失真与感知/表征指标的取舍，不能宣称全部指标全面胜出。自动表征和预测一致率不等于人工验证的语义正确性。
- 部分尺度允许退回完整尺度配置，两者有相同调制与编码权限。7 dB 消融增量为零，13 dB 很小；19 dB 一致率差 +1.00 百分点，区间 [−1.80, 3.80] 跨零。KEEP 的 CRC 拒绝不自动成为灰图，低 SNR 的可靠性代价保留。
- A3 的 BPG 是 **native256**，不是自适应降采样搜索。其 7/13/19 dB 只有 3/12/30 个实测链路样本，对应 1/4/10 个来源；RX/E2E 为条件均值。SOURCE_UNFIT 的真实源码尝试成本单列，缺失 RX/E2E 不补零。19 dB 部分尺度 + VAR 的 E2E 均值高于完整尺度 + VAR，负结果保留；不混入 HiFi 旧环境的耗时。
- Swin 标为 **SwinJSCC-80k (adapted)**。20 张实现对齐通过不证明头部全局最优或充分收敛。C7 未训练，C13 正文超 N1024；19 dB 始终在训练和校准范围外，原 500 图策略未替换。
- A5 使用预定 development 固定源和噪声，不冒充共同 500 图 holdout。HiFi 只展示实际拥有的同源、同 N、同 SNR 缓存，不用其他 SNR 替代；不同波形不统称同一观测。
- A6 完成的是机制和可复现性核对。ARPC、Ada-TokenCom 外部 native 性能为 **NOT_RUN**；现有 ARPC 实现尚未覆盖本项目的付费 N1024 无线协议适配。不把内部完整尺度或算术方案重命名成外部方法。

## 数据精度与重现

所有数值以原精度 CSV 为准；Markdown 和 LaTeX 仅做显示四舍五入。DINOv2-L 字段为 `dinov2_vitl14_cosine`；ConvNeXt 是原图预测一致率，均值显示为百分数，配对差显示为百分点。LPIPS 差值不翻号。
新 BPG 选择仅使用旧校准 **100 来源**，不是 1000；原 500 图用于后来加入的冻结基线评测。没有因补充分析重算原已发布 bootstrap 区间。

```text
python scripts/plot_paper_adaptive_bpg.py --data-dir results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1 --output-dir paper/figures/adaptive_bpg_holdout500_replot
python experiments/paper_supplement_20261008/export_paper_tables.py --root .
```

作图输出使用新的独立目录。上述命令只读取完成的数据并排版；不训练、不推断、不模拟信道、不重新 bootstrap。实际模型/收发重现还需要对应目录记录的环境、权重、输入资产和协议。

三张超过单文件限制的逐帧表以[无损文本分片](large_table_parts_v1/README.md)发布。在仓库根目录运行 `python results/paper_supplement_20261008/large_table_parts_v1/restore.py --output-root .` 可恢复原文件；恢复后逐字节SHA须与清单一致。这只合并已有文本，不进行评分或统计。原完成凭证中的远端绝对路径是执行身份及哈希记录，并不表示所有缓存都随Git发布。依赖完整缓存的审计和实验入口不保证在裸克隆中直接运行。

## 保存范围与迁移状态

本轮本机保存了核心结果表、逐帧指标、500 个源级评分检查点、必要完成凭证、图表和固定展示缓存。完整 holdout 浮点重建、参照特征等大缓存仍在原服务器；模型权重、源码候选缓存及第三方阅读缓存也不由本页承诺全部入库。
`paper/VAR_COMM_Aclass_review_20261009.zip` 是本机审阅包，省略大模型权重和完整重建缓存；它不是 GitHub 下载资产的声明，也不是完整项目备份。
**完整迁移归档尚未完成。** 换服务器继续工作前仍需按完成凭证中的路径清点和转移依赖资产，不能仅凭本页、源码仓库或审阅 ZIP 认定可以完整续跑。
本轮没有自动启动 B/C 类实验或新增主模型训练；15 分钟监控保持暂停。
