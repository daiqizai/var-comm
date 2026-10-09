# Kodak24 冻结方法泛化实验：交付索引

本轮实际完成 24 张 Kodak 图像的固定中心 256×256 RGB 裁剪评估：N1024、SNR 4/10/19 dB、每来源每方法三次噪声、四种方法，共 864 个重建帧。所有权重与策略沿用冻结版本，没有训练、重新校准或在 Kodak 上选择新的调制编码配置。VAR 使用无条件 null 类别嵌入，不向接收端提供源图类别。

四种方法为：部分尺度数字传输与无条件 VAR 补全、潜空间连续 JSCC、自适应降采样 BPG + LDPC，以及 SwinJSCC-80k（adapted）。结果包含指标之间的取舍；不利结果和区间跨零的比较均保留，具体结论见结果报告。

## 结果与统计

| 内容 | 入口 | 说明 |
|---|---|---|
| 完整结果与限制 | [RESULTS.md](RESULTS.md) | 中文结论、英文段落、均值与配对比较、执行证据 |
| 可入稿英文结果 | [manuscript_results.tex](manuscript_results.tex) | 与实际 CSV 一致的结果叙述 |
| 推断前冻结协议 | [protocol.json](protocol.json) | 样本、预处理、SNR、噪声、冻结策略、预算和预定展示源 |
| 单方法统计 | [analysis_v1/summary.csv](analysis_v1/summary.csv) | 48 个方法×SNR×指标均值及 95% 区间 |
| 来源配对统计 | [analysis_v1/paired.csv](analysis_v1/paired.csv) | 36 个“本文方法 − 参考方法”差值及已有配对区间 |
| 来源均值 | [analysis_v1/source_means.csv](analysis_v1/source_means.csv) | 每来源先平均三次噪声，保留 1152 行 |
| 接收状态 | [analysis_v1/status_counts.csv](analysis_v1/status_counts.csv) | 成功与失败状态全部纳入，不按成功帧筛选 |
| 统计完成凭证 | [analysis_v1/completion.json](analysis_v1/completion.json) | 统计输入、输出与执行身份 |
| 逐帧四指标 | [metrics_v1/rows.csv](metrics_v1/rows.csv) | 864 帧×四指标，共 3456 行原精度数值 |
| 评分完成凭证 | [metrics_v1/completion.json](metrics_v1/completion.json) | 实际评分与来源特征准备计数，复用相同 RGB 的记录 |

CSV 是数值依据；报告与图表只进行显示格式转换。DINOv2-L 使用 `dinov2_vitl14_cosine`。ConvNeXt 指标为与源图预测的一致率，不是 Kodak 分类准确率；图中均值显示为百分数，配对差显示为百分点。LPIPS 差值保持“本文 − 参考”，负值更好，不翻转符号。

## 实际接收与配置证据

| 内容 | 入口 | 说明 |
|---|---|---|
| 执行证据索引 | [evidence/index.json](evidence/index.json) | 共同源清单、四方法帧索引与完成凭证、实际配置和监督等待记录 |
| 各方法实际完成凭证 | [actual_receipts/](actual_receipts/) | 顶层实际执行记录；启动记录本身不代表完成 |
| 逐帧资源与诊断 | [frame_resources.csv](frame_resources.csv) | 864 帧的接收状态、付费符号、码参数、已记录能量与接收后诊断 |
| 冻结工作点配置 | [frozen_configs.csv](frozen_configs.csv) | 四方法×三 SNR，共 12 个配置 |
| BPG 源侧配置 | [bpg_source_configs.csv](bpg_source_configs.csv) | 24 来源×三 SNR，共 72 项实际分辨率、QP、完整码流大小与身份 |
| 导出范围 | [RESOURCE_EXPORT.md](RESOURCE_EXPORT.md) | 字段含义、缺失项和缓存保存边界 |
| 导出完成凭证 | [resource_export_completion.json](resource_export_completion.json) | 只读资源导出的输入和输出哈希 |

BPG 仍采用原冻结的源侧分辨率/QP 搜索规则；源码失真只用于发送端编码，不观察噪声结果，也不据 Kodak 重选 MCS。接收后的正确性诊断不回馈接收器。原方法的失败输出保留在评分与图表中；软件执行失败不被伪装成灰图。缺失或不适用的资源字段不补零，例如 BPG 逐帧 JSON 未单独保存头部与正文能量时只保留实际总能量。

## 论文图与固定样例

| 图组 | 入口 | 交付形式 |
|---|---|---|
| 四指标主曲线与来源配对增量 | [paper/figures/kodak24_N1024](../../paper/figures/kodak24_N1024/README.md) | 两张 2×2 图；矢量 PDF、可编辑文字 SVG、600 dpi PNG、图注和实际作图行 |
| 4/10/19 dB 固定五列样例 | [paper/figures/kodak24_fixed4_N1024](../../paper/figures/kodak24_fixed4_N1024/README.md) | 三张 4×5 图；PNG/PDF/SVG、图注及 `display_mapping.csv` |
| 实际 PNG 视觉检查 | [visual_QA.json](visual_QA.json) | 本次实际打开图像的检查记录；独立于渲染完成凭证 |

样例严格使用协议预定来源索引 `[0,1,2,3]`、噪声种子 `2001`，所有方法共用同一原图和工作点；不按方法挑选最佳图像或噪声。照片直接取自已完成的浮点重建缓存，不锐化、不增强，也不替换失败输出。照片版 PDF/SVG 内嵌图像像素，文字保留为矢量；曲线图 PDF/SVG 为真正矢量图。

## 解释边界

- 这是 24 张固定中心裁剪图的有限跨图像集合证据，不是原尺寸 Kodak 全图传输，也不证明所有预训练模型的数据均与 Kodak 无重叠。
- Swin 的 19 dB 在训练和校准范围外，始终保留并标注；适配版不能表述为官方最优或充分收敛版本。
- 每来源先平均三次噪声，再按同一来源图配对。不同波形的方法没有共享同一次实际接收观测，相同噪声标签也不表示相同接收信号。
- 95% 区间为来源级、点态 percentile bootstrap 区间：本次新数据只使用一组 10,000 次重采样，种子 `2026100901`。没有重算旧结果区间；未作多重比较校正，不能据此宣称普遍全面优越。区间跨零也不证明等价。
- 潜空间连续 JSCC 沿用已验证的 A3 单图 Dc 批量 B1 路径。原 common500 的 Dc 为 B3，未宣称两种批量逐位相同。

## 安全重绘

在仓库根目录运行以下命令，仅读取已经完成的统计 CSV，向一个尚不存在的新目录导出论文曲线图；不会训练、推断、产生信道噪声或重新 bootstrap。若示例输出目录已经存在，请改用新的目录名。

```sh
python experiments/generalization_kodak_20261009/plot_only.py --data results/generalization_kodak_20261009/analysis_v1 --out paper/figures/kodak24_N1024_replot
```

本索引与轻量证据不构成完整服务器迁移包。源 PNG、浮点重建 NPZ、模型权重、完整 BPG 候选缓存及运行环境的远端路径和哈希保留在证据中；不能由 Git 中存在索引推断这些大资产已全部随仓库保存。发布状态以实际 Git 提交及推送记录为准，本页不宣称已推送。
