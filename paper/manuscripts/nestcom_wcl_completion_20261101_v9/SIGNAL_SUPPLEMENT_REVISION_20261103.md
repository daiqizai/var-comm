# Fig.1 信令与补充材料整理（2026-11-03）

本轮沿用四段引言及 Table I(A/B/C)，只修订图文表述、阅读顺序与复现材料。基准提交 `027d5d7dcfb531f81d85b257f4f426d2c916984f`。未运行新实验、模型推理、指标、bootstrap、计时或训练。

- Fig.1 由原生 PPTX 修改后统一导出 PDF、SVG、PNG：header 输入改为 Profile ID；共享预配置策略向 TX/RX 提供 MCS a；profile decode 仅给出端点 e。保留三面板、蓝/橙语义及整体尺寸；(b)/(c) 原生图形保持。SVG 为路径化导出，PPTX 为可编辑文字与图形源。
- 正文区分 NULL profile、NULL frame 和解码失败，明确有效 NULL 使用完整 N 符号、两位零 payload；灰图保留统计分母。定义独立终止算术码长及 source-bit capacity utilization，区分 token prefix 与 prefix bitstream。
- 明确分别测量 GPU 同步的 TX/RX，total 为二者之和；Ada argmax 结论限定 auxiliary configurations。原计时数值与科学结果未改。
- S1 开头加入算法概要/阅读索引，保留协议与紧凑配置索引。完整环境、哈希、路径、LDPC 构造和校准得分集中于实际可下载的复现包。S3 改为 Matched Endpoint and Receiver Controls。S7 依次呈现主计时、粒度、编译 Ada 与其配对质量、历史 SMT 和实现诊断；两类 CPU 协议不合并。
- 复现包含 77 个文件，所有清单哈希与相对链接已核对。两张移出的长表完整保留；全部原有数值与置信区间未变。权重、数据集和原生图像缓存仍遵守原存储策略，不宣称私有 PFS 路径是公开下载链接。

交付与验收：双栏 **5 页**、单栏 **15 页**、补充 **35 页**，同步 Word **15 页**；摘要 **89 词**，正文仍为 **2 处**集中补充指引。70 页在本轮实际渲染并查看；最终未发现裁切、遮挡或未解析引用。Word 保留 4 张原生表、6 个编号公式及 74 个可编辑 OMML 元素。

[双栏正文](manuscript_doublecolumn.pdf) · [单栏正文](manuscript_singlecolumn.pdf) · [补充材料](supplementary.pdf) · [Word](manuscript_review.docx) · [可编辑 Fig.1](figures/figure1_final.pptx) · [复现清单](REPRODUCTION_CHECKLIST.md) · [复现包生成入口](scripts/build_reproduction_companion.py) · [本轮验收](validation/signal_supplement_20261103/artifact_review.json)。正常 main 发布状态以 Git 历史及发布回执为准。

仓库按既有规则保存复现包内的逐项文件及生成脚本，不跟踪 ZIP。完整 `reproduction_companion.zip` 随本地/PFS 文件交付，可由上述脚本在 Git checkout 中重建。
