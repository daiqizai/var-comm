# VAR_COMM A类补充实验交付

执行依据：`VAR_COMM_remaining_experiments_20261008.md`。原主结果版本为 `252176e041758ecb2d3e81b6fde5b587e7e17bb7`；原 N1024 六档 SNR、500 来源、每来源三次噪声的结果保持不变。

## 当前实际状态

截至2026-10-09，A类首轮所需科学运行和图表已实际完成。以下状态依据实际退出凭证和输出，不依据后台启动或脚本准备状态。A2未满足条件的配置扩展、A6外部native性能实测仍明确保留为NOT_RUN/UNSUPPORTED。

| 项目 | 已核实的实际产物 | 剩余工作 |
|---|---|---|
| A0 复用与实现检查 | 161行覆盖表、17项检查通过，首轮必需缺项为0 | 以最新覆盖表区分本地核验和远端保留资产；HiFi common500等条件性缺项仍保留 |
| A1 自适应降采样 BPG + LDPC | 32张源码摸底3076次编码；20张往返检查；100张校准5400帧；500来源×6 SNR×3噪声的9000帧、24个均值/区间、96条新配对结果、资源/失败表和曲线均完成 | 原分辨率基线独立保留，另补72条native256配对结果 |
| A2 Swin 实现与配置 | 20/20 native 同观测对齐通过，44 个输出下载核验；支持性限制已记录 | 首轮无新增付费策略，复用原500图；不扩展未训练通道 |
| A3 单输出成本 | 五方法共 960 次预热/测量尝试，1080 次 PHY 调用；1038 个输出哈希通过；时延、显存、参数和存储表已导出 | 无新增计时 |
| A4 资源与失败机制 | 24 个配置、36000 个帧家族记录、54000 个质量记录；144 个原均值完全一致；四组图已生成和查看 | 无新科学运行 |
| A5 固定样例和参照 | 13 dB的native256与adaptive BPG两版六列图、10/19 dB四列机制图、逐图指标及100图六种参照均已导出；PNG已实际打开检查 | adaptive新增16帧实际收发正常结束；15张新评分、1张同图结果复用 |
| A6 最近邻核对 | ARPC、Ada-TokenCom的18项机制差异、作者源码和可运行性说明 | 外部native性能实测为NOT_RUN；首轮不自动下载或改造大模型 |

## 可直接复用的图表与说明

- 原三组主结果与两项消融：`paper/figures/mainraw64_holdout500/`。
- 五方法统一500图曲线及adaptive减native配对图：`paper/figures/adaptive_bpg_holdout500/`，两张组合图加八张独立子图，每张均有矢量PDF、可编辑文字SVG、600 dpi PNG。
- adaptive BPG正式数据：`results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1/`，包含原精度`summary.csv`、`paired.csv`、`source_means.csv`、9000帧`rows.json`和500个源检查点。
- 新基线英文稿件段落、全部96条比较及中文结论边界：`results/paper_supplement_20261008/a1_bpg_adaptive/metrics_analysis_v1/`。
- native256 BPG的72条新增配对：`results/paper_supplement_20261008/native_bpg_paired_v1/paired.csv`。
- adaptive码长、资源、全部失败状态与六个完整BPG码流样例：`results/paper_supplement_20261008/a1_bpg_adaptive/a1b_adaptive_v1/resources_v1/`。
- 单输出耗时表：`results/paper_supplement_20261008/a3_timing/final_v1/timing_single_output.csv`；其中 `timing_table.csv` 是字节相同的别名。
- 显存、参数与权重：同目录的 `model_storage.csv`、`all_summary_rows.csv`；运行环境和样本口径见 `README.md`。
- 资源、失败状态、源码信息与质量：`results/paper_supplement_20261008/a4_resources/v2/figures_r2/`，每组均有 PDF、SVG、600 dpi PNG。
- A4英文稿件分析及真实配置表：`results/paper_supplement_20261008/a4_resources/analysis_v1/MANUSCRIPT_ANALYSIS.md`。
- 13 dB六列native256 BPG样例与80个重建指标：`paper/figures/a5_six_methods_N1024_13dB_group02/`。
- 13 dB六列adaptive BPG样例与80个重建指标：`paper/figures/a5_adaptive_six_methods_N1024_13dB_group02/`；另16个原图参照单元格明确不计分。原五列逐像素一致。
- 10/19 dB固定机制图和原逐图指标：`paper/figures/a5_mechanism_N1024_10_19dB_group02/`。
- Swin对齐与范围限定：`results/paper_supplement_20261008/a2_swin/REPORT.md`。
- 最近邻比较：`results/paper_supplement_20261008/a6_neighbors/comparison.csv`、`REPORT.md`。
- 可直接插入稿件的时延、存储及六档数字配置表：`paper/tables/paper_supplement_20261009/`中的两个LaTeX文件。

## 新BPG结果对论文结论的影响

自适应降采样BPG在全部六档均有500/500源可以编码进容器；因此原分辨率BPG的低码率不可行现象不能概括传统数字方案。仅7 dB有15/1500帧正文CRC拒绝并输出冻结灰图，其他五档均1500/1500实际解码。所有失败都计入均值。1 dB有82.4%的源选择32×32输入；19 dB仍有54.8%选择低于256×256的分辨率。

自适应BPG在全部六档的PSNR高于本文方法，源级配对95%区间均高于零；本文在LPIPS、DINOv2-L及ConvNeXt原图预测一致率上优于该基线，六档的配对区间均排除零。结论是失真与感知/表征指标之间的取舍，不能写成本文在所有指标上全面优于传统数字系统，也不能把表征或一致率优势直接写成人工验证的语义正确性。

例如13 dB，自适应BPG为23.128 dB、LPIPS 0.53915、DINOv2-L 0.42792、一致率33.6%。相对本文的差值为PSNR +2.84287 dB [2.75854,2.92977]、LPIPS +0.37678 [0.36542,0.38852]、DINOv2-L −0.33177 [−0.35200,−0.31142]、一致率 −48.0百分点 [−52.4,−43.4]。展示文本四舍五入，CSV保留原精度。

自适应BPG相对native256在四指标、六档均改善。与Swin的结论随工作点和指标变化：13 dB的DINOv2-L及一致率差值区间跨零；19 dB自适应BPG四项更好，但该点仍在Swin训练与校准范围外。各方法和所有工作点都保留，不能删除有利于外部基线的结果。

## 写入稿件时必须保留的口径

统一计时在同一 RTX4090D 环境下对固定16张 development 图进行：7/13/19 dB，每case预热1次、测量3次；batch1、实际单输出、GPU同步。13 dB 软件端到端均值（扣除单独实测的软件加噪窗口）为部分尺度+VAR 195.05 ms、完整尺度+VAR 266.54 ms、连续JSCC 21.44 ms、Swin 26.97 ms。19 dB部分尺度+VAR为260.15 ms，完整尺度+VAR为251.27 ms，保留这项负结果。

计时中的 BPG 是原分辨率 native256 版本，不是本轮新增的自适应降采样版本。其7/13/19 dB只有1/4/10个源能编码进容器，对应3/12/30个实测链路样本；接收与链路端到端均值仅针对这些样本。其余 SOURCE_UNFIT 的真实源码尝试成本单列，RX/E2E为空，不能当零时延或总体部署平均。

Swin使用固定80k检查点和付费信息适配。20张native诊断中最大RGB绝对差为5.364418029785156e-7；这不证明256符号头部全局最优、模型充分收敛或未训练通道有效。19 dB仍超出训练和校准范围。C13正文需要1664符号；C7可被代码接受但没有训练，因此没有据此替换原500图策略。

部分尺度与完整尺度共享调制及编码权限，部分尺度可退回完整尺度赢家。7 dB增量为零；13 dB增量很小。19 dB一致率差为+1.00百分点，95%区间[-1.80,3.80]跨零，不能写成确定的一致率提升。CRC拒绝不等于灰图：KEEP保留硬译码token，低SNR的可靠性代价与全部失败状态都保留。

图像展示使用预定development固定组02的源[4,21,24,29]，不冒充共同500图holdout。HiFi只展示有真实同源同N同SNR缓存的13 dB点；不从其他SNR替换。Swin与HiFi共享实际观测，其他波形只共享源图和工作点，不声称观测完全相同。不按方法挑最好的噪声。

新BPG策略只在旧校准100源选择，目标为全帧逐图PSNR经三噪声、源级平均；500图只用于后来增加的冻结基线评测。允许的非学习降采样是该基线的源码处理，恢复后统一对原256×256图像评分。原分辨率结果另行保留，不能用其低码率不可行推出所有传统数字方案不可行。

新增adaptive BPG收发累计31,916次，低于独立34,604次上限；展示补图另用32次。原主实验534,231次账本未变化。四指标评分仅对2635个新增唯一图像调用，复用379个原native同图结果，500个参照特征另计。新统计仅针对新基线/新比较使用原10000次源级bootstrap规则，没有重算原已发布区间。

## 重现绘图

在已有numpy和matplotlib的环境中，从仓库根目录运行以下命令，指定尚不存在的独立输出目录。该命令仅读取完成的CSV并绘图，不启动模型、信道或bootstrap：

```text
python scripts/plot_paper_adaptive_bpg.py --data-dir results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1 --output-dir paper/figures/adaptive_bpg_holdout500_replot
```

固定样例的浮点缓存、方法映射与逐图指标保存在各图目录，原始收发证据路径在`collection.json`中。各图的图注、数据版本和重现入口见对应README。

## 保存范围

原科学文件、原区间和两项原消融没有修改。新实验、表格与图均保存在独立目录；没有新增主模型训练，没有启动B/C类条件性实验，也没有恢复15分钟定时调度。

本轮结果表、逐帧指标、图表、代码与必要完成凭证已保存在本机。完整holdout浮点重建和参照特征大缓存仍在原服务器；本报告不宣称整个项目的迁移归档已经完成。新增产物未自行提交或推送，也未修改发布过滤器。

便于审阅的图表与文字包为`paper/VAR_COMM_Aclass_review_20261009.zip`。它包含图、图注、作图数据、时延/配置表与结果说明，省略大模型权重和重建缓存；完整科学记录仍按上面的独立目录保存。

A4原运行日志在其完成文件生成后多写了最终一行，只有该日志的旧哈希发生差异；科学CSV全部核验通过。原完成文件保留，`v2/DELIVERY_MANIFEST.json`另行绑定闭合日志和新版图，排版修正没有重算原统计。
