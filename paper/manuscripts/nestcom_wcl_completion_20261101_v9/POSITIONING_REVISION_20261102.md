# NeST-Com 当前 V9：论文定位与证据组织修订

本轮基于 `99015482bafb11a3a65d9ab4a72a22a6210da3eb`，接续用户关于 ARPC 定位、普通前缀截断和创新边界的讨论。活动目录不变；保留此前历史稿和验收记录。

## 当前论证

全文主线改为：**保留尺度级预测，让传输可以在尺度内部结束；通过独立终止码流的实际长度选择可装入数字帧的前缀。** 引言依次交代固定帧问题、next-scale 与 ARPC 的源编码背景、预测单元与传输单元可以分离的设计动机，以及对应证据。

ARPC 是最接近的渐进源编码先行工作；不将其原论文称为已实现的同任务通信系统，也不把 NeST 描述成直接使用 ARPC 代码或全部模型组件。已完成的 ARPC+LDPC 是本项目数字链路适配，质量、付费 caption 和处理时间完整保留在补充 S1/S2/S7。

VAR 的 known-token overwrite、残差递推与后续条件更新明确归属原方法。A0/B0/C0 用来解释实际收到的边界 token 的两种作用，不再承担“接收端算法原创”的论证。主策略结果、固定粒度控制和完整系统比较各自说明其能支持的结论。

“优于普通前缀截断”不能作为未定义的总体结论。若普通前缀方案也正确执行独立终止、实际码长装入和相同递推，其核心过程可能与当前方法一致；现有证据支持相对完整尺度传输及匹配网格的具体收益，不支持对所有前缀算法的排他性创新主张。

## 图表与数值

- Fig.1 用户最终 PPTX/PDF/SVG/PNG 与 Fig.3 已选两例保持原字节。
- 正文 Fig.2 为六方法，84 个均值与 84 组 CI，面板数 24/24/18/18；补充 Figure S1 为完整七方法，92 个均值与 92 组 CI。
- 原始 92 行 CSV 不变。Table II 原七行输入不变，正文显示六行，ARPC 质量与时间完整保留于补充。
- Fig.2 尺寸 181.353259 × 64 mm；600 dpi PNG、矢量 PDF/SVG、印刷尺寸和灰度预览同步。
- 正文涉及 Swin/Latent/BPG 的 24 个 N1024 DISTS/DreamSim 配对区间已逐项核对。Ada 的高预算优势、Swin/BPG 的 PSNR 优势、JSCC 更低延时和个别退化样本均保留。
- 62 个已有 CSV、132 个已有输入文件的 SHA256 与基准一致。本轮科学调用、模型调用、PHY、指标、bootstrap、训练与计时均为零。

## 实际构建与审查

双栏 5 页，单栏双倍行距 15 页，可编辑 Word 15 页，补充 27 页。摘要按空白分词为 91 词。62 个最终页面均已实际查看；补充前三页与先前已查看图像 SHA256 完全相同，其余最终页重新打开检查。

LaTeX 三个最终构建均实际 wait=0，进程已退出；Word 为 3 张原生表、6 个展示公式、60 个 OMML 元素。打包 Word 渲染器缺少 LibreOffice，原失败保留，随后使用已安装 Microsoft Word 16.0 成功导出并逐页检查。未复制旧 PASS。

当前验收入口：`validation/positioning_20261102/`。`validation/current_revision.json` 指向本轮；其余旧轮次记录按其原范围保留。

## 发布与复现

唯一 Git 为 PFS `VAR_COMM` 的 main。本机仅为编辑与检查镜像。按既有授权进行精确文件暂存、必要仓库检查和普通提交推送；实际检查回执见本轮 `repository_checks.json`，提交/远端 SHA 回执保存在本轮独立 PFS 与本机执行记录中。没有强推或恢复任何暂停研究任务。

重绘主图：`python scripts/draw_figure2_1x4.py`。
重绘完整补充图：`python scripts/draw_figure2_1x4.py --include-arpc --stem figure2_complete_with_arpc --no-aliases`。
其余格式沿 `scripts/build_supplement_v9.py`、`scripts/build_word.py` 与当前 LaTeX 入口构建。
