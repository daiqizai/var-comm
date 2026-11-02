# Fig.1 局部修订 — 2026-11-02

活动稿仍为 V9，基准提交 `a60beb0e580b74356e1a9fbc02969af03cfebf55`。本次只修改 Fig.1、对应图注、导出来源和同步文稿。

(c) 保留原粗尺度历史、边界预测和蓝/橙覆盖网格。边界残差形成初始状态后，每轮按 `fhat_s = fhat_(s-1) + C_s(qtilde_s)` 更新；回路线同时接回下一尺度 context 和残差累加的保留状态，最终 `fhat_10` 进入 `D_c`。尺度范围在面板内。`m=9` 时该范围为空，初始边界状态就是最终状态；完整端点 `(10,0)` 的直接解码由正文说明。

(a) 两个物理层框明确仅处理 body。紫色支路从拟合选中的实际端点和固定 MCS 生成 protected profile header，合帧点位于 AWGN 前。接收端独立恢复 profile；`a` 控制 body 解码，`e` 控制算术解码。Noisy header、a、e 分开标注。

(b) 使用 Independent termination / actual-length check，并明确 Selected prefix / endpoint。相同 MCS、三条容量示意和预算公式保留，目标上限写入图注。

共 41 处数学标签改成原生 Office Math，正文使用 Calibri、数学使用 Cambria Math；PPTX 嵌入 Cambria Math，PDF 的实际字体为嵌入的 Calibri / Cambria Math。原 Nimbus Sans / STIXGeneral 缺失时发生的替换已消除。SVG 字形仍为路径，文字与图形编辑源是 PPTX。

画布保持 **181.864 × 86.360 mm**。双栏正文中的实际图宽约 **181.3566 mm**，图保持矢量。PDF/SVG/600 dpi PNG（4296×2040）及印刷尺寸、灰度预览由同一最终 PPTX 的 PowerPoint PDF 导出生成。

双栏 **5 页**、单栏 **15 页**、Word **15 页**。本轮检查全部 35 页和独立 Fig.1；最后两处标签微调后重新编译，只有双栏第 2 页、单栏第 3 页的栅格发生变化，已再次打开检查，其余页与本轮已查看版本逐字节相同。Word 最终 15 页全部打开检查。原 27 页补充材料字节不变，沿用上一轮验收。原 CSV、Fig.2/3、实验和统计不变。

## 编辑与导出

直接编辑 `figures/figure1.pptx` 的原生文本、公式和图形。若从保存的前版本重建，运行 `scripts/edit_figure1_recurrence.mjs <新临时目录>`；Node 需 `@oai/artifact-tool` 和 `jszip`，可由 `RUNTIME_NODE_MODULES` 指向已有运行库。然后运行 `scripts/export_figure1_native.ps1 -InputPptx <输入> -OutputDirectory <新输出目录>`。该脚本使用已安装 PowerPoint，并恢复 SaveCopyAs 取整前的 EMU 画布。`scripts/export_figure1_views.py <导出的PDF> <新输出目录>` 生成 SVG、600 dpi PNG、印刷和灰度预览。

论文仍使用原构建入口；Word 由同一 LaTeX 文本和当前图 PDF 生成。`validation/fig1_recurrence_20261102/` 保存本轮原生包、字体、数据不变、编译与视觉检查记录。旧报告和旧 PASS 为历史记录。
