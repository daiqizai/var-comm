# Fig.3 最终选图与接入 — 2026-11-02

以远端 main 7c7f04f31eae24561a898961434da158662bbe10 为基准，落实用户选择 C10-016 / C10-017。活动目录继续使用 V9。

Fig.3 为 N=1024、10 dB、两行五列：Source / WHOLE / SwinJSCC / Ada-TokenCom / NeST-Com。两源索引为 135 和 201；全文 ID、原像素身份、模型输出位置、PNG 与科学数组哈希均在 inputs/fig3_selected_20261102/native_mapping.json。原始每方法第一信道重复保留：WHOLE/NeST/Ada seed 6201，Swin seed 2001，不声称同噪声。

图下显示 DISTS / DreamSim 四位小数。NeST 两例分别为 0.1470 / 0.0742 和 0.1356 / 0.1718，均低于图中另外三方法。它们是在完整候选图库查看后选择的感知优势示例，不代表随机样本或全体排名。Swin 的 PSNR 两例更高；Ada 在第一例 LPIPS、第二例 CLIP-I 更好，完整五指标 CSV 与正文/补充材料保留这些取舍。总体 10 dB 的 Ada 优势及原始聚合结论未改。

图尺寸 152.4 × 70.358 mm，完整 256×256 视野，无裁切、拉伸、增强或补图。标题 8.5 pt，指标 7.5 pt，PDF/SVG/600 dpi PNG 和印刷/灰度预览齐备。当前 Figure 1、Figure 2、全部旧输入数据和三张表逐字节不变。旧 7 dB 示例及中位/退化示例继续保留在补充材料。

双栏 PDF 5 页；双倍行距单栏 PDF 15 页；可编辑 Word 15 页；补充 PDF 27 页。共 62 页已实际逐页打开检查，另检查 Fig.3 的实际尺寸、300 dpi 和灰度预览。三个 LaTeX 文件各两次真实编译 exit 0。Word 由当前正文与表格生成，包含 3 张原生表、6 个显示公式及 59 个 OMML 元素；因本机没有 soffice，保留打包渲染入口失败记录，使用已安装 Word 16 导出并检查，未沿用旧 PASS。

新增推理、PHY、指标计算、统计、训练、调度均为 0；复用 8 个既有重建与其既有五指标，2 张源图。科学原数组与日志仍在 PFS。正常仓库检查及 Git 发布状态以 validation/repository_checks.json 和本轮外层 publication_receipt.json 为准。
