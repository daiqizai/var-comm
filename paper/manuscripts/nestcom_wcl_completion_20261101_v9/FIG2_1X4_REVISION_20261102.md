# Fig. 2 横排修订：2026-11-02

在当前活动 V9 稿上接续，基准提交 `63881bdd38bb82f3c4b0a9c5b2c2de01b9a4c4ea`。

- Fig. 2 改为通栏 1×4：左组固定 N=1024，右组等总信息预算；DISTS / DreamSim 分别对应两组面板。
- 实测正文 textwidth 为 516 TeX pt，即 181.353259 mm；图为 181.353259 × 64 mm。刻度/图例 7.5 pt、轴标题 8 pt、面板标题 8.5 pt。
- 92 个均值、92 组置信区间全部使用原 CSV，面板点数 28/28/18/18。输入 SHA256：`a2122c6568f2f4cf2211ed9a303d9370a9b376d634194e22839750ba2687435a`。右组仅将 bits 除以 1000；六个预算及真实数值间距全部保留。
- 图外两行共用图例；Ada-TokenCom 的 predicted-class、fixed-frame 适配在设置与图注中说明。正文分别解释完整系统比较 (a,b) 和等信息预算比较 (c,d) 的作用。
- PDF/SVG 为矢量；PDF 字体嵌入，SVG 保留文字；PNG 为 600 dpi。`figure2.*` 是同字节兼容副本。印刷尺寸与灰度预览同时更新。
- 双栏 PDF 5 页、双倍行距单栏 PDF 15 页、Word 14 页，34 页均实际打开检查。Word 保留原生正文、3 张表与 59 个 OMML 元素。Fig.1、Fig.3、数据、表格与 27 页补充材料原字节保留。
- Word 标准渲染脚本因本机无 LibreOffice 无法运行，沿用已安装 Microsoft Word 16 原生导出后检查全部页面；未将此失败标成成功。
- 本轮没有模型、PHY、指标、统计重算或新训练。原历史审阅/发布回执属于基准版本；当前交付状态看本轮外部发布回执。

## 重现

在本目录执行 `python scripts/draw_figure2_1x4.py`，再按 README 的两遍 pdflatex 与 Word 构建流程生成交付。`scripts/draw_figure2.py` 转调当前入口。所有输入/输出哈希和实际验收见 `validation/fig2_1x4_20261102/` 及当前 `validation/artifact_review.json`。

## 技能与来源

采用附件绘图脚本；参考项目历史 `fig2_main_1x4`、resolved_style、template_geometry 和 SKILLS_USED。应用 figures4papers 的最终尺寸、字体、CI、矢量和简洁配色规范，以及本地已存 K-Dense scientific-visualization 1.4 的出版与色彩规范。确切技能路径和 SHA256 记录在 `skills_and_provenance.json`，不冒称此前本地副本是最新上游版本。PDF / documents 技能用于编译后的实际逐页验收。

软件工作流引用：Kassis et al. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。此为技能使用说明，不新增论文科学引用。

附件没有单独的 `figure2_caption_candidate.tex` 文件；当前图注以完整执行指令及 V9 原图注为依据重建，并保持 `fig:performance`。
