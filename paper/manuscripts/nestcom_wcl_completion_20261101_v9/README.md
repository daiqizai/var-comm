# 当前交付：贡献论证修订（2026-11-03）

主线改为“保持尺度级概率计算，按独立终止的实际码长细化固定帧传输端点”。正文加入码长性质、粒度与成本取舍；已有接收机制准确归属，相关工作以各自解决的问题组织。

[双栏 PDF](manuscript_doublecolumn.pdf) · [单栏 PDF](manuscript_singlecolumn.pdf) · [可编辑 Word](manuscript_review.docx) · [补充材料](supplementary.pdf) · [修订说明](CONTRIBUTION_NARRATIVE_20261103.md) · [当前验收](validation/current_revision.json)。页数 **5/15/16/36**，摘要 **88 词**，本轮72页实际审阅。三图、四主表与科学输入保持；无新实验。下方为历史快照。

--- Historical record / 历史记录 ---

# 当前交付：协议用词、策略索引与处理成本修订（2026-11-03）

当前 MCS/目录预配置和 12-bit endpoint profile 的含义已统一。补充 S1.3 集中索引新旧策略；正文补回 NeST/WHOLE 完整 TX/RX 耗时；运行历史移至复现日志。未新增实验，三图、四主表与既有科学输入不变。

[双栏 PDF](manuscript_doublecolumn.pdf) · [单栏 PDF](manuscript_singlecolumn.pdf) · [可编辑 Word](manuscript_review.docx) · [补充材料](supplementary.pdf) · [修订说明](WORDING_POLICY_REVISION_20261103.md) · [当前验收](validation/current_revision.json)。页数 **5/15/16/36**，摘要 **91 词**，全部 **72 页**已实际检查。下方均为历史修订快照。

--- Historical record / 历史记录 ---

# 当前交付：统一校准与补证据（2026-11-03）

统一N1024校准与18,000正式帧、N512严格匹配粒度、快速Ada32源质量、算术码长检查均实际闭合。新主结果、全部不利发现、完整候选表及紧凑策略表已接入。当前双栏/单栏/Word/补充为 **5/15/16/35页**，摘要91词；71页已逐页渲染并打开检查。

[双栏PDF](manuscript_doublecolumn.pdf) · [单栏PDF](manuscript_singlecolumn.pdf) · [可编辑Word](manuscript_review.docx) · [补充材料](supplementary.pdf) · [本轮结果与边界](UNIFIED_CALIBRATION_20261103.md) · [验收入口](validation/current_revision.json)。Fig.1用户原字节不变；Fig.2主84/补92点，Fig.3固定两源并更新相同第一重复。下方全部为历史修订记录，其页数、策略、数字与PASS不适用于本轮。

--- Historical record / 历史记录 ---

# 当前：Ada合理实现计时与V9交付

32源新编译收发已实际闭合，TX 114.55 ms、RX 2949.76 ms。旧串行开销不作Ada方法成本；新数值实现与旧质量分开报告。双栏/单栏/Word/补充 **5/15/15/30** 页，当前页面验收已完成。[修订说明](TIMING_PREFILL_20261103.md)。以下为历史状态。

---

# 当前计时公平性修订 — 2026-11-02

主表可报告计时已改用同一GPU、两个不同物理CPU核的32源复测。**Ada计时已从主表撤下**：本地串行实现未对齐原文prefill/torch.compile路线，统一硬件不足以解决该问题。原“两核”实为同核SMT对；审计诊断、prefill不等价检查和被排除的GPU干扰记录均保留。图像质量、冻结策略及三张主图不变。

当前双栏/单栏/Word/补充页数为 **5/15/15/29**，全部页面已重新检查。[修订说明](TIMING_FAIRNESS_20261102.md) · [结果与账本](../../../results/nestcom_timing_fairness_20261102_v1/delivery/REPORT_zh.md)。以下段落均为各自历史状态。

---

# 当前论文定位修订 — 2026-11-02

主线已调整为“保留尺度级预测，细化固定帧的传输端点”。ARPC 作为最接近的渐进源编码先行工作；其已完成数字适配结果移至完整补充比较。VAR 的覆盖与递推机制明确注明出处。主策略、匹配粒度和机制干预分别承担对应证据任务。

当前产物：[双栏 5 页](manuscript_doublecolumn.pdf)、[单栏 15 页](manuscript_singlecolumn.pdf)、[可编辑 Word 15 页](manuscript_review.docx)、[补充 27 页](supplementary.pdf)。62 页已检查；摘要 91 词。正文 Fig.2 为 84 点/84 CI，补充完整图为 92 点/92 CI；所有原始 CSV 和用户最终 Fig.1、Fig.3 不变。本轮未执行新实验。

详细修改与边界见 [本轮报告](POSITIONING_REVISION_20261102.md)，当前验收入口为 [本轮验证](validation/current_revision.json)。下方各轮页数、点数、摘要词数和发布状态均为其历史快照。

---

# 当前 Fig.1 最终用户微调版 — 2026-11-02

已检查并接入用户最后保存的布局，原 PPTX 字节保留。当前正文使用 [最终 Fig.1 PDF](figures/figure1_final.pdf)，编辑源为 [最终 PPTX](figures/figure1_final.pptx)。三面板文字/数学/信息流与方法和图注一致；双栏/单栏/Word 重新编译与逐页查看，页数 **5/15/15**。见 [本次检查记录](FIG1_FINAL_REVIEW_20261102.md)。下方 thin-v5/v6/v7 记录均为历史状态。

---

# Fig.1 下方面板布局跟进 — 2026-11-02

最新可编辑变体：[thin-v7 balanced PPTX](figures/figure1_thin_v7_balanced.pptx) · [PDF](figures/figure1_thin_v7_balanced.pdf)。左侧装帧框和容量条加长；(c) 标题、输入和递推内容跟进右移后的框体，并收紧横向间距。两面板已分别成组。详见 [布局和检查记录](FIG1_PANEL_BALANCE_20261102.md)。当前已编译论文引用未变，以下为历史记录。

---

# 用户手动编辑版 Fig.1 清理 — 2026-11-02

最新可继续编辑的变体：[thin-v6 clean PPTX](figures/figure1_thin_v6_clean.pptx) · [PDF 预览](figures/figure1_thin_v6_clean.pdf)。保留用户调整的位置，清除箭头重影和紫色框两侧残片。当前已编译论文仍使用先前的 thin-v5 PDF，本轮未替换正文图或重编译。见 [清理与验收](FIG1_USER_CLEANUP_20261102.md)。下方记录为各轮历史状态。

---

当前图的可编辑文件：[thin-v5 PPTX](figures/figure1_thin_v5_editable.pptx)。数学标签可取消组合后调整文字。

# 当前 Fig.1 箭头与对齐复核 — 2026-11-02

已进一步将普通重绘连线简化为直线或一次转折，保留一条必要反馈回路，并清理旧短线、重复框边及标签错位。Frame 为链路上方独立横向示意；稿件仍为 **5/15/15 页**。见 [本轮复核记录](FIG1_ARROW_ALIGNMENT_20261102.md)。下方历史记录中的“下方帧示意”描述属于前一提交。

---

# 当前 Fig.1 thin-v5 — 2026-11-02

已按用户新提供的 thin-v4 PDF 整理右下角、补回残差并简化箭头；按随后意见将帧结构移至信道下方横向示意条，居中求和节点，改用直角连线。当前正文采用 `figures/figure1_thin_v5.pdf`，双栏/单栏/Word 为 **5/15/15 页**。新图由用户矢量 PDF 与局部绘图脚本生成；旧 `figure1.pptx` 对应上一版，保留为历史源，不是 thin-v5 的编辑源。详见 [本轮修改与验收](FIG1_THIN_V5_20261102.md)。下列既有记录保留各自历史范围。

---

# 当前 Fig.1 局部修订 — 2026-11-02

Fig.1 保持三面板和原尺寸，补全了更细尺度递推反馈、保护头与主体的分离关系，并改用原生数学公式。双栏/单栏/Word 已重新生成，页数仍为 **5/15/15**。完整修改与新验收入口见 [Fig.1 修订记录](FIG1_RECURRENCE_20261102.md)。以下既有段落保留为历史记录，其中“全部图不变”等描述仅适用于对应旧轮次。

---

# 当前定稿修订 — 2026-11-02

已按 `NeST_Com_Final_Revision_Package_20261102.zip` 合并并独立重建，基准为 `eee31804dc73fb1e43c1734ea21554a55c174c3d`。首次交付为匿名稿工作目录修订；用户随后明确要求“推到git”，授权本轮正常提交与推送 main。原未发布验收记录保留为首次交付快照；后续授权见 `validation/final_revision_20261102/publication_authorization.json`。作者信息待作者提供。

当前产物：双栏 **5 页**（IEEE 10 pt，摘要 **100 词**）、双倍行距单栏 **15 页**、可编辑 Word **15 页**（3 张原生表、6 个展示公式、60 个 OMML 元素）、补充材料 **27 页**。62 页全部重新渲染并逐页检查。三张主图及原有 56 个输入 CSV、3 个表格 CSV 保持原字节；新增 Ada 校准附件为已完成结果的精确复制。

当前修订与验收以 [本轮报告](FINAL_REVISION_20261102.md) 和 [独立验收入口](validation/current_revision.json) 为准。下列各轮段落及旧 validation 文件均为历史记录，其中旧词数、公式元素数、页数和发布状态不代表本轮。

---

## 当前 Fig.3 修订 — 2026-11-02

已采用 C10-016 / C10-017 的 N1024、10 dB 两行五列对照，加入 SwinJSCC。图下 DISTS / DreamSim 均为对应原第一重复的实际指标。双栏/单栏/Word/补充材料页数为 5/15/15/27，62 页均已逐页检查。数据、取舍与来源见 [Fig.3 修订记录](FIG3_SELECTION_20261102.md)。下面保留历史版本记录；当前验证文件以本轮为准。

## Current figure revision — 2026-11-02

Fig. 2 is now a 181.353259 × 64 mm four-panel horizontal figure, integrated in all three main manuscript exports. All 92 means and 92 confidence intervals are unchanged. Actual output: 5-page double-column, 15-page single-column, 14-page editable Word; all 34 revised pages inspected. Build the figure with `python scripts/draw_figure2_1x4.py`. See [revision and provenance](FIG2_1X4_REVISION_20261102.md). The historical publication paragraph below describes the preceding V9 delivery; this revision has its own outer publication receipt.

# NeST-Com V9

Anonymous manuscript: **NeST-Com: Fixed-Budget Next-Scale Token Communication**.

- `manuscript_doublecolumn.pdf`: 5 pages, 10 pt, 99-word abstract.
- `manuscript_singlecolumn.pdf`: 15 pages, double spaced.
- `manuscript_review.docx`: editable body, 3 native tables, 59 OMML elements; 15 pages in Word.
- `supplementary.pdf`: 27 pages; full S1–S8 recipe/results.
- `main.tex`, `main_singlecolumn.tex`, `supplementary.tex`, `sections/`, `tables/`, `references.tex`: compilation sources.
- `figures/figure1.pptx`: native structural figure. Fig.2/3 scripts, PDF/SVG/PNG and print-size previews are included. Text editing for Fig.1 uses PPTX; path-based SVG glyphs are not represented as editable text.
- `inputs/`, `figure_table_sources.json`, `input_manifest.json`: pinned values and identities. Native scientific arrays and weights remain in PFS; display PNGs do not replace metric inputs.
- `validation/`: actual compile, data, Word and all-page visual review.
- `completion_matrix.csv`, `final_report_zh.md`: scope, results and limitations.

Build from this directory using `pdflatex main.tex` twice, similarly for `main_singlecolumn.tex` and `supplementary.tex`. Figure scripts use Python, NumPy, Pillow and Matplotlib. Word is built from the same LaTeX sections and table CSVs with `scripts/build_word.py`; its actual runtime and conversion receipt are recorded in validation. Reader-only CPU and asset verification entry points are described in `experiments/nestcom-v9-completion-20261101/README.md` at repository root.

Based on remote main `dde679637b30f1166bc540d4c7b6fa91fe457e8c`. The `repository_commit` field in `inputs/reproduction/reproduction_config.json` records this scientific base revision; the reader-verification scripts and exported reproduction files belong to the accompanying V9 artifact revision. V8 and all previous evidence are preserved. No training, new calibration, main-experiment replay, new timing or stopped branch was launched. Both required repository checks passed. Normal main publication was verified at artifact commit `c4e85c4d984e12b8c6e337f8cf97fd5cf8872b94`; validation/publication_receipt.json records it. A metadata-only follow-up records the receipt; the outer delivery receipt gives the final remote SHA.
