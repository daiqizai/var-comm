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
