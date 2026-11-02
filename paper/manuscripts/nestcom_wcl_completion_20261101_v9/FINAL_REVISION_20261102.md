# 定稿修订与独立验收 — 2026-11-02

## 交付状态

按用户提供的最终修订包完成正文、图注、补充材料与 Word 构建修复。基准为远端 main 的 `eee31804dc73fb1e43c1734ea21554a55c174c3d`；首次交付保留为未提交工作目录修改。用户随后明确要求“推到git”，授权正常提交并推送 main；不投稿、不对外发送。全部匿名稿修订已完成，没有新增实验缺口。

## 已落实的修订

1. 方法明确 K=0 是空边界前缀，e=(10,0) 已收到全部尺度时直接解码，其余情况进行边界预测与覆盖。
2. 最大可行 token 数限定在校准目标以内，不宣称全局最优。
3. Table I 就地定义 A0/B0/C0，并区分 500、100、32 源实验的样本与条件。
4. 引言具体说明 ARPC 的粗尺度传输与后续尺度预测；按问题、设计、证据组织。
5. NULL 正文保留协议行为及 S1 入口，去除开发记录口吻。
6. 摘要 100 词，计时工况明确 N1024、10 dB；保持当前 Token 配置与已完成结果。
7. Fig.2 保留 1×4、92 个均值及 92 组置信区间，图注明确等信息预算下三方法。
8. Fig.3 保留 Truck / Comic book（C10-016 / C10-017）两例、WHOLE 与 SwinJSCC 列及原第一重复；未换噪声、修图或重新评分。Selected examples 的说明保留。
9. 图表统一 Ada-TokenCom，设置中解释 predicted-class、fixed-frame adaptation。
10. S1.4 解释候选表范围，补入 Ada N1024 六个候选和三个冻结胜者的原字节文件及来源。
11. DISTS 使用正式 2022 年卷期页码；修复 ResARC 引号并核对引用首次出现顺序。
12. 同步 LaTeX、图注、输入清单及 Word / 补充材料构建脚本；保留 Word 原生页码字段、表注和表后间距修复。

## 实际构建与查看

| 文件 | 实际页数 | 本轮结果 |
|---|---:|---|
| manuscript_doublecolumn.pdf | 5 | latexmk 实际退出 0；全部 5 页打开检查 |
| manuscript_singlecolumn.pdf | 15 | latexmk 实际退出 0；全部 15 页打开检查 |
| manuscript_review.docx | 15 | 实际重建；Word 原生渲染；全部 15 页打开检查 |
| supplementary.pdf | 27 | latexmk 实际退出 0；全部 27 页打开检查 |

未发现 Overfull、未解析引用、遮挡、裁切、缺字或表格越界。标准 IEEE 10 pt 主文件与版心参数未改动。Word 含原生正文、3 张原生表、6 个展示公式、60 个可编辑 OMML 元素和 3 张图。PDF/Word 共 62 页逐页查看，不能把历史 PASS 当作此次检查。

本轮先实际尝试 documents 技能的 render_docx.py；因本机没有 soffice.exe，记录原错误后使用已安装 Word 16 的隐藏、只读 COM 导出进行全页检查。未使用旧 Word 预览代替新渲染。

## 数据与证据边界

原 inputs 中 56 个 CSV 和 3 个 table CSV、原图与补充图保持原字节。新增 Ada candidate_summary.csv、frozen_policies.json 与 PFS 原始结果逐字节相同；六候选按既有 DINO 选择得到三个冻结胜者。Fig.2 四面板分别 28/28/18/18 点，所有 CI 与六个真实预算坐标保留。全部既有输入清单和图表来源哈希通过核对。

保留 Ada 的感知指标优势以及 SwinJSCC/BPG 的 PSNR 优势，不改成全指标领先、全面 SOTA 或表示优越性的因果结论。8.7% / 20.5% 与 0.660 / 15.422 s 均来自既有冻结结果。无新增图像推理、训练、PHY、指标模型调用或 bootstrap；这里只核验既有 CSV 的算术与身份。未恢复暂停支线。

## 验收文件与复现

本轮独立证据位于 `validation/final_revision_20261102/`：编译退出记录、源文件哈希、四文档哈希、全页预览哈希、Word 转换结构、数据完整性、主张核对、范围与作者信息状态。`validation/current_revision.json` 指向本轮，旧验证和旧发布回执原样保留。

从本目录分别运行 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`、`main_singlecolumn.tex`、`supplementary.tex`；Word 使用 `scripts/build_word.py`。完整数据入口为 `input_manifest.json`、`figure_table_sources.json` 和 `inputs/reproduction/`。原生科学数组及模型仍在 PFS，不进入交付包或 Git。

## 完成清单

- done：25 个覆盖文件按原哈希核验后合并；构建产物独立重建。
- done：12 项文字、图注、书目及 Word 修订。
- done：原图与原 CSV 不变；新增 Ada 附件身份核验。
- done：5/15/15/27 页文档实际编译或导出，62 页实际打开检查。
- done：独立验收、修改清单、复现输入和完整交付包。
- pending-author-input：作者顺序、单位、通信作者与邮箱，以及适用的资助信息。按已确认匿名稿继续保留；不影响本轮匿名稿交付。
- not-run-out-of-scope：新实验、训练、模型推理、重采样统计、历史暂停支线。
- authorized-followup：用户后续明确授权正常 Git 提交与推送；原未发布验收快照保留，当前授权见 publication_authorization.json。
- not-run-by-instruction：投稿或对外发送。

作者块后续填写可能改变分页，届时须重新检查五页限制；本轮没有编造作者或资助信息。
