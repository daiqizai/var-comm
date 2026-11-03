# 当前版本：14 页补充阅读版（2026-11-03）

补充材料由 35 页缩至 14 页，减少 60%。核心算法/证明、匹配控制、失败与不利结果保留；重复全量表和运行过程移入完整复现文件。正文双栏/单栏/Word、原图及全部科学输入保持原字节。14 页阅读版已编译并逐页查看，无新科学计算。

入口：[修改说明](SUPPLEMENT_READER_20261103.md)、[复现清单](REPRODUCTION_CHECKLIST.md)、[逐节证据索引](reproduction/reader_edition_index.md)、[本轮验收](validation/current_revision.json)。本地/PFS ZIP 提供完整配套文件；Git 保存独立文件和构建脚本。

--- 以下为此前版本记录 ---

# 当前交付：Fig.1 信令与补充材料整理（2026-11-03）

Profile ID/预配置 MCS、NULL 和分别测量 TX/RX 的表述已统一。保留四段引言和 Table I(A/B/C)，科学数据、置信区间、Fig.2/3 原文件不变。S1/S7 阅读顺序调整，完整复现记录集中至可下载配套包。

[双栏正文](manuscript_doublecolumn.pdf) · [单栏正文](manuscript_singlecolumn.pdf) · [补充 PDF](supplementary.pdf) · [Word](manuscript_review.docx) · [复现清单](REPRODUCTION_CHECKLIST.md) · [复现包生成入口](scripts/build_reproduction_companion.py) · [修改记录](SIGNAL_SUPPLEMENT_REVISION_20261103.md) · [当前验收](validation/current_revision.json)。页数 5/15/35/15，摘要 89 词；70 页本轮实际查看，未新增实验。

--- Historical record / 历史记录 ---

# 2026-11-03 阅读连贯性与补充入口

最终引言采用用户确认的四段论证；四个关键术语在引言、方法和补充对应说明中统一。英文与中文对照及术语定义见 `READING_FLOW_20261103.md`。

- 方法/实验设置末集中指引补充，保留必要结论和配置差异。
- 引言末段突出容量不匹配、实际码长拟合、接收信息利用的因果链；表 I 完整保留。
- 时延去掉内部目录代号；编译 Ada 仅配自身 32 图质量。
- 压缩粒度、Ada 敏感性与 Kodak 逐点数字，补充增导航；科学表和图像不变。
- 同步三份 PDF 和 Word，5/15/15/36 页，71 页实际审阅。

--- Historical record / 历史记录 ---

# 2026-11-03 固定帧问题与相关工作论证

- 按表示、先行研究、固定帧协同约束和系统设计四段重写引言。
- 明确Ada已有联合前缀/MCS适配；保留ARPC渐进源编码与VAR接收机制归属，以及ResARC生成残差补偿。
- 同步Word及两版PDF，5/15/16页；上一轮跨页修正、全部科学数值与图表保持。

--- Historical record / 历史记录 ---

# 2026-11-03 跨页与措辞精修

- 删除跨页总结与一处重复总结，收紧等预算引导句，使Ada敏感性数据完整落在第4页。
- 修正平均主体、差值区间跨零、端点上限和源码长度的指代；表IV直接说明MCS单位。
- 同步LaTeX图注和Word，双栏/单栏/Word为5/15/16页，摘要88词；科学数值和图像资产未改。

--- Historical record / 历史记录 ---

# 2026-11-03 整体框架定位

- 摘要以固定预算数字通信框架开篇，说明源前缀、信道保护与接收重建的联系。
- 引言按整体框架、技术机制、证据安排贡献，保留相关工作与接收机制归属。
- 结论总结完整系统及其支撑机制，保持原有结果范围。
- 双栏5页、单栏15页、Word16页，摘要86词；方法、结果、补充及全部图表数据未改。

--- Historical record / 历史记录 ---

# 2026-11-03 贡献论证

- 统一摘要、引言、方法和结论，正面解释尺度级概率与尺度内独立终止的配合。
- 正文增加有条件的终止码长非递减性质，明确当前实测仍全扫描。
- 说明 Eighth/Token 的质量成本取舍、当前与历史策略对应及已有 ARPC 适配结果。
- 精简补充过程文字，保留复现记录和全部数值；Fig.3 选择说明改为客观规则。
- 重建双栏5页、单栏15页、Word16页、补充36页；摘要88词，72页实际验收。

--- Historical record / 历史记录 ---

## 2026-11-03 — Wording and policy-context clarification

- Unify the preconfigured MCS/directory and endpoint-profile semantics in the body, Fig.1 caption and S1.3.
- Index current and historical MCS/ceilings/directory identities; correct receiver-intervention lineage to Token prefixes at 7/10 dB and the final-scale structural check at 19 dB.
- Restore NeST/WHOLE complete TX/RX costs to the body and shorten Ada diagnostics; archive execution history without deleting evidence.
- Rebuild and review 5/15/16/36 pages; no new experiments or changed scientific data/figure assets.

--- Historical record / 历史记录 ---

# 2026-11-03 — unified calibration and bounded evidence

Replaced the three N1024 token-system policies/results after common12-MCS two-stage calibration; updated24Fig.2 rows,6Fig.3 display images and TableI(A)/II. Added TableIV, explicit fitting pseudocode/rounding/scan costs, N512matched-grid five-metric evidence, compiled32quality paired to its own timing, and encoder termination/monotonicity audit. Preserved Fig.1 bytes, selected source identities, historical N512/equal-bit/Kodak/mechanism/threshold/timing evidence and all failed variants. Final artifacts5/15/16/35pages;71pages actually inspected. No training, optional search optimization or new timing.

--- Historical record / 历史记录 ---

## 2026-11-03：合理Ada实现完成

单次条件prefill TX及匹配因果prefill RX经独立算术恢复验证并完成32源正式计时。成功资格复用，环境失败保留；正文/S7同步新时间，旧质量/图源不变，重新编译与实际页面验收。

## 2026-11-02：计时公平性复核

撤下未对齐作者实现的Ada主表计时和NeST/Ada速度比；保留审计数据及两源prefill不等价结果。修正CPU拓扑、统一GPU与2/2线程，更新其余Table I(B)/II计时。图像质量与图源不变，全部交付重新编译并逐页检查。

## 2026-11-02 — Positioning and evidence roles

Reframe the argument around scale-wise prediction with within-scale, actual-length frame fitting. Credit ARPC as the closest progressive source-coding precedent and VAR for inherited overwrite/recursive state updates. Separate matched granularity, family-policy and mechanism evidence. Move the ARPC digital-link adaptation from the main figure/table into the complete supplementary comparison without altering any input data. Rebuild and visually review 5/15/15/27 pages; retain the user-final Fig.1 and Fig.3 exactly. See POSITIONING_REVISION_20261102.md.

---

## 当前 Fig.3 修订 — 2026-11-02

已采用 C10-016 / C10-017 的 N1024、10 dB 两行五列对照，加入 SwinJSCC。图下 DISTS / DreamSim 均为对应原第一重复的实际指标。双栏/单栏/Word/补充材料页数为 5/15/15/27，62 页均已逐页检查。数据、取舍与来源见 [Fig.3 修订记录](FIG3_SELECTION_20261102.md)。下面保留历史版本记录；当前验证文件以本轮为准。

## 2026-11-02 — Fig. 2 at final print size

Reflow the frozen 92 records into a 1×4 vector figure; synchronize caption, argumentation, Word image mapping and exports. Validate 5/15/14 pages, embedded fonts, all 92 CI segments, grayscale and physical dimensions. Figures 1/3, tables and all scientific results are unchanged.

# V8 → V9

补齐当前目录 paid-NULL 与 33 项实际协议回归；导出可复现 CDF/算术编码/PHY 配置、1241 行候选、24 个 LDPC 布局及真实 TX/RX 参考。
完成 Token Kodak216 个逻辑事件，复用合法旧输出，补全四方法五指标与24源配对；正负结果同时进入正文和 S8。
恢复 Ada 自身 sampling/argmax 控制，核验165行逐源分布，并增加预先固定的中位秩/退化案例。
Fig.2 为四面板并加入 WHOLE 与六个等信息预算；Fig.3 保留电视一行，完整比较移补充；Table I(B) 加匹配 TX 时间。
引言、结果和结论围绕源信息装填及实际工作区间组织；加入 ResARC 区别；保持 Ada 高预算优势及 JSCC 更低延时。
所有格式实际重建，未复制旧 PASS。补充最终27页，单栏15页，Word14页；61页实际审查。
