# NeST-Com 贡献论证与相关工作修订（2026-11-03）

本轮基于 `62f3e21edddd320efac25dc348398a1402bf1959`，完成当前 V9 的论证修订和文档重建。核心表述为：保留尺度级概率计算，把传输端点细化到尺度内，并按独立终止后的实际码长满足固定数字帧容量。

## 前人基础与各自贡献

ARPC 的官方摘要明确从 VAR 的多尺度残差量化和预测概率出发，将尺度选择、熵编码及未收尺度生成用于渐进压缩，并提出 group-masked bitwise multi-scale quantizer。它的价值由面向压缩的具体设计体现，而不是声称所有组件均从零发明。[ARPC 官方摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a255ea760cfedf42a988d9de6ec902cf-Abstract-Conference.html)

Ada-TokenCom 使用自回归概率、算术编码、条件生成及 Lyapunov 优化，围绕前缀长度、MCS 和长期符号预算组织通信方法。这些已有基础不抹去其新的联合设计问题。[Ada 原文](https://arxiv.org/html/2608.28086v1)

NeST 的论证据此采用同一标准：说明表示、编码粒度和帧容量之间的新关系，再给出实现条件、性质和受控证据。正文从固定帧问题出发展开方法；VAR 的接收操作在方法中准确归属，不再反复用否定性句子收缩贡献。此处是技术定位判断，并非优先权或录用保证。

## 实际修改

摘要、引言和结论统一突出“尺度级概率计算与尺度内传输端点分离”。II-B 加入当前编码器的 `b_t=E_t+P_t+2`、非递减性质及固定遍历/非修改式终止条件；当前仍实际全扫描，没有增添未验证的提速主张。

结果段明确 Eighth 在当前匹配点提供较低计算成本，Token 以额外 30.7 ms（14.1%）TX 获得更充分容量利用和小幅质量增益。正文直接报告 NeST/WHOLE TX/RX、NeST 与 Swin 总成本，保留 Ada 的优势、失败和混合结果。H10/U10 的参数一致性与目录差异、Kodak 历史 Ada19 配置差异均明确。补充 S6 选例描述客观化，来源仍为135/201，没有换图。

S7 压缩重复的过程说明；原段落保存在 `reproduction_logs/narrative_revision_detail_20261103.md`，旧复现日志和全部数据保持。三图、四主表和科学输入逐文件哈希与基准一致，补充中的数值表块全部相同。

## 文档验收

双栏/单栏/Word/补充为 **5/15/16/36 页**，摘要 **88 词**。最终双栏、单栏、Word 页面已实际打开检查；补充36页在本轮首轮渲染时逐页打开，最终渲染逐页字节一致。共72页，无观察到的裁切、重叠、缺字或表格越界。三份 LaTeX 编译实际退出0，无溢出或未定义引用警告。Word 保留4张原生表、69个可编辑 OMML 数学元素和6个编号公式。

新增模型、PHY、指标、统计、计时和训练调用均为 **0**；暂停支线和调度未更改。正常 main 提交/推送及远端 SHA 由本次 Git 历史和外层发布回执记录；仓库必要检查单独保存在当前验收目录，旧 PASS 不用于本轮。

交付：`manuscript_doublecolumn.pdf`、`manuscript_singlecolumn.pdf`、`manuscript_review.docx`、`supplementary.pdf`；定位与验收见 `validation/current_revision.json`。
