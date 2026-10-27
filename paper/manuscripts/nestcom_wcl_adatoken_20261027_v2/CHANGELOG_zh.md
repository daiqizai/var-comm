# 本轮修改说明

修订父版本：1f2a86a41b65fe90c0bf245076212b0a27852deb。旧目录 v1 完整保留，新目录 v2。主实验及三张图的原始像素、曲线和全部均值/区间保持不变。

| 位置 | 改动 | 证据 |
|---|---|---|
| II.A | TX匹配SNR、RX真实方差、数字与连续功率口径、68符号旋转BPSK与956正文 | p0/evidence_audit.json、原 ep_phy/budget_model/swin_model |
| II.B | 144候选、100源初筛、1000源复评、m10修正、DINO含灰图、并列规则 | p0/complete_candidate_records.json；原 T1-B frozen_policies.json |
| II.B | 4/10/19实际目标/MCS；独立终止码长；降token数回退到(4,0)；无可行端点为工程错误 | p0/frozen_strategies.json；ep_plan.py、tb_plan.py |
| 引言、II.C | 本地粗尺度CDF，尺度内端点与实际装帧；边界未收到位置不重新以K条件预测；覆盖同时更新残差/context | 原 VAR接收实现与已完成B诊断 |
| III.A | 按最终源码解析/输出审核接收失败；TX回退单列 | p0/final_reception_counts.csv；Kodak ARPC原生72输出SHA复核 |
| III.B | 主策略、受控网格、A0/B0/C0三种证据用途及干预定义 | 原 a_delivery/b_delivery；Table I既有输入 |
| III.C/D | Ada数据集、19dB全256token；ARPC温度0.5/0.25/argmax；Kodak跨零CI准确表述 | 原 C正式交付和ARPC敏感性配对表 |
| III.D | 新增冻结10dB策略8/10/12dB诊断，完整保留8dB全部灰图失败；10/12相同接收输出 | p1_delivery的24均值、12配对CI、1800逐次记录 |
| 结论 | SNR失配已完成，工作边界据实报告；P2不作为已完成结果 | p1_delivery/REPORT.md |
| Fig.1图注 | 77.70/95.21明确为common500逐图利用率的均值 | Table I及原family_snr_summary |
| Word/LaTeX | 同一文字源同步，匿名、10pt双栏五页及12pt双行距单栏 | 编译收据及visual_review.json |

上述 p0/p1_delivery 均位于 results/wcl_revision_20261027_v1。新版仅补P1：1200新物理帧、2400包译码，600旧10dB帧逐项等价复用；零新源模型/VAR恢复，461个缺失指标标量。原link末尾CUDA状态断言导致exit1，已保留并以独立只读元数据核验闭合，不伪称原exit0。P2未启动。
