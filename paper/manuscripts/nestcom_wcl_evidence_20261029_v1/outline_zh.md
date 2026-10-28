# 最终稿结构
标题：NeST-Com: Fixed-Budget Next-Scale Token Communication。匿名。

中心论点：以完整粗尺度和逐尺度残差为源图描述，通过尺度内端点、独立终止实际码长装入和前缀约束接收，在固定数字帧中保留更多源图信息；边界实收 token 同时更新残差与后续尺度条件。

摘要：同模型双预算增益与源码容量利用率，保留完整系统的指标取舍。
I：固定帧问题；VAR 原始生成 benchmark 与 next-scale 结构的选型理由；整尺度容量失配；本方法及不同证据的边界。
II.A：表示、AWGN/能量/CSI、付费头部与 body、冻结适配 decoder。
II.B：端点、独立终止实际码长、离线两阶段校准、在线固定 MCS fallback、两预算不同 NULL/错误规则。
II.C：预测、覆盖、残差与条件更新；无同尺度再预测。
III.A：六系统、样本/训练/校准、原噪声、四指标、source-paired统计。
III.B：Table I 的主策略/粒度/接收干预，然后 E2 的混合结论。
III.C：Fig2/TableII六系统取舍、E1 N102410结果、Kodak。
III.D：N512描述性W−Ada、配对NeST−WHOLE、TableIII外部取舍、Fig3固定身份示例、E1低预算、失败数及独立校准限制。
III.E：短述N2048、8dB全拒收的失配观察、软件成本；Ada未计时。
IV：固定帧贡献、同模型证据、低预算及感知取舍、SNR不确定性。
