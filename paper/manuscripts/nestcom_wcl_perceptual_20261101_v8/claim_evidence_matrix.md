# V8 主张—证据映射

所有输入继承 V7 提交 `231c3e4b5b8a975330a4baf6fd3ad6116c6d7623`；精确哈希见 `figure_table_sources.json`、`validation/claim_audit.json`。CSV 路径均相对本目录。区间为既有直接来源配对、逐点 95% bootstrap，未作多重比较校正；不解释为逐图必然改善。

|主张/位置|证据与样本|范围/限制|文件|
|---|---|---|---|
|共享模型改善像素与感知，III.B|common500，7 个 N/SNR ×5 指标，35 区间有利|主系统各自冻结策略；非纯网格单变量|inputs/e1/paired_deltas.csv、summary.csv|
|10 dB 利用率 77.70→99.79%，DISTS -8.7%、DreamSim -20.5%|N1024/common500，原 3 次信道源内平均|相对 WHOLE；源码容量利用率|inputs/e1/capacity_summary.csv、summary.csv；validation/claim_audit.json|
|匹配粒度 Token−Quarter 有利，III.B|control100，共同 (8,126)、16QAM 2/3、共享噪声|非 common500 主策略直接分解|inputs/e3/paired_deltas.csv、summary.csv、efficiency.csv|
|边界残差及后续条件均有增量，III.B|同一实际前缀，A0/B0/C0，10 dB PSNR/DISTS/DreamSim|A0 保留生成；不泛称各阶段五指标全显著；最后尺度无后续增量|inputs/e4/paired_deltas.csv、summary.csv|
|对 ARPC/Swin/Latent/BPG 两距离一致有利，III.C|common500，4 方法×4 SNR×2 指标，32 区间有利|完整系统；Swin/BPG 四点 PSNR 更高|inputs/e1/paired_deltas.csv、summary.csv|
|Ada 预算相关交叉，III.C/D|NeST 七点 PSNR 有利；距离按点交叉|1024/7、19 DreamSim 跨零；512/19、1024/10 两距离 Ada 有利；保留类别及模型差异|inputs/e1/paired_deltas.csv、summary.csv|
|E2 紧信息预算区间，III.C|common500，6 总信息预算；含 57bit、Ada 额外10bit|无 PHY/MCS；不是无线鲁棒性或顺序单变量因果|inputs/e2/summary.csv、paired_deltas.csv|
|90bit 中 WHOLE 用3.1、NeST用89.9，III.C|B1212→1302，平均 source-bit 差值|源均值，不保证每图；WHOLE DS 约.2047，NeST .1837→.1743|inputs/e2/representation_and_capacity.csv、summary.csv|
|NeST .660s，Ada15.422s，比值23.4，III.C/TableII|timing32，同 RTX PRO6000、2核、batch1|实际完整 TX+RX；非空口时延。NeST 相对 WHOLE +13.1%，JSCC 更快|inputs/e5/timing_summary.csv、execution_closure.json|
|Token比Quarter TX+16.0%；125.96与352不同|timing32 与 control100 分别标注|352实际长度检查，1概率遍历，9CDF步骤；不是352模型前向|inputs/e5/source_fitting_accounting.csv；inputs/e3/efficiency.csv|
|N512内部五指标一致有利，III.D|common500，3 SNR×5 指标|N512/N1024 独立校准；Ada19dB两距离更好|inputs/e1/summary.csv、paired_deltas.csv|
|固定策略的门限，III.E|control100×20noise×5SNR×3methods=30000|8dB全拒收，9dB15.75/14.95/13.50%，9.5dB.20/.10/.10%，10dB无观察拒收；灰图纳入总体|inputs/e6/summary.csv、execution_closure.json|
|Fig.3 源结构保留|source84 TV、143 patterned leather；N1024/7；各法原repeat0|Selected examples；不同历史noise；全图DISTS/DS对应显示那次输出|inputs/f1/selected_native_mapping.json、fig3_metrics.csv、selected_examples.json|
|训练、策略、评价目标明确|只读核对历史代码及冻结配置|MSE+.1LPIPS/DINO/五评价器不同；不声称直接优化DISTS/DS或人类偏好|validation/training_calibration_identity.json；inputs/e1/frozen_policy_metadata.json|

不采用新 paid-NULL 主配置，不改结果/统计 bank，不增加新科学计算。完整不利/混合结果仍在补充和 CSV。
