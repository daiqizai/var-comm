# 主张—证据映射

| 主张/正文位置 | 数据与人口 | 文件 | 边界/保留的混合结果 |
|---|---|---|---|
| next-scale粗到细及同尺度CDF因子化，II.A | VAR原方法 | references.tex/VAR；II.A公式(2) | 原生成SOTA归属VAR原任务；非本文通信排名 |
| N1024同模型PSNR +.616/.701/.319，util77.70→95.21 | common500，各3重复 | tables/table_i.csv A；inputs/internal_common500_paired.csv | 同模型策略族；非完整系统因果分解 |
| Kodak迁移 | 24源 | table_i A；inputs/internal_kodak_paired.csv | 19dB DreamSim区间跨零 |
| 粒度提升利用率与PSNR | 固定100源/模型/MCS/最大端点 | table_i B；inputs/a_summary.csv、a_paired.csv | Eighth−Quarter PSNR+.123；DreamSim/CLIP-I区间跨零 |
| 边界token两条作用 | 固定100源同实收前缀 | table_i C；inputs/b_paired.csv | +.220、+.440为有序干预；最后尺度传播为零 |
| Completion对coarse解码 | 100源，N512，4/10dB，各3重复 | inputs/e2_summary.csv、e2_paired_deltas.csv；supplement S4 | DreamSim−.420/−.452；PSNR4dB区间跨零、10dB+.771 |
| 六系统实际取舍 | common500，N1024 | figures/figure2_data.csv；tables/table_ii.csv | Swin/BPG较高PSNR；Ada10dB三感知指标较好 |
| N512同模型增益与容量 | common500，各3重复 | inputs/paired_deltas.csv、capacity_summary.csv、frozen_policies.json | +.392/.399/.792；10dB同(7,0)/16QAM2/3；util79.30→93.46 |
| N512 NeST−Ada | common500 | tables/table_iii.csv；inputs/summary.csv | Ada三个SNR CLIP-I及19dB DreamSim更好 |
| greedy敏感性 | 固定100源诊断 | inputs/e1_summary.csv、e1_paired_deltas.csv；supplement S3 | N512对argmax PSNR4.63/4.60；原sampling仍主基线 |
| 灯罩轮廓例子 | 单源N51210dB seed6201 | figure3_metrics.csv；figures/figure3_roi.json | 编辑性例子；全图指标，不宣称恢复文字/桌纸 |
| 已有附加结果与时间 | 各原实验人口/协议 | supplement S5；configuration index | N2048 DISTS不利、失配全拒收、无Ada速度比较 |
| 无可行端点行为 | 冻结源码＋7项CPU合成 | validation/implementation_status.json | 不等同新增无线实验；两预算行为保持差异 |
