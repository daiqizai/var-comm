# 投稿前第一批交付：m10 边界与固定16图补齐

开始与本轮科学闭合时HEAD均为`bdb3e3c12d2ea2c18f4371e11cd912d5e8529100`，与执行包参考一致；没有把旧README中的等待状态当成新任务。范围为T0、T1-A、T2及T3资料准备，交付后停止。旧raw R1 NO_GO、LE0/LE1、LE2和已完成六方法计时均保留。

## 结论

1. **m10可以装入，必须修正候选范围。** 当前cal1000实际终止码长3823/6071/6568.05/6855 bit（min/median/P95/max）。q6,r3/4容量4273可装4源；q6,r5/6容量4751可装18源。其余十档零。8次独立RX全部通过，24,000旧终点码流完全一致。完整尺度与部分尺度两家族都应纳入m10，不能只增强本文方法。
2. **原生raw已补齐固定16表。** 新增48个raw感知/结构评分和64个PSNR，复用144个原分数。raw在DISTS、DreamSim均值上更低，在PSNR、MS-SSIM上更低。全部16来源和三种方向取舍都报告，不宣称总体优于HiFi，不混入common500或13dB熵码结果。
3. **资料准备已齐。** 七种源角色清单/交集、真实LDPC表、分组件参数和训练事实、全部旧指标、同先验四方法表、容量利用率、共享H800原计时、九套正式导出图（含全16图库）均可追溯。新的common500主图标为m10对照待修订，未覆盖旧图或改旧成绩。

## 后续工作量与停止边界

只在两档MCS出现m10新增实际前缀，22个source–MCS组合、去重18源。两家族×三SNR×三噪声，改变子集396逻辑帧；若真正同profile则198物理帧。完整候选校准上界36,000逻辑帧、72,000包，必须先逐帧证明可复用部分再给净新增数；不能误把改变子集当完整补测。最多18个m10干净端点质量缺口，cal1000源码TX/tokenizer无新增缺口。详情及源ID见候选差异清单。

**T1-B、T1-C均未执行。** 无新信道包、无线图像、图像恢复、扩散、训练、bootstrap、计时、LE3/LE4或功率扩展。未恢复任何旧活动调度；暂不新增科学备份包。第一批停止，后续校准/测试须另有范围授权。

## 文件导航

- [m10判断与下一批缺口](m10_candidate_decision.md)；[逐源码长](m10/m10_lengths.csv)；[12档装入表](m10/m10_fit_by_capacity.csv)；[候选差异登记](m10/candidate_delta_manifest.json)。
- [固定16小表及图注](fixed16_table.md)；[64图像对四指标](fixed16/frame_scores.csv)；[均值中位数](fixed16/summary.csv)；[直接逐源差值](fixed16_source_paired_differences.csv)。
- [实现事实表](implementation_fact_sheet.md)；[物理工作点](physical_workpoints.csv)；[源角色](cohort_roles.csv)；[交集](cohort_intersections.csv)；[资产](asset_inventory.csv)；[论断边界](claim_evidence_matrix.csv)。
- [待修订主图](figures_v2/common500_main_provisional.png)；[固定16量化](figures_v2/fixed16_descriptive.png)；[预定前两张](figures_v2/fixed16_first_two.png)；[解释性失真例图说明](illustrative_failure_note.md)；[全部图库1](figures_v2/fixed16_gallery_1.png)、[2](figures_v2/fixed16_gallery_2.png)、[3](figures_v2/fixed16_gallery_3.png)、[4](figures_v2/fixed16_gallery_4.png)；[单码字系统图](figures_v2/single_codeword_system.png)。同名PDF/SVG及600dpiPNG齐。
- [全部指标原值和区间](figures_v2/common500_all_metrics.csv)；[同先验四方法](same_prior_four_methods.csv)；[原共享计时](shared_H800_timing.csv)。RAW_WHOLE没有三项补充指标，明确留空而非新评分。

## 工程记录

历史m10恢复和当前缓存比较均保留；当前1000个旧m9流不匹配，所以按授权独立编码当前m10。科学两任务均真实退出0且父子进程gone，ledger未解决项0。首次绘图因BPG旧/新命名不一致在布局阶段exit1，未计算任何科学分数；第二版仅统一显示别名并保留原方法标签，实际退出0，全部九张PNG已打开审查。中途一次SSH在CPU差异表执行前断开，未产生输出或科学调用；核验后仅生成CPU派生表。原证据不删改。
