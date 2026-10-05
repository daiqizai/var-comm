# H / C-REAL：补充登记与启动

已按用户补充冻结动作空间、预算、选择规则、成功标准和执行顺序：先 H，再 C-REAL。

**当前已完成资格、质量表、误块率粗测、预筛及9600帧实际CPU接收；首次GPU启动失败已保留现场，独立恢复状态见下文。尚无H/C系统收益结论。**

| 已核实事项 | 结果 |
|---|---|
| 16/64QAM 的 8 个实际码率布局 | 资格通过，1608 次实际 CPU 译码 |
| raw64 全部合法整数 K | 四码率分别 236 / 316 / 356 / 395 个动作 |
| 四码率最大动作 | m7+K81 / m8+K61 / m8+K101 / m8+K140 |
| 200 张校准图、m6–m9 | 800 次独立算术编码/解码均一致 |
| CPU 正式误块率表 | 49152次正文译码已完成；未触发登记精化门槛 |
| 主系统基线 | S1 已选光栅+KEEP；完整1000源校准仍待完成 |
| C-REAL | 完整期望质量选择规则已冻结，H 完成后执行 |

## 登记与证据

- [完整补充说明](../results/content_real_64qam_20261006/REGISTRATION_ADDENDUM.md)
- [已核实阶段与未交付范围](../results/content_real_64qam_20261006/STATUS.md)
- [CPU 自动接续边界](../results/content_real_64qam_20261006/COARSE_SCHEDULE.md)
- [H 科学协议](../results/content_real_64qam_20261006/H_CODEC_PROTOCOL.json)、[20万次分阶段预算](../results/content_real_64qam_20261006/resource_budget.json)
- [C 完整选择规则](../results/content_real_64qam_20261006/C/C_REAL_PROTOCOL.json)、[成功与停止标准](../results/content_real_64qam_20261006/SUCCESS_CRITERIA.json)
- [独立评测骨干绑定](../results/content_real_64qam_20261006/INDEPENDENT_METRIC_BINDING.json)
- [实际后端动作目录](../results/content_real_64qam_20261006/H/qualification/catalogue.json)
- [源码往返与码长表](../results/content_real_64qam_20261006/H/source200/source_lengths_per_image.csv)
- [原始与公开副本 SHA 清单](../results/content_real_64qam_20261006/PUBLIC_EXPORT_MANIFEST.json)

资格曲线每布局/SNR仅64个码块，正式可靠性仍由后续2048次粗测与登记精化确定。无噪声往返正确不能证明有噪声图像质量改善。所有最终结论仍需失败计入的真实链路评测、独立指标、完整主系统参考和发送端成本。

旧 A1/A2、旧 C 停止结论和既有模型保持原版本。本批没有训练更新，未启动最终 holdout。

## 预筛执行补充

已在正式误块率数据产生前冻结[预筛解释与接续规则](../results/content_real_64qam_20261006/prescreen_registration_v1/PRESCREEN_SCHEDULE.md)，并启动独立CPU等待器。它在质量表和正式误块率批次全部收尾后运行完整候选排名；若需要精化，只登记请求，不自动调用额外译码。

[新增工程测试与执行登记](../results/content_real_64qam_20261006/prescreen_registration_v1/PUBLIC_EXPORT_MANIFEST.json)已保存。尚无新的系统质量结论。

## 后续接收与选策代码准备

[准备范围和模拟测试](../results/content_real_64qam_20261006/prepared_execution_v1/PREPARED_CODE.md)已归档。新增有界精化、实际载荷接收、独立算术源解码与重建，以及200源逐帧PSNR选策代码；接收阶段不读取发送端真值。

这些代码尚未登记或启动科学运行。完整1000源校准、全部指标、正式评测、发送端计时和C-REAL仍待完成，当前没有新的系统质量结论。

## 初始真实载荷阶段

[已完成测量、首次失败与恢复状态](../results/content_real_64qam_20261006/initial_payload_execution_v1/STAGE_PROGRESS.md)已归档。实际接收完成9600帧、19200次计费译码；GPU恢复进度以该页的封存快照为准。原始失败证据保留，完整H主表与C-REAL尚未完成。

<!-- H_INITIAL200_SELECTION_R2 -->

## 初始200源选策与独立收尾认证

已完成200张固定校准源图 × 16个整尺度候选 × 3个噪声，共9600帧实际接收重建，并按登记的逐源平均PSNR规则选出8个整尺度策略。下表使用同一批200源选择并报告，不能据此作配对优势或最终泛化结论；它不是development结果，也不表示完整1000源校准或H系统优势。

| SNR | 臂 | 目标尺度 | 实际源尺度（图数） | 调制 | 码率 | 初始200源平均PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| 13 dB | H16-R | m8 | m8: 200 | 16QAM | 5/6 | 20.1562 dB |
| 13 dB | H16-A | m9 | m8: 197，m9: 3 | 16QAM | 5/6 | 20.1871 dB |
| 13 dB | H64-R | m7 | m7: 200 | 64QAM | 1/2 | 18.4604 dB |
| 13 dB | H64-A | m9 | m8: 199，m9: 1 | 64QAM | 1/2 | 20.1609 dB |
| 19 dB | H16-R | m8 | m8: 200 | 16QAM | 5/6 | 20.1562 dB |
| 19 dB | H16-A | m9 | m8: 197，m9: 3 | 16QAM | 5/6 | 20.1871 dB |
| 19 dB | H64-R | m8 | m8: 200 | 64QAM | 2/3 | 20.1562 dB |
| 19 dB | H64-A | m9 | m9: 200 | 64QAM | 5/6 | 21.7294 dB |

实际源尺度是发送端按登记容量回退后的前缀，按200张源图计数；目标m9不表示每张图都能发送到m9，需结合m8/m9的实际分布解释算术编码的收益。该列不是接收CRC成功率，PSNR仍由实际接收重建计算。

GPU worker已成功退出并封存全部图像和分数；原render owner在观察worker退出时遇到竞态而失败，原owner仍没有成功完成凭证。后续独立只读认证核验了原失败、worker退出0、全部输出SHA和69960次既有计费，并精确归档释放两个已登记STOP。原失败与22件归档保留，遗漏的局部STOP另有补充证据；没有补写旧owner成功记录，也没有重跑PHY或图像。

本次CPU选策owner及其worker已成功完成并退出。6个partial候选仅保留原预筛引用，尚未在该阶段完成partial校准。完整1000源校准、统一指标、development、主系统校准及C-REAL仍待各自登记执行；本次没有自动启动这些阶段。

[完整选策与收尾证据](../results/content_real_64qam_20261006/initial200_selection_r2/README.md)包含原始SHA与公开脱敏副本SHA。早先提交445a07e保留为当时阶段快照。
