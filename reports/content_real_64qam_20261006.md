# H / C-REAL：补充登记与启动

已按用户补充冻结动作空间、预算、选择规则、成功标准和执行顺序：先 H，再 C-REAL。

**当前已完成 64QAM/LDPC 资格验证和 200 张图的算术码流往返检查；完整候选质量表正在运行。尚无 H/C 系统收益结论。**

| 已核实事项 | 结果 |
|---|---|
| 16/64QAM 的 8 个实际码率布局 | 资格通过，1608 次实际 CPU 译码 |
| raw64 全部合法整数 K | 四码率分别 236 / 316 / 356 / 395 个动作 |
| 四码率最大动作 | m7+K81 / m8+K61 / m8+K101 / m8+K140 |
| 200 张校准图、m6–m9 | 800 次独立算术编码/解码均一致 |
| CPU 正式误块率表 | 已登记等待接续，49152 次译码额度，尚未开始 |
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
