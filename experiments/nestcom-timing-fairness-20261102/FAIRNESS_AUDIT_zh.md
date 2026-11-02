# 计时公平性核对

**用户进一步指出的问题是Ada额外实现开销，而不是单纯的CPU分配。** 随后核对[原文IV-E和Table IV](https://arxiv.org/html/2608.28086v1#S4.SS5)：发送端一次prefill，并使用torch.compile；编码19ms、解码1012ms（RTX5090/Xeon8260），统计tokenizer、AR概率计算、softmax和AC。原文还说明RX为数值一致性重复整前缀计算。因此，本地逐token TX、未用的无条件TX分支和部分审计不能冒充Ada方法必需开销。原文数字与本地完整数字收发边界不同，亦不能直接代填。

本轮硬件复测不足以恢复Ada计时作为公平外部比较的资格。正文Ada计时单元留空，所有实际测量仍作为适配实现审计保留；删除NeST/Ada固定加速倍数论断。独立的两源prefill资格检查在原总资源预算内判断是否可直接替换旧CDF/码流路线，不将未经验证的编译快路径写成完成。

本轮已在原32源、冻结10 dB/N1024策略和实际收发边界下完成资源复测。旧主实验与图像质量结果保持原样。两个固定源的4次prefill检查也已实际结束：全部1024组整数CDF与原增量路线不同，独立终止流分别仅4/256、4/256、2/256、2/256精确一致。这不能直接替换原TX并继续配旧RX，也不说明prefill不可行；作者路线一致的Ada端到端计时仍未完成。完整结果见[本轮报告](../../results/nestcom_timing_fairness_20261102_v1/delivery/REPORT_zh.md)。

| 检查项 | 核对结果 | 本轮处理 |
|---|---|---|
| CPU硬件 | 原38/39是socket0/core19的两个SMT线程，并非两个物理核；原线程池6/2。所有方法原来使用相同配置 | 改38/40，对应两个不同物理核，实测区域线程池2/2；不把配置错误等同于只给某个方法更少资源 |
| GPU | 原阶段均有设备身份；本次初始GPU0被外部进程占用 | 原始退出码与账本保留；一次登记切换GPU6，所有最终GPU结果在同一GPU6，不混入GPU0的已完成计时 |
| 收发边界 | 原计时已包含实际TX/RX，并有CUDA边界同步、两次预热、三次测量 | 保留同边界与同输入；不是相加离线小组件估算 |
| 源数据与接收输入 | 完整32源身份、预处理、冻结模型、波形和接收图像可核对 | 每个输出在计时外与原事件hash比较；RX只读收到的符号，不读源tokens或未发信息 |
| 软件环境 | 六个主方法依赖的主模型环境为Torch2.11+CUDA12.8；ARPC有单独历史环境 | ARPC保持补充历史结果，不把它混入本轮六方法计时 |
| Ada实现效率 | TX串行评估256个位置，并计算条件/无条件两支；只有条件支进入熵CDF。强制FP32 math attention | 本轮冻结实现不改数学路线；不能把该耗时解释为Ada算法达到优化极限后的耗时 |
| Ada RX上下文 | 先解出前缀，再延续同一上下文生成尾部 | 未发现“同一前缀被完整重放两次”的依据，不据此解释慢速 |
| 审计开销 | Ada每token有限值检查会同步，且有未用的CDF审计hash；NeST有未用完成token hash | 仅固定前2源补充诊断，挪出/移除这几项审计；保留CRC、长度、parser及算术规范性检查，不替换32源主结果 |
| 统计 | 三次计时重复不是三个独立源；同一source应等权 | 先源内平均，再用固定10000×32来源bank；不把来源区间当整台共享主机的负载不确定性 |
| 粒度成本 | 主Table I(B)的3个TX单元也沿用旧资源配置 | 在同预算内补Whole/Quarter/Eighth TX，Token精确复用主NeST TX；不重算图像指标 |

两类结论需要分开：资源与计时边界可以统一，模型结构、数值核和实现优化程度仍不同。论文报告冻结实现的处理成本，并解释尺度级预测与串行token概率计算的工作差异。当前复测不支持“算法上普遍快某个固定倍数”。

判据与执行：代码及注册文件不可热改，全部完成须有实际wait0、PID/start_ticks消失和完整账本；未知失败保留、不自动重试。最终结果使用`finalize_clean.py`，旧计时与干扰记录均保留。只有审计后闭合的数据进入当前论文。

代码定位：

- [原计时边界与TX实现](../nestcom-perceptual-followup-20261030/timing_end_to_end_identity_v5.py)：`main` 中的两端同步及计时，`tx` 中的逐token概率与独立终止循环。
- [Ada冻结接收模型](../adatoken-response-20261026/c_predclass_native.py)：`PredictedClassNative.logits` 的两分支和math attention；继承的 `Native.cdf` 仅使用条件分支；`tail` 才合成CFG。
- [本轮资源与审计诊断](fair_worker.py)：`setup` 在加载身份检查后设置有效线程数；`install_audit_diagnostic` 只移动或去除列出的审计操作；`after_trial` 在计时外校验原输出hash。
- [最终统计闭合检查](finalize_clean.py)：逐阶段实际退出与账本、CPU拓扑、设备身份、全部输出hash、累计预算和来源统计。

方法核对参考：[PyTorch 2.11 benchmark utilities](https://docs.pytorch.org/docs/2.11/benchmark_utils.html)与[benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)说明预热、设备同步与线程配置的重要性；本轮执行继续使用已冻结的软件版本。
