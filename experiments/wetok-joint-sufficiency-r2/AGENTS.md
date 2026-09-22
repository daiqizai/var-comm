# 原配方充分性续训

遵守VAR_COMM根规则。本目录是原Joint/Grid5000完整结果之后的四臂匹配训练充分性修订；先读`docs/CONTINUE.md`和`docs/protocol.md`。旧Joint/Grid/R-only/geometry的代码、参数、optimizer快照与结果只读，不修改其哈希或重开旧队列。

必须从四个实际5000末端模型和Adam恢复，不从较早selected点重启，不从原7000父点重新开始。数据global12000→17000，同源顺序/增强/SNR/标准噪声，原loss/1e−4学习率、连续接口、N/E、视觉冻结不变。普通迭代不称为next-scale，额外训练不当新机制。

2026-09-14已完成真实四模型/Adam恢复与CPU driver断点测试，并实际启动至7500的续训，见`docs/CONTINUE.md`。不要重启重复实例或修改已绑定训练/恢复源码。后续启动仍核实授权GPU0与现有任务，不停他人进程、不改驱动/MPS/时钟、不租新服务、不提交/push。
