# 同起点联合收发续训控制

遵守根目录与VAR_COMM规则。当前已于2026-09-14激活Joint训练，接续见docs/CONTINUE.md；原wetok-innovation-r1完整5000、全量评测/分析和reference资格已PASS，GPU无更新梯度检查已通过。原源码/模型/Adam/配置/结果均不改写。正常工程资格核验不是要求未训练候选先在质量上胜出的科学gate。

共同起点为同一204×30 single父点，single/no-history/state-history三个基础结构，初始权重、原损失、数据/增强/SNR/噪声和新增更新机会对应已有RX-only控制。唯一训练因素是通信E是否更新，视觉模型不更新。新臂不声称同y，只保持同源、同物理N/E和同噪声。所有接收历史来自自身，不接受测试真特征。

Encoder有效batch4只调用一次，微批次R/图像梯度须正确累加回共同发送图；不得因为复用RX-only的no_grad而切断图像到E的梯度。旧父点、旧控制和旧强系统只读。新输出留在VAR_COMM/outputs，不租新设备、不改驱动/时钟/MPS、不停他人任务，不提交/push。
