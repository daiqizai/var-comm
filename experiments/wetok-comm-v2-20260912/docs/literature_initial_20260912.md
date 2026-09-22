# 当前原始资料核查（非新颖性完成证明）

2026-09-12查询作者/原论文入口：

- WeTok：`https://arxiv.org/abs/2508.05599`，网页最新v3（2026-02-09）。本轮不是自动换到最新配置，而是继续已实测的指定ImageNet/stride16 EMA与固定作者commit；native LFQ及确定性Decoder接口以本地SHA锁定代码为准。
- WeTok作者模型卡：`https://huggingface.co/GrayShine/WeTok`；作者源码`https://github.com/zhuangshaobin/WeTok`。本地源码LICENSE为Apache-2.0；不从模型卡无独立license metadata推断权重再分发/商用权利。
- HJSCC：`https://arxiv.org/abs/2408.16340`（当前v5，2025-02-27），已研究层级潜变量及条件通信映射。A1的粗到细结构本身不能作为首次贡献，需同骨干、同观测控制和资源收益。
- Ada-TokenCom：`https://arxiv.org/abs/2608.28086`（2026-08-28），已有部分token传输与MCS/速率适配方向。当前A0/A1不做此类自适应，也不能重命名旧VAR前缀/MCS工作为新颖性。

本轮已阅读实际WeTok Encoder/Decoder/LFQ、官方推理接口与配置，代码没有套用RGB SwinJSCC权重或旧VAR 4096分类头。
这些是启动范围核查，不是“无人做过”的证明；有稳定通信机制证据后应继续阅读全文和最新作者实现，补HJSCC等合理强对照。
