# 固定数字基础＋源残差：近邻原文核对

2026-09-15重新取得下列固定版本原文；不是性能复现，也不声称穷尽截至今日的文献。HTML归档在`outputs/HYBRID-SOURCE-CORRECTION-20260915/literature/`。

| 原工作 | 原文机制及对本实验的约束 |
|---|---|
| HDA-DeepSC，*Hybrid Digital-Analog Semantic Communications*，`2405.12580v2` | 数字基础特征＋模拟特征残差、接收融合及资源分配已有直接先例。“双支路”“传残差”不能作为首创。其训练的数字支路使用容量/可靠数字恢复近似，同时论文也做了实际LDPC检验，不能说全文只是假设理想数字信道。 |
| HiFi-DiffCom，*DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling*，`2406.07390v2` | 用实际信号与既有JSCC重建约束生成恢复。论文自己的10dB表中，NTSCC为29.88dB/LPIPS0.166，HiFi-DiffCom＋NTSCC为29.02dB/0.088。这是其口径的失真—感知取舍，不是本项目3060-use排名。 |
| RDP-JSCC，*Rate-Distortion-Perception Controllable Joint Source-Channel Coding for High-Fidelity Generative Communications*，`2408.14127v1` | 将率失真JSCC与条件生成恢复结合，控制失真—感知工作点。兼顾两类代理指标的研究问题已存在，不能仅以增加生成器声称创新。 |

## 本轮实际研究什么，而不是换名字

新实验具体固定了原VAR生成基础、**RGB源残差的真实独立传输**、每帧完整N/E与CRC失败输出、TX/RX基础可能不一致，以及两个同参数gain对照。可靠度只取实际CRC与接收波形到ML码字的距离；TX不知道RX错误。这些是可核验的实验设计差异，**尚不是已经成立的新颖性结论**。

与此前R2接收器内部的residual不同，新连续支路真正占用1110次信道、传递源token之外的信息。但会削弱数字保护，并增加发送端VAR补全成本。必须以完整raw/arithmetic数字、固定更多token/更多保护、强Deep及同分配融合控制验证收益，不能只胜缩短数字基础。

本轮没有复现HDA-DeepSC或DiffCom原系统，也不能宣布超过它们。若固定分配开发结果支持收益，仍须将具体机制与最近混合传输/可靠度融合工作进一步对齐，再确定论文贡献。

## 可复核来源

- HDA：`https://arxiv.org/html/2405.12580v2`；本地HTML SHA256 `7af4432c4ec4231be039f7dabc6c525e794b41b0f99f4adc869383454b3c3cc5`。
- DiffCom：`https://arxiv.org/html/2406.07390v2`；SHA256 `5fcea6c43f3b328816ae6964cd98e345261aa9428aff3570aaf8ad9f7fb2881c`。
- RDP-JSCC：`https://arxiv.org/html/2408.14127v1`；SHA256 `bcbd0994b26d52b9f6044fafe314579dd673f330d37ad45e4e54f43aecec3bea`。

初次搜索工具没有返回正文、arXiv API返回rate exceeded的限制已遇到；随后通过直接原文HTML完成核对。没有访问认证资源、下载模型或执行论文作者代码。
