# latent增强后续：N3572/N4084数字模式校准协议

这是独立于已完成41方法矩阵的新对照，不重跑或改写旧结果。

- 原1000张calibration、三个噪声、五个SNR；候选raw/arithmetic×N3572/N4084×m7/m8/m9×D0/Dc；
- 使用原VAR概率算术码、header/CRC/tail/FEC和实际长度；所有码流实际计费。当前协议header的合法mode域仍是m7/m8/m9，m10不加入，不截断、不丢弃超长图；
- 每个family、budget、renderer分别按既有`reliability/quality/goodput`规则拟合动作，候选成功定义为header、body CRC、label、mode和发送prefix均正确；质量规则使用校准LPIPS并受可靠性候选PSNR guard；DINO只报告；
- 校准动作冻结后，只从已经完成的development固定候选矩阵读取相应SNR/mode/render结果，不在development挑选模式；
- mode动作只依赖名义共享SNR，不增加免费逐图信令。所有失败保留；不访问holdout、不训练、不改原checkpoint。

新产物写入`outputs/VAR-LATENT-ENHANCEMENT-20260917/followup/digital_policy_v1/`，完成后另写冻结动作、development源图级区间和N3572/N4084对照表。
