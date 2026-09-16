# VAR 语义保留与通信资源贡献消融结果

日期：2026-09-04

分析 ID：`ANALYSIS-VAR-SEMANTIC-RATE-CONTRIBUTION-IMAGENET100-001`

## 结论

当前系统中，m8/m9 的高语义保留并不是较短 VQ 前缀单独实现的；冻结 VAR 在接收端补全未发送尺度，贡献非常显著。

- m8 只发送 full-VQ 的 `37.5%` raw index bits。官方 Decoder 下，前缀本身只保留 full-VQ 平均 DINO 的 `58.43%`，VAR 补全后提高到 `94.91%`，追回 `87.76%` 的剩余 DINO 缺口。
- m9 发送 full-VQ 的 `62.35%` raw index bits。官方 Decoder 下，前缀本身保留 `84.38%` DINO，VAR 补全后提高到 `98.38%`，追回 `89.64%` 的剩余缺口。
- 使用微调 Decoder 时结论一致：m8/m9 的 VAR 补全分别把 DINO retention 从 `60.72%/84.86%` 提高到 `94.48%/98.56%`。

因此，更准确的系统表述是：**发送端用低尺度真实 token 锚定内容，接收端 VAR 用生成先验和预共享类别补出未发送尺度，以计算和先验知识替代一部分通信 payload。** 不能表述成“VAR 自己传输了这些语义”。

## 严格配对设计

测试总体为此前冻结的同一批 ImageNet-1k validation 100 张图，预处理统一为短边 bicubic resize 到 256 后 center crop 256。没有训练或修改 VQ tokenizer、VAR、官方 Decoder 或微调 Decoder。

对官方与微调 Decoder 分别比较三种输入：

1. `full-VQ`：10 个尺度全部使用真实 VQ token；
2. `prefix-only`：只使用前 8 或 9 个真实尺度，未发送 residual scale contribution 保持为零，不运行 VAR；
3. `VAR-completed`：使用与 `prefix-only` 完全相同的真实前缀 token，再由冻结 VAR 闭环 argmax 补全剩余尺度。

主归因量是同图、同前缀、同 Decoder 下的 `VAR-completed − prefix-only`。两臂通信 payload 完全相同，唯一差异是接收端是否运行 VAR。PSNR、SSIM、LPIPS-Alex 和 DINOv2 ViT-S/14 使用同一实现；置信区间为 10,000 次 image bootstrap paired 95% CI。

实现还将 10 个真实尺度按 progressive prefix 路径重新累积，并与原 full-VQ `f_hat` 比较，最大绝对误差为 `0.0`，确认 prefix-only 的 latent 构造与 full-VQ 路径一致。

## 通信资源账本

codebook 大小为 4096，因此每个 raw VQ index 固定计 `12 bit`。尺度为 `[1,2,3,4,5,6,8,10,13,16]`。

| 预算 | 已发送 token | raw index bits | 比 full 少 | 节省比例 | full/当前 bits |
|---|---:|---:|---:|---:|---:|
| m8 | 255 | 3,060 | 5,100 | 62.50% | 2.667× |
| m9 | 424 | 5,088 | 3,072 | 37.647% | 1.604× |
| full | 680 | 8,160 | 0 | 0% | 1.000× |

m8 未发送的是第 9、10 尺度，共 `13²+16²=425` 个 token；m9 未发送的是第 10 尺度，共 `16²=256` 个 token。这些 token 不是由信道送达，而是由接收端 VAR 预测。

当前实验沿用既有协议：ground-truth ImageNet class index 预共享，计 `0 bit`。如果部署时必须发送类别，保守加 `10 bit` 后：

- m8 为 `3,070 bit`，相对 full 仍节省 `62.377%`；
- m9 为 `5,098 bit`，相对 full 仍节省 `37.525%`。

这里是 source-representation 的 raw index payload，不含 entropy coding、packet header、FEC、调制和物理信道噪声，不能直接当作端到端净码率或可靠吞吐量。

## 官方 Decoder 结果

### 绝对均值与语义保留

| 条件 | bits | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | DINO ↑ | DINO retention |
|---|---:|---:|---:|---:|---:|---:|
| m8 prefix-only | 3,060 | 18.5008 | 0.47480 | 0.30565 | 0.54831 | 58.43% |
| m8 VAR-completed | 3,060 | 19.6805 | 0.51191 | 0.17760 | 0.89060 | 94.91% |
| m9 prefix-only | 5,088 | 20.2390 | 0.53685 | 0.20286 | 0.79175 | 84.38% |
| m9 VAR-completed | 5,088 | 21.2316 | 0.56976 | 0.13589 | 0.92317 | 98.38% |
| full-VQ | 8,160 | 22.8357 | 0.63801 | 0.10569 | 0.93836 | 100% |

### VAR 的同前缀边际贡献

| 预算 | 指标 | VAR − prefix-only | paired 95% CI | VAR 胜图数 |
|---|---|---:|---:|---:|
| m8 | PSNR ↑ | +1.17961 dB | [+0.96010, +1.42897] | 91/100 |
| m8 | SSIM ↑ | +0.037102 | [+0.030972, +0.043518] | 95/100 |
| m8 | LPIPS ↓ | -0.128050 | [-0.136810, -0.119826] | 100/100 |
| m8 | DINO ↑ | +0.342293 | [+0.305664, +0.380167] | 100/100 |
| m9 | PSNR ↑ | +0.99255 dB | [+0.80736, +1.19886] | 93/100 |
| m9 | SSIM ↑ | +0.032914 | [+0.028411, +0.037660] | 95/100 |
| m9 | LPIPS ↓ | -0.066965 | [-0.071869, -0.062340] | 100/100 |
| m9 | DINO ↑ | +0.131423 | [+0.110745, +0.154606] | 98/100 |

官方 Decoder 下，VAR 对四项指标的平均改善置信区间都不跨零。尤其 m8 的 DINO 从 `0.5483` 提高到 `0.8906`，说明低尺度前缀主要提供粗语义锚点，而最终接近 full-VQ 的语义相似度高度依赖 VAR 补全。

## 微调 Decoder 结果

### 绝对均值与语义保留

| 条件 | bits | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | DINO ↑ | DINO retention |
|---|---:|---:|---:|---:|---:|---:|
| m8 prefix-only | 3,060 | 18.9249 | 0.50510 | 0.29755 | 0.56468 | 60.72% |
| m8 VAR-completed | 3,060 | 20.9072 | 0.55637 | 0.20046 | 0.87863 | 94.48% |
| m9 prefix-only | 5,088 | 20.8029 | 0.57149 | 0.19986 | 0.78920 | 84.86% |
| m9 VAR-completed | 5,088 | 22.4029 | 0.61007 | 0.15586 | 0.91663 | 98.56% |
| full-VQ | 8,160 | 23.8288 | 0.67207 | 0.12099 | 0.93000 | 100% |

### VAR 的同前缀边际贡献

| 预算 | 指标 | VAR − prefix-only | paired 95% CI | VAR 胜图数 |
|---|---|---:|---:|---:|
| m8 | PSNR ↑ | +1.98230 dB | [+1.78312, +2.19735] | 100/100 |
| m8 | SSIM ↑ | +0.051269 | [+0.045176, +0.057623] | 99/100 |
| m8 | LPIPS ↓ | -0.097091 | [-0.104366, -0.089867] | 100/100 |
| m8 | DINO ↑ | +0.313951 | [+0.279106, +0.349447] | 100/100 |
| m9 | PSNR ↑ | +1.59999 dB | [+1.41551, +1.80006] | 100/100 |
| m9 | SSIM ↑ | +0.038582 | [+0.034326, +0.043093] | 100/100 |
| m9 | LPIPS ↓ | -0.044000 | [-0.048553, -0.039286] | 97/100 |
| m9 | DINO ↑ | +0.127432 | [+0.107183, +0.148899] | 96/100 |

微调 Decoder 改变了像素/感知取舍，但没有改变归因结论：相同 m8/m9 前缀下，运行 VAR 都显著提高 PSNR、SSIM、LPIPS 和 DINO。VAR 追回的 DINO 缺口比例分别为 `85.94%` 和 `90.51%`。

## 对“VAR 起多大作用”的直接回答

1. **语义保留作用：** m8 是最强证据。只发送 m8 前缀而不运行 VAR，DINO retention 只有约 `58–61%`；运行 VAR 后为约 `94.5–94.9%`。也就是 VAR 单独贡献约 `+0.314～+0.342` DINO，并追回约 `86–88%` 的 full-VQ 语义缺口。
2. **通信节省作用：** m8/m9 分别少发 `5,100/3,072 raw bits`，即节省 `62.5%/37.65%` VQ index payload；接收端 VAR 预测相应的 `425/256` 个未发送 token。
3. **m9 的定位：** m9 前缀本身已经较强，DINO retention 约 `84–85%`，但 VAR 仍把它推到约 `98.4–98.6%`，追回约 `90%` 的剩余语义缺口。
4. **不是无代价压缩：** 即便有 VAR，官方 Decoder 的 m8/m9 相对 full-VQ 仍分别少 `3.155/1.604 dB` PSNR，LPIPS 分别高 `0.07191/0.03020`。VAR 更擅长恢复语义和自然纹理，而不能完整恢复所有源图像细节。

## 边界与后续必要检查

- 当前类别条件是真实且预共享的。如果类别不可预共享，需要把类别 bit 纳入账本，或使用接收端类别预测，并单独测试类别错误时的 semantic drift。
- 本轮是 noiseless source-representation 消融。数字 VAR 的真实信道结果仍应使用已经修正的标准复 AWGN、FEC 和 packet success 口径单独报告。
- DINO retention 是均值比值，用于当前系统内部配对定位，不等价于任务准确率、互信息或所有下游语义性能。
- `prefix-only` 是“未发送 residual contribution 置零”的自然接收端对照，不代表经过专门训练的无 VAR scalable decoder 上界。

## 产物与复核

正式目录：`outputs/analysis/ANALYSIS-VAR-SEMANTIC-RATE-CONTRIBUTION-IMAGENET100-001/`

- `per_image_metrics.csv`：1,000 行逐图结果；
- `mean_metrics.csv`：10 个 condition group；
- `paired_var_gain.csv`：16 个 Decoder×预算×指标配对比较；
- `rate_semantic_summary.csv`：码率、DINO retention 和 full-gap recovery；
- `budget_ledger.json`：raw bits 与类别 side-information 账本；
- `semantic_retention_vs_bits.png`、`var_marginal_gain.png`、`representative_grid.png`：三张主图。

独立复核确认：10 组各 100 张、1,000 个唯一配对键、无非有限指标；从逐图 CSV 重算的所有均值和 paired delta 与正式汇总逐浮点一致。配置、脚本和模块分别与输出快照 SHA 完全一致；full progressive replay 最大误差为 `0.0`。
