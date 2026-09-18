# m8数字基础＋连续latent增强：development评测协议

登记日期：2026-09-18（Asia/Shanghai）。本轮只使用原100张development图，不访问新的holdout，不重新训练，不重新挑选阶段A/B checkpoint。

## 目标

把阶段B三个已经选定的训练checkpoint与原m8基础、同Dc接收端精化及同总资源数字方案放在同一批图像、SNR、噪声和指标实现下比较。结果分为：

1. Decoder适配和新增连续观测的质量差异；
2. 同一总N/E下，连续增强与raw/整帧VAR概率算术链的取舍；
3. 后续另行登记的完整TX/RX在线成本。

不在development上逐图选择模式，不用真实图像质量选择输出。raw和arithmetic的m7/m8/m9全部保留，分别以D0和阶段A选定Dc渲染，避免把Decoder收益误记为源信息收益。

## 固定通信条件

| 家族 | N | header/data复使用 | E | 模式 |
|---|---:|---|---:|---|
| raw | 3060/3572/4084 | 68/(N−68) | 6120/7144/8168 | m7/m8/m9 |
| arithmetic | 3060/3572/4084 | 94/(N−94) | 6120/7144/8168 | m7/m8/m9 |
| 连续增强 | 3572/4084 | 68+2992/512或1024 | 7144/8168 | 固定m8基础 |

每个复符号平均能量为2；AWGN每实坐标方差为`1/γ`。增强支路只对自身坐标归一化，原3060基础波形坐标不因追加支路而改变。类别、mode、length、CRC和tail按各协议计费。

raw使用原header协议和同一终止K7卷积码，仅将数据端实际使用量改为`N−68`；arithmetic使用原VAR概率、13-bit length field、94-use header和`N−94`数据使用量。若码流在低资源点发生译码失败，保留实际候选或预定灰图，不用真值修复。

## 数据与统计

- source：原`COMMUNICATION-CONVERGENCE-20260915/DEVELOPMENT_001`的100个target，绑定原`source_tokens.npz`的前100个target-development条目及原预处理像素；
- SNR：1、4、7、13、19 dB；噪声seed：2001、2002、2003；
- 指标：PSNR、LPIPS-Alex、DINO cosine。DINO只报告，不做本轮选择；既往数字模式开发曾暴露DINO，报告中保留该限制；
- 每张源图先平均三个噪声，再进行源图级统计与配对区间；失败帧不能删除；
- 每个source目录保存逐帧CSV、重建数组、接收状态和SHA。评测可断点续跑，已完成source不重算。

本轮结束后再写质量汇总；不要把不同N混成单一排名。在线TX/RX时延、峰值显存、参数量和VAR补全成本另在固定子集上测量，不能用这轮包含指标计算的总时间冒充在线时延。
