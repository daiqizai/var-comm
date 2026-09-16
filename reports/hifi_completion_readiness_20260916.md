# HiFi收尾已接入，当前不等待它再做其他工作

2026-09-16。原checkpoint、完整默认采样、原三个噪声、元数据、失败回退全部保持；没有启动第二个推理worker、没有共享/抢占GPU或新增训练。

## 完成条件与自动CPU收尾

The CPU-only closure script waits for complete receipts from author_queue_001, development_hifi_001 and hifi_metrics_001; it never ranks partial results.

必须有1500次HiFi、1500同会话ADJSCC、1500作者信息假设视图；系统汇总只新增两种HiFi视图，独立ADJSCC不重复计数。机制比较使用实际同会话ADJSCC，系统比较只用实际付费N4204的HiFi和两条同N4204数字链。原四强系统和全部Swin/ADJSCC码率继续保留。所有失败均计入。

输出包括源图级配对区间、全SNR的完整RX均值/中位数/p95/最大值、实际采样步数范围、质量—资源和质量—RX图，以及：

- `reports/external_baseline_positioning_result.md`
- `reports/research_positioning_one_page_final.md`

在全量回执存在前不生成这两份最终报告，不根据部分HiFi质量重选设置；出现实现/数据异常就保留记录并停止收尾，不自行重采样或开新分支。该CPU流程不包含Git推送或聊天主动通知。

## 同ADJSCC的数值一致性说明

当前只对预登记样例0/25/50/75、1dB、seed2001作了输入资格检查，没有读取HiFi图像质量来选模。四例发送波形、数据观测和标准噪声的SHA均严格一致；但两种推理会话的ADJSCC RGB不是逐字节相同，最大绝对差分别为3.5763e−7、2.3842e−7、2.3842e−7、2.9802e−7。这与FP32舍入量级一致，具体执行分支原因未独立隔离，不能虚称全部bit-exact。

HiFi路径保留作者checkpoint autograd flags但不更新参数；裸ADJSCC独立路径不同，已知该差异不能据此被当作一次训练。全量收尾将逐图报告最大RGB差和精确相同比例，沿用原作者功能验证的2e−4容差，而不是为某个性能结果修改阈值。若超出即停下核查。机制差分使用HiFi会话内实际基础输出；独立ADJSCC的质量和计时仍保留，不以扩散共驻显存/耗时弱化它。

当前先行的一页定位见`research_positioning_one_page_20260916.md`，基于已完成非扩散结果；两者按时间/证据范围区分，不改写旧报告。没有把m9候选饱和当整体数字极限，也没有启动混合/CSI/新骨干或新holdout。
