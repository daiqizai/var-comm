# 提前完成的 14 组 selected development 评测（2026-09-29）

用户要求先评测已完成模型，并检查能否与训练并行。本轮采用 GPU 串行、CPU 核验并行：原 GPU 独占和温度门控保留，先将第三 seed 的 m8 安全保存于 17,642 步，调用未修改的原 C_evaluate 完成 14 组已停止训练、冻结 selected 的模型。未修改模型、loss、PHY、FEC、数据、选模或停止规则。

本轮是明确的部分范围交付，**不是最终 16 组网格或整个合并实验完成**。新 holdout 和内容选择器继续暂缓。原完整网格仍保留，当前结果不冒充原最终网格回执。

## 实际执行和恢复

- 56/56 真实 selected 权重 cache/online/replay 资格通过。
- 100 原 development 源 × 14 模型 × 五 SNR × 三噪声 seed = 21,000 主质量帧，1,400 封存单元。
- 12 个混合模型另有 36,000 条实际 B_RX/C_RX 无增强接收诊断记录，保留付费 N；不是免费侧信息或额外部署选择器。
- 10 原 development 源/模型、每源三次 warmup，五 SNR、seed 2001、两次 repeat：1,400 完整在线 TX/RX 计时和 420 warmup。每次质量重放误差 <=2e-5，实际波形能量逐帧保留。
- CPU 发布器核验所有 cell/seal、registration/selected/checkpoint/source bindings、完整 source/SNR/noise 键、原 CSV 全字段、摘要重算、计时覆盖/算术和全部接收预览 SHA。
- 原队列已自动接续，真实 PID/start_ticks/cmdline 与新 launch 回执一致；原 m8 训练重新运行。历史 common-metrics、N4084 等待进程保持；author/N3060 因上游显式暂停而保护退出的证据原封归档，独立 CPU 恢复器在主队列成功接回后按顺序恢复原入口。五个实际进程身份全部核验。恢复不是新增历史方法 GPU 质量验收。
- 原 thermal_guard_v2 和退休 controller 不重启，未操作他人任务或共享硬件设置。

轻量数据位于 [C_priority_selected_development_v1](../results/token_channel_efficiency_20260923/C_priority_selected_development_v1/index.json)，恢复回执另存 [C_priority_selected_resume_v1](../results/token_channel_efficiency_20260923/C_priority_selected_resume_v1/index.json)。

## 同一比较对象与结果

新 21,000 帧和此前真实 continuous/QPSK/16QAM development 的 87,000 帧按一致的 source/preprocessing/Decoder 身份合并，共 108,000 帧、7,200 次实际在线计时。保留各自 run/context/noise namespace，不宣称不同方法拥有完全相同的噪声观测。

主表 SHA：22e536dfc8bc74fd81e2ef133ae3a50e4bde5d89f9d843d5114a8a5f66706e84。合并对象 SHA：1aef30bb7528448c1b9e080cbb8be5f0ed74b4319b4bc0757942b1d1a3956a65。

第一 seed、13 dB（质量是 100 源 × 三噪声平均，时间是登记 10 源 × 两 repeat 的平均）：

| 方法 | N | PSNR dB | LPIPS | 在线 TX+RX ms |
| --- | ---: | ---: | ---: | ---: |
| H6-V | 4084 | 24.564713 | 0.078991 | 129.506754 |
| H6-P | 4084 | 24.581337 | 0.078153 | 24.819642 |
| 重新训练 selected P4084 | 4084 | 25.466113 | 0.063653 | 20.803342 |
| 复用原 selected P3060 | 3060 | 24.428483 | 0.080868 | 20.722588 |

旧 continuous_grid 的历史 P4084 10k（PSNR 25.131426、LPIPS .069738）保持独立方法名和 lineage，不与上述重新训练 P4084 混淆。第一 seed P4084 继承历史 10k，第二 seed P4084 从零训练；seed 统计不能解读为完全相同初始化历史的纯随机效应。

在该点，H6-V 相对重新训练 P4084 的 PSNR 差为 -0.901400 dB，10000 次源图 bootstrap 的 95% 区间 [-0.994588, -0.802518]；LPIPS 差 +0.015338，区间 [+0.013982, +0.016684]。primary 1/4/7 dB 平均 PSNR 差 -0.459981 dB，区间 [-0.536504, -0.381136]。这反映当前固定配置和数据，不推广为混合方案普遍无效。

总计保留 0 个 header 失败、3,514 个 body CRC 失败；失败帧仍参加平均，不删低 SNR 样本。计时为 CPU RGB 到 CPU 波形、CPU 观测到 CPU 接收 RGB，信道噪声在 RX 外；训练和缓存校准耗时未充当在线计时。

## 资源、统计与诊断

- 完整源图均值、176 项配对比较、八张质量-N/在线时间 SVG、逐帧原 CSV 无损分片、模型/context 索引、selected/lineage、失败表和实际能量账本齐全。每张图和统计从同一合并对象生成。
- 原质量目标保持冻结，m6 仅由第一 seed 校准选定。资源比较预先固定第一 seed 的连续 P2048/P3060/重新训练 P4084，以及 N3060/N4084 的 H6-V 或 H6-P。只报告测试过的最小 N，不插值、不在 development 重选 m 或 seed。
- 当前网格无正 N 节省。high：1/4 dB 双方点估计均未达；7 dB 双方 N4084，节省 0；13/19 dB 混合 N4084 对连续 N3060，节省 -33.4641%。balanced：1 dB 混合 N4084 对连续 N3060，-33.4641%；4/7/13/19 dB 混合 N3060 对连续 N2048，-49.4141%。coarse 在五个 SNR 均为混合 N3060 对连续 N2048，节省 -49.4141%；所有未达标均保留。
- 资源 bootstrap 使用相同源图重采样，区间**条件为双方均达到冻结均值目标**，另列双方达标次数和各自未达次数。例如 high 4 dB 点估计双方未达，即使少量 bootstrap 子样本达标也不能写成总体达标。
- H6-V/H6-P 各有三个训练 seed；H8-V/P4084 当前只有两个。训练 seed 标准差与源图 bootstrap 区间分开，缺失第三 seed 不补造。
- source bits、12-bit 控制头、CRC/tail、母码和 rate-match 槽数按已冻结 fixed-mode PHY 逐帧派生并明确标注，实际 E 来自在线波形测量。m8 mother 6164 bits 经原 puncture 发送 5984 coded slots，不能将 source 3060 bits 当作 N。
- 校准 source-informed 三噪声交叉诊断独立输出；该校准集已用于 codec 选择，结果不宣称无偏泛化、理论界或可部署内容选择器。B_RX/C_RX 是同次真实接收的无增强参考，不重复 TX/信道。

## 工程验证与未完范围

新增工具仅为独立 CPU 发布器和三个回归，不修改活动训练/评测/恢复 worker 依赖。首次调用因缺 source import 路径退出；第二次核验发现原 CSV 为混合/纯连续使用字段并集，空白缺省字段需按原序列化规范比较。两次均在创建发布目录前退出，日志保留；修正独立工具后完整核验通过，不是 GPU 实验失败。已发布结果目录不可覆盖或重复运行 main。

根 264 项 CPU、repository、release、fsck 已通过。Matplotlib 原始 SVG path 行尾空格按已封存 SHA 保留，diff 检查仅排除这些 SVG，其余文件检查通过。独立远端验证按工程要求执行；最终独立复验状态以 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_priority_development_verification/receipt.json 为准，不把 CPU 工程测试称为新增 GPU 质量结果。全部 122 索引文件 75,411,293 字节，每项 <10MB；八张 SVG 结构检查及同数据 PNG 视觉检查通过，未上传原图、权重、缓存或秘密。

后续仍包括第三 seed H8-V/fresh P4084 及原校准延长、原完整 selected 网格、四个历史 GPU 验收/重评分/计时 worker、最终全方法配对比较和远端交付。当前提前评测不关闭这些待办。
