# Joint E/R 2500更新：小幅校准改善，尚无尺度结构胜出

2026-09-14 14:32完成三臂2500更新、完整校准和CPU审计。实际E/R更新、Adam步数、配对数据/增强/SNR/标准噪声、能量与LPIPS选模均PASS。未做新的development回放或holdout访问。

## 同机会校准比较

完整1000张校准图、1/4/7/13/19 dB、固定校准噪声。两侧仅在各自不超过2500更新的完整校准点中选择；监控子集不参与选择。负ΔLPIPS表示Joint较好。

| 接收结构 | Joint实际2500 LPIPS | Joint选中step / LPIPS | R-only同机会选中step / LPIPS | Joint选中−R-only |
|---|---:|---|---|---:|
| single-pass | 0.197908 | 2500 / 0.197908 | 1000 / 0.198920 | −0.001012 |
| multiscale no-history | 0.200385 | **0 / 0.200233** | **0 / 0.200233** | 0 |
| multiscale state-history | 0.198812 | 2500 / 0.198812 | 2500 / 0.201791 | −0.002979 |

single与state从1000到2500的完整校准LPIPS分别下降0.003207和0.002491；no-history末端也略降，但仍不如其起点。不能隐藏选中0，也不能把state从更差起点回升当成结构增益。

Joint state相对Joint no-history选中值为−0.001421，但相对Joint single仍为**+0.000904 LPIPS**，PSNR为21.12145 vs 21.22102 dB。当前不能宣布next-scale/多尺度条件结构优于一次性接收。

这些是选模人口上的结果，不是独立检验；不与另一批100张development的数字/Deep均值混排，也不据此替代强系统结论。

## 逐SNR的Joint−同结构R-only选中LPIPS

| SNR dB | single-pass | no-history | state-history |
|---:|---:|---:|---:|
| 1 | −0.003843 | 0 | −0.002510 |
| 4 | −0.001303 | 0 | −0.003960 |
| 7 | −0.000973 | 0 | −0.003930 |
| 13 | −0.000149 | 0 | −0.002511 |
| 19 | +0.001210 | 0 | −0.001985 |

各SNR使用同一个全校准选择的checkpoint，没有逐SNR另选最优点。Joint single在19 dB并非更好；state平均PSNR相对其R-only选中点下降约0.108 dB，应与LPIPS一起报告。

## 继续决定与边界

保持原5000更新候选计划，三臂共同从**实际2500末端模型和Adam**续训到5000，global data从9500到12000。不是从no-history选中的0点重启，不增加loss、温度、模型、功率或数据。完整曲线仍有变化且工程检查通过，值得完成原计划；不把5000当永久研发上限，也不预设继续训练必胜。

5000完整审计后执行原注册评测：三Joint、全部六R-only及七系统参考；原100张仍仅为development。若条件结构有增量，仍需匹配Joint普通全网格控制排除额外串行计算。当前不提前启动其它训练候选。

截至2500，记录的累计墙钟GPU时段约**1.73263小时**，不是专有算力计费测量。14:33续跑检查发现GPU上另有CAP-VPR进程313578；不停止或抢占，待其结束再启动5000。不能由该进程的CPU启动时间反推此前GPU重叠时段，亦不把等待计作新增训练时间。

自动收尾工具新增6项CPU检查，连同原16项共22项通过；当前训练、profile及reference绑定源码保持原SHA。它只编排既有评测/统计，不改实验数学。

## 证据入口

- `outputs/WETOK-JOINT-SENDER-R1-ANALYSIS/calibration_0002500/`：完整曲线、逐SNR选中结果和同机会差值。
- `outputs/WETOK-JOINT-SENDER-R1-TRAINING/milestones/step_0002500.json`，SHA `6e4ae8a551e314ab8e306c05b328bbd9fb3878567b8307a59da0b5485574c358`。
- `experiments/wetok-joint-sender-r1/docs/continue_5000.md`：后续固定协议；实际状态/PID以`CONTINUE.md`及进程为准。
- `experiments/wetok-joint-sender-r1/docs/completion_queue_20260914.md`：自动收尾的边界和断点行为。
