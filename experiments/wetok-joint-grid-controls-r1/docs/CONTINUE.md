# 当前接续：Joint普通全网格与创新量控制

**本试验已完整完成并冻结。** 37800质量主行、1584配对区间、CPU复核及1760条独立计时均通过，结果`reports/wetok_joint_grid_result_20260914.md`。当前4/8/16没有被匹配控制支持为独立优势；普通迭代不可改名next-scale。旧trainer/reviewer/finisher/observer均应核实为已结束，不重新运行本轮。

新接续为`../wetok-joint-sufficiency-r2/docs/CONTINUE.md`，只完成四实际5000模型/Adam端点核验和协议，尚未启动GPU训练。以下活动记录全部为历史。

研究总目标未完成。原Joint三基础臂5000训练、33600主行评测和全部统计/CPU审计已于2026-09-14 16:48完成，不再启动其恢复器或旧训练。结果见`reports/wetok_joint_result_20260914.md`。

## 实际活动任务

- 2026-09-14 18:29完成2500及完整校准审计，两臂均选2500；已从实际末端共同续5000，**trainer650391 / reviewer650392 / observer650393**。19:05已核实4600更新，原1000/2500进程均已结束。
- 最新训练日志为`outputs/WETOK-JOINT-GRID-CONTROLS-R1-train-0005000.log`，校准审计流程为`outputs/WETOK-JOINT-GRID-CONTROLS-R1-step0005000.pipeline.json`。2500报告`reports/wetok_joint_grid_milestone2500_20260914.md`。
- **finisher689944 / completion observer689945**已挂上自动收尾，等待当前5000后执行“可共享质量→CPU统计→独占计时”；状态`outputs/WETOK-JOINT-GRID-CONTROLS-R1-step0005000-finish.pipeline.json`。不要手动另开一套评测。18项CPU测试及100源图/33600参考复用核验通过；`docs/evaluation_preparation_20260914.md`。

以下首2500及之前的启动记录为历史：

- 2026-09-14 17:55首1000及完整校准审计PASS，两臂均选择1000。已从实际1000模型/Adam共同继续2500，**trainer612129 / reviewer612130 / observer612131**；新入口状态仍为训练目录`status.json`，流程为`outputs/WETOK-JOINT-GRID-CONTROLS-R1-step0002500.pipeline.json`。旧570837/579794/585355均已结束，不重启。
- 当前日志改为`outputs/WETOK-JOINT-GRID-CONTROLS-R1-train-0002500.log`和`outputs/WETOK-JOINT-GRID-CONTROLS-R1-watch-0002500.log`；观察目录`outputs/WETOK-JOINT-GRID-CONTROLS-R1-PASSIVE-GPU-2500-20260914/`。启动回执`outputs/WETOK-JOINT-GRID-CONTROLS-R1-launch-2500-20260914.json`。
- 同1000机会校准：普通full-grid state .200014，Joint innovation .202011，原Joint multiscale state .201303，R-only full-grid innovation .199371。不要与原5000结果混排；报告`reports/wetok_joint_grid_milestone1000_20260914.md`。原设置不变，不据这一短校准宣布有效或失败。

以下首1000的PID及启动记录为历史：

- trainer **570837**，review watcher **579794**，只读GPU观察器 **585355**；必须以实际/proc与GPU核实，不只看旧PID或状态文件。
- 17:27启动本目录两臂至1000更新，先完成起点全1000图校准。17:35已保存100步；随后CPU读取真实保存点核实**300步**、两个E/R均改变、Adam覆盖111/117个参数张量且步数一致，原绑定源码不变。不是只做准备或空进程。
- 状态：`outputs/WETOK-JOINT-GRID-CONTROLS-R1-TRAINING/status.json`；流程：`outputs/WETOK-JOINT-GRID-CONTROLS-R1-step0001000.pipeline.json`。
- 日志：`outputs/WETOK-JOINT-GRID-CONTROLS-R1-train-0001000.log`、`outputs/WETOK-JOINT-GRID-CONTROLS-R1-watch-0001000.log`。
- 观察：`outputs/WETOK-JOINT-GRID-CONTROLS-R1-PASSIVE-GPU-20260914/`。不改MPS/驱动/时钟，不停其它任务；训练开始时核实GPU空闲，之后的并发只读记录。

## 这轮在比较什么

1. `full_grid_state_history`：16/16/16、自身历史、完整y，无内部E调用，2927358个通信参数。与已完成Joint 4/8/16 state同参数/同三次共享R，但FLOPs不严格相同。
2. `full_grid_innovation`：原有16/16/16创新量结构，RX内部重编码两次，2933775个通信参数；只改变E也训练，数学复用原R-only实现。

原7000 single父点、fresh Adam、数据/增强/SNR/标准噪声、loss、N3060/E6120、连续接收接口和视觉冻结均不变。不是从Joint5000末端继续；各新臂计划5000新更新，数据global7000→12000，与既有控制对应。旧三Joint不重训。

## 已验证与被冻结的内容

- 6项CPU测试通过：同参数/状态来源、旧创新量冻结模式对应、图像梯度、TX与内部E的微批次梯度相加、真实训练driver的断点等价和错误噪声拒绝。
- 真实CPU父点初始化已通过，初始E/R与既有父点完全同权重；启用梯度引起的已有FP32调度差在原1e−6容限内，未修改容限或旧Encoder。
- 17:23真实GPU零更新profile PASS：图像梯度到E/R，所有模型参数前后相同；两个臂的同y初始策略切换像素差为0。
- profile前反向约0.3895/0.4022秒每有效batch4，峰值分配约5.01/5.03 GB；**不包含Adam、数据、checkpoint或全校准，不能拿它当完整工期**。
- `src/grid_controls/`、`configs/study.yaml`、`docs/protocol.md`、`docs/activation_decision_20260914.md`、`scripts/train.py`及profile绑定源码已冻结。不要边跑边修改；有真实缺陷保留原证据并明确修订。
- `scripts/profile_grid.py`避免了Python标准库`profile`命名冲突。该问题在首次GPU profile前已修复，没有改任何旧结果。

## 下一步

1. 当前等5000训练完成后的现有watcher自动执行`review.py`；首1000/2500审计均完成，不重跑，不另开训练。
2. finisher已自动排队质量、统计和独立计时；所有18方法保留，旧16方法的完全相同质量记录冻结复用，不新训练旧基础模型。
3. 质量阶段明确不产出当前时延；计时按固定32图/五SNR/一个噪声重测11个学习模型。共享GPU下耗时不做独占速度结论，不改写旧记录；检查每个阶段真实PID和回执，资源检查失败按已实现边界保留现场。
4. 普通迭代不是next-scale。若当前网格因素仍有独立增量，还需注意size描述与阶段编码的关系，见`docs/attribution_limits.md`；不把一次对照当成所有机制已隔离。
5. 仍需强系统竞争力、充分训练证据、训练重复/鲁棒性及新holdout；不标记整个研究完成。

解释器：`experiments/backbone-eval-20260912/.venv/bin/python`。GPU0为已授权本机4090 D；不租新设备、不改全局依赖、不提交/push。
