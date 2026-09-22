# 当前接续：四臂训练充分性R2运行中

## 已完整结束；新接续转R3准备

2026-09-14 23:30，R2质量/CPU分析/独立计时全部完成；46200主行、48400图像CPU核验、3744主配对区间及2400独立计时行均已复核。所有旧训练/收尾/观察进程已退出，不重启。完整报告`reports/wetok_r2_result_20260914.md`。

主1/4/7 dB，MS−普通full-grid LPIPS **+0.002350**，CI[+0.000808,+0.003969]，DINO更低、无明确速度优势；当前4/8/16不再机械续训。innovation LPIPS0.220543较普通full-grid0.230345好，但仍落后数字自适应0.183763/感知Deep0.205291；增加6417参数、两次E及1.68758ms RX时间，不能直接归因物理残差或称next-scale。

新准备入口 **`../wetok-reencoding-vector-control-r3/docs/CONTINUE.md`**：同全网格/参数/E调用/训练机会，仅比较融合向量q与y−q；残差能量门控仍相同。R3当前只有登记，未实现/未训练。先资格核验原7000父点、零融合初始化与完整训练历史，不能从R2残差最优点切换短训冒充公平控制。

本目录训练/评测/图表和对应源码已封存，后续只读引用。研究总目标未完成，没有新holdout或训练重复。以下活动PID/阶段描述均为历史。

## 当前：10000训练/完整校准/审计完成，质量评测已启动

2026-09-14 23:09:03，四臂实际10000模型/Adam、global17000、继承历史与新5000批次/噪声/功率及选模审计PASS。single/MS选7500，ordinary full-grid/innovation选10000。报告`reports/wetok_r2_milestone10000_20260914.md`。10000完整校准/训练/成本PNG/PDF/CSV已存`outputs/WETOK-JOINT-SUFFICIENCY-R2-ANALYSIS/figures_0010000/`，不覆盖7500。

当前**quality983552 / finisher880742 / observer880743**，23:12实际提交23/100源图。原trainer880740/reviewer880741已正常退出，不重启。流程仍见`outputs/WETOK-JOINT-SUFFICIENCY-R2-step0010000-finish.pipeline.json`；动态子进程/日志路径以该文件及/proc核实。最终质量还未完成，不从部分行得出排名。

1. 等同一quality进程完整提交100源；如遇共享资源保护，由现有finisher按原协议续跑，不另开副本。
2. 随后现有队列自动做CPU产物/统计审计和独立15模型计时；质量可共享，计时遇其它GPU进程等待，不停他人任务。
3. `scripts/report_quality.py --execute`已准备并通过测试，只能在quality和quality_analysis回执完整后画图；保留同WeTok数字、VAR固定/自适应和Deep强参考，不能把不同训练历史当同机会结构因果。全套19项CPU测试PASS。
4. 训练、质量、分析和计时源码/协议仍绑定SHA，不边跑边改。所有已封存旧结果只读。
5. 最终仍需解释主1/4/7 dB、各支持SNR、无噪映射差距、训练预算和额外E调用/融合参数；不得把普通迭代改名next-scale或据校准宣布系统成功。研究总目标未完成，后续修订依据完整结果。

以下为已完成训练阶段的保留记录，当前动作以上方为准。

## 最新：7500审计完成，实际续到10000

22:09核实已到8400总更新，四个自有进程仍存活。完整7500训练/校准曲线和分项成本已生成并查看，见`reports/wetok_r2_training_curves_20260914.md`；PNG/PDF/CSV在`outputs/WETOK-JOINT-SUFFICIENCY-R2-ANALYSIS/figures_0007500/`。新增成本/绘图四项检查后全套15项CPU测试PASS；训练/已排队收尾源码绑定未改。`scripts/report_training.py`可在10000完整审计结束后生成新的10000图表目录，不能覆盖已封存7500图表。

2026-09-14 21:35完成7500全1000图校准与CPU模型/Adam/数据/功率/选模审计，四臂LPIPS均改善，均选择7500。报告`reports/wetok_r2_milestone7500_20260914.md`。21:43从实际7500模型和Adam续到10000，当前**trainer880740 / reviewer880741 / finisher880742 / observer880743**；21:46实际已保存7600总更新、R2新增2600、global14600。下方769675等为已结束历史进程，不重启。

训练状态仍为`outputs/WETOK-JOINT-SUFFICIENCY-R2-TRAINING/status.json`；新日志`outputs/WETOK-JOINT-SUFFICIENCY-R2-train-0010000.log`。审计流程`outputs/WETOK-JOINT-SUFFICIENCY-R2-step0010000.pipeline.json`；收尾流程`outputs/WETOK-JOINT-SUFFICIENCY-R2-step0010000-finish.pipeline.json`；背景观察`outputs/WETOK-JOINT-SUFFICIENCY-R2-PASSIVE-GPU-10000-20260914/`。始终按/proc身份与新日志确认，不因状态文件旧就重启。

**质量/统计/独立计时收尾已实现、11项CPU测试通过，并实际排队。** 包含真实质量driver断点续跑、18个旧参考不变与保存图像/波形/特征CPU审计；真实15模型计时driver从三层波形档案回放、共享竞争退出后续跑；2400行/70个时延配对区间及严格资源错误分类测试。

1. 不再开训练或评测副本；等待真实10000全校准和reviewer审计。10000总更新对应本R2新增5000、global17000。
2. finisher随后只给四新模型做质量推理，保留全部18旧方法，完成46200主行及CPU配对分析；质量显式共享准入，当前时延字段为空。
3. 固定32源×五SNR×seed2001、全部15学习模型独立计时；遇其它GPU进程仅等待/保留已提交源图，不停别人的任务。
4. 训练与收尾源码/config/protocol均已绑定SHA。不要边跑边改或改哈希绕过；实际缺陷保留现场，先停止相应待执行的自有收尾队列再明确修订，不影响已冻结旧结果。
5. 分清六个R2内部同机会结构对照、四臂相对自身5000的额外训练效果，以及历史/强系统参考；不把普通迭代叫next-scale。研究总目标仍未完成，尚无R2 development或新holdout结论。

以下为7500阶段的保留记录，当前动作以上方为准。

研究总目标未完成。原Grid5000质量、CPU复核和独立计时已经全部完成，旧finisher689944及其子进程应以实际/proc核实，不再重启旧任务。

## 已有证据

- 4/8/16对普通full-grid主LPIPS差−0.000230，CI[−0.001220,+0.000759]；DINO更低，时延区间跨0。不能继续把普通多次读取的收益直接称为next-scale优势。
- Joint full-grid innovation主LPIPS0.231342，较普通全网格0.236567改善，但仍不及Deep0.205291/数字自适应0.183763；noiseless LPIPS0.108550仍高于native0.086811。
- 最佳模型train/cal仍改善，四臂完整校准最佳点在5000；故准备同配方等机会续训，而不是马上加新模块或宣布收敛。

## 实际活动任务

2026-09-14 20:09实际启动：**trainer769675 / reviewer785157 / passive observer785158**。必须查/proc与GPU，不从旧PID或状态推断仍在运行或已停止。20:10保存5001；20:26读取真实5500点确认四臂Adam5500、global12500、原5000历史与来源不变；20:32实际状态已到5800（本轮新增800）。

状态：`outputs/WETOK-JOINT-SUFFICIENCY-R2-TRAINING/status.json`；流程：`outputs/WETOK-JOINT-SUFFICIENCY-R2-step0007500.pipeline.json`；日志`outputs/WETOK-JOINT-SUFFICIENCY-R2-train-0007500.log`和`outputs/WETOK-JOINT-SUFFICIENCY-R2-watch-0007500.log`。GPU背景在`outputs/WETOK-JOINT-SUFFICIENCY-R2-PASSIVE-GPU-7500-20260914/`。

CPU逐值恢复及真实driver断点等价测试已通过；`resume_initialization.json`绑定实际四模型/Adam，`actual_activation.json`记录真实新增更新，均在本轮PREPARATION目录。未重置optimizer、未重跑起点校准，不再从头搭建driver或开第二套训练。

`src/sufficiency/`、`configs/study.yaml`、`docs/protocol.md`、`scripts/train.py`及旧forward/VJP依赖已由运行元数据绑定SHA，不边跑边改；真实缺陷须保留现场并明确修订，不能改旧哈希放行。

已读取两个封存来源：single/MS来自`WETOK-JOINT-SENDER-R1-TRAINING/milestones/step_0005000_optimizer.pt`，普通/innovation来自`WETOK-JOINT-GRID-CONTROLS-R1-TRAINING/milestones/step_0005000_optimizer.pt`。模型类使用已冻结的JointSenderSystem和GridJointSystem，损失/VJP复用`joint_sender.runtime`。

1. 等当前7500里程碑及watcher自动审计，不再开重复训练。区分总更新数、新增更新数与global data step，不把5800总步误称新增5800。
2. 7500全校准/Adam/新数据审计后，根据曲线共同恢复到10000；必须使用实际7500模型/Adam，而不是较早selected点。
3. 固定100图监控不选模，完整1k校准只新增7500/10000；旧完整校准已继承。没有R2 development结果，后续评测仍需准备并分离质量与时延。
4. 继续记录GPU背景；不改MPS/驱动/时钟、不停其它任务，不使用未经授权设备/收费/私有数据。
5. 本轮只检验训练充分性/匹配结构稳定性，不把更多训练当新机制，普通迭代不改名next-scale。

前轮结果：`reports/wetok_joint_grid_result_20260914.md`；本轮启动：`reports/wetok_r2_activation_20260914.md`。原quality/timing均已封存；新轮仍不得访问新正式test。
