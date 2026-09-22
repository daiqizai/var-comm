# 当前实际接续：Joint E/R训练充分性控制

**2026-09-14 16:48本试验已全部完成并冻结。** 恢复后100张、33600主行及全部统计/CPU复核通过；旧trainer/reviewer/evaluator/恢复器/观察器均结束，不再重启。结果`reports/wetok_joint_result_20260914.md`。当前活动研究已转到`../wetok-joint-grid-controls-r1/docs/CONTINUE.md`，两匹配控制已实际训练。下方等待/运行记录全部为历史。

研究总目标未完成；原RX-only试验已完整结束并冻结。Joint5000训练、完整校准和审计均已PASS。原评测在49张提交后因其它GPU任务触发保护退出，**当前由恢复监督器等待空闲后续跑**。不要重启trainer、旧finisher或另开重复评测。

## 活动任务

- 2026-09-14 16:25最新核实：第一次恢复已从49张推进至**55张**；新evaluator480961在16:19再次遇到GPU进程482008并按保护退出。恢复监督器473779/观察器473780仍存活，已自动回到等待状态，不需要手动再启动。`evaluation_attempt_001.json`保存实际子进程退出与提交计数；原49张回执和全部冻结源码校验不变。

- **恢复监督器473779 / 只读观察器473780**，2026-09-14 16:12启动并已实际核实存活。入口`outputs/WETOK-JOINT-SENDER-R1-EVAL-RECOVERY-20260914-001/status.json`；该目录保存资格、旧失败现场、每次恢复的日志及后续分析日志。
- 当前前49张已提交、16464主行保留。旧evaluator443167/finisher348503/observer357923均已退出；不能把这些旧PID写成仍在运行。GPU释放并连续30秒检查空闲后，恢复器自动执行原`evaluate.py --resume`，再执行原分析。
- 当前资格重新核验PASS、共31项CPU测试PASS。没有修改原模型、评测/统计或其源文件哈希；不停止其它GPU任务、不重新训练。细节`docs/evaluation_recovery_20260914.md`。

## 已完成训练与第一次评测历史

- 2026-09-14 15:50，原trainer348501/reviewer348502已正常结束。**evaluator443167、finisher348503、只读观察器357923**实际存活，已保存首张源图336行。以`outputs/WETOK-JOINT-SENDER-R1-EVALUATION/step_0005000/status.json`及`outputs/WETOK-JOINT-SENDER-R1-step0005000-finish.pipeline.json`为当前入口。
- Joint三臂此时均选择5000：single .197125、no-history .197853、state .194679。state在完整校准上优于两个基础版，但仍不及R-only full-grid .187311；不先当独立确认或强系统胜出。报告`reports/wetok_joint_milestone5000_20260914.md`，图`reports/wetok_joint_calibration_figures_20260914.md`。
- 下方1000/2500/训练中PID条目为历史证据，不据其重启。最终统计仍由现有finisher自动运行。

- 12:51完成起点完整校准并保存首个真实Joint更新；CPU读取实际resume再次核实三臂E均已改变、历史和Adam均为step1。不是只做profile或只启动空进程。该梯度来自当前联合目标，image-only通路的非零梯度由先前独立profile验证。

- trainer **348501** / review watcher **348502** / finisher **348503**，必须以实际/proc和GPU进程核查。2026-09-14 14:37从实际2500末端继续5000；原274550/274551与188608/188609均正常结束，不重启。14:33检查有其它GPU任务，待其离开后通过空闲检查才启动，未停止他人进程。
- 状态：`outputs/WETOK-JOINT-SENDER-R1-TRAINING/status.json`。
- 2026-09-14 15:10实际核实3600/5000保存点及三进程均存活，3500监控已结束。state监控LPIPS由3000的.197592波动到3500的.201347；这些不是全校准，不据此改配方、重启或选模，继续原5000流程。
- 14:41发现其它GPU工作在运行期间加入。只读观察器**357923**从14:44记录并发/硬件背景，`outputs/WETOK-JOINT-SENDER-R1-PASSIVE-GPU-20260914/`；不改训练或停他人任务。最终时延若受并发影响，不作独占速度结论，见`docs/passive_timing_context_20260914.md`。
- 流程：`outputs/WETOK-JOINT-SENDER-R1-step0005000.pipeline.json`及`outputs/WETOK-JOINT-SENDER-R1-step0005000-finish.pipeline.json`。
- 日志：`outputs/WETOK-JOINT-SENDER-R1-train-0005000.log`、`outputs/WETOK-JOINT-SENDER-R1-watch-0005000.log`、`outputs/WETOK-JOINT-SENDER-R1-finish-0005000.log`；评测阶段日志为`outputs/WETOK-JOINT-SENDER-R1-step0005000-finish.stages.log`。
- 2500已完成；single/state选中2500，no-history仍选中0。校准LPIPS为.197908/.198812/.200233；state仍不及single，不能宣布尺度结构成立。恢复的是实际2500模型/Adam，不是选中0；`docs/continue_5000.md`。初始训练同R-only原7000 single父点和fresh Adam规则，当前global data从9500续至12000。
- 只新增通信E的优化，R仍训练；204×30、N3060/E6120、原loss/学习率/micro1/effective4、视觉冻结和自身接收历史均不变。控制各5000机会已完成且reference资格PASS。

## 原RX结论

主1/4/7 development：single LPIPS .242495，multiscale innovation .242830，full-grid .236632；数字自适应 .183763、感知Deep .205291。多尺度创新量对single的区间跨0，且没有对full-grid的实测速度优势。
full-grid相对single有LPIPS -.005862、PSNR +.14277 dB、DINO +.01089，但不能叫next-scale，也未分离普通迭代和重编码。13/19 dB它可胜旧VAR/Deep的LPIPS，仍弱于同WeTok数字链。原候选停止继续扩展，六臂结果保留作强对照。报告`reports/wetok_innovation_result_20260914.md`。

## 资格与实现

- 真实GPU图像梯度与冻结权重检查PASS，原16项加收尾工具6项及只读观察器5项共27项CPU测试PASS。后5项只检查观测工具，不是新增模型质量证据。输入梯度通过R/信道回到E，精确微批次VJP不是ST近似。
- `outputs/WETOK-JOINT-SENDER-R1-PREPARATION/reference_qualification.json`已封存原RX5000的评测与分析；不可覆盖或改变SHA放行。
- 新评测会重测Joint三结构和全部六R-only结构，另留七系统参考，共33600主行＋600支持点；不会只挑弱对照。Joint与R-only不是同y，只是同源/标准噪声/物理资源和训练机会。
- 训练/profile、初始化/reference资格、本次train metadata及收尾队列绑定源码已冻结，包含Joint评测和新finisher。不可边跑边改；有真实缺陷要保留原证据、明确另立修订。`continue_5000.md`也已绑定，不往该文件追加实时状态；用本文件/PROGRESS记录。

## 下一步

1. 核实恢复监督器473779及观察器473780；若出现新evaluator，其PID记录在恢复状态中。1000/2500/5000训练审计均已完成，不覆盖重跑；不恢复已结束的trainer。
2. 恢复器等待GPU空闲后执行原`evaluate.py --step 5000 --resume --execute`，之后自动运行原`analyze.py`；**不手动启动第二份评测或统计**。只有明确的GPU竞争退出才可自动重试，其它错误保留现场检查。
3. 检查33600主行、600支持点、900无噪诊断及全部强控制，依据实际开发结果决定后续；不以监控子集或bit误差下降宣布成功。
4. 若Joint条件结构有图像增量，还需对应Joint普通全网格控制，不能将多计算或E解冻本身当尺度创新。保留当前full-grid强参照。
5. 仍需要合理强对照、必要训练重复、鲁棒性和新holdout；整体研究远未完成，不标记goal complete。

解释器仍为`experiments/backbone-eval-20260912/.venv/bin/python`。不改驱动/时钟/MPS/全局环境，不停其它GPU任务，不租新服务，不提交/push。

最新报告`reports/wetok_joint_milestone2500_20260914.md`；启动回执`outputs/WETOK-JOINT-SENDER-R1-CONTINUE5000-20260914/launch.json`。临时启动器348398已完成退出，不是需要重启的训练进程。原RX的清晰展示图与单图入口`reports/wetok_receiver_figures_20260914.md`，Deep支持点和旧条件异常分开标注；没有更改任何原性能结果。

CPU期间补充的HJSCC资格核查见`reports/hjscc_source_protocol_review_20260914.md`：作者代码小型快照已冻结，但没有加载权重、跑性能或启动新的科学训练；不影响当前5000自动流水线。
