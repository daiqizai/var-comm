# 当前：R3向量归因已登记，尚未实现或启动

## 当前：训练/校准已结束，用户要求收敛暂停

本机UTC+8 2026-09-15 02:00:18，R3原10000上限训练/完整校准/模型Adam及数据审计完成；1099666/1099667/1099668已退出，不重启。新传输评测/独立计时没有启动，不据已实现代码假定已完成或已排队。

用户最新要求优先：阅读`reports/convergence_report_20260915.md`（相对VAR_COMM）。暂停新增架构、自动参数搜索、大网络或复杂机制，报告完成不自动解除暂停；不按下方旧“立即补齐并挂队列”继续启动GPU任务。保留全部checkpoint、源代码、测试与失败记录。

R3五SNR完整校准从7500的0.176963降至10000的0.172232；对R2 residual同10000的0.174380有LPIPS均值改善但PSNR较低。只有校准，不是development或独立test结论。只保留此一学习待验证候选资格，不加训；数字自适应VAR重列候选主系统。

原型辅助评测驱动、timing、finisher已实现，11项CPU测试通过，但未实际启动；它们只保留备查。后续是否补纯评测/计时或新验证，须遵守新的收敛边界，不自动恢复旧持续扩展队列。

## 用户反馈边界

先读`docs/mainline_boundary_20260915.md`：R3只按已登记范围收尾，不自动衍生更多通用接收器小变体；之后回到尺度结构在通信中的具体作用，不能用实验数量代替主方法进展。当前训练1099666/审计1099667仍正常，01:12已到5000完整校准。评测的contracts/references/archive audit及quality/analyze驱动已新增但尚未完成测试/排队；timing/finisher还需补齐，不启动重复训练。

## 最新：已实现、首1000审计PASS，实际继续10000

本机UTC+8，2026-09-15 00:35首1000完整校准/模型Adam/配对数据/功率/选择审计PASS。00:38由实际1000端点继续10000；当前**trainer1099666 / reviewer1099667 / observer1099668**。00:46读取实际2000模型/Adam，global9000，前1000完整历史和校准/源码保持不变。报告`reports/wetok_r3_activation_20260915.md`（相对VAR_COMM）。下方“未实现/未启动”是登记时历史，不能据此另起副本。

模型与训练/恢复、CPU资格、真实GPU零更新profile、review/watch已实现。4项CPU/实际driver断点测试PASS；起点真实参数/CPU函数/GPU波形RGB完全匹配原残差初始点，全部10000参考批次已核验。首1000同机会校准prediction LPIPS0.198349、residual0.202011，只是过程结果，不改变10000预算。

1. 核实/proc和`outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-TRAINING/status.json`后继续已有队列，不启动重复训练。新日志`outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-train-0010000.log`，流程`outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-step0010000.pipeline.json`。
2. **立即补齐最终评测驱动/测试/自动收尾**，可与当前训练并行开发；只有配置`configs/evaluation.yaml`与`docs/evaluation_protocol.md`已登记，不能假定最终质量已排队。23方法48300主行（新2100）、1600无噪、600支持点；主1056区间、支持288（新18），独立16模型2560计时行/105区间。
3. 保留全部R2的22个冻结参考，明确R3对R2 residual是同参数/回算/完整历史的主要对照；其它R2和更早参考作用不同。质量与独立时延分开，不能新方法用新时延、旧方法复用历史时延。
4. 当前训练/配置/原protocol/模型/common与GPU profile/CPU资格均绑定SHA，不边跑边改或篡改哈希放行。评测可以新增文件，但不得改这些已绑定文件或旧R2源码/结果。新增可读解释见`docs/interpretation_limits.md`。
5. 目前没有R3 development、新holdout或最终机制结论；不要从首1000校准提前宣布胜负。当前连续状态残差不是合法native消息的精确似然，全网格不改名next-scale。

原1000模型/Adam SHA `d01257598f5f4c1c02235a6427adc7a0a276d4650a5c2ded6acadaa323365821`；原1000审计回执SHA `793a32f3abc85c926b3f4ec112346b2f6f279eebfd657a8cdd293d4c690f776f`。CPU/GPU准备证据分别位于本轮PREPARATION和PROFILE目录；GPU入口叫`profile_model.py`，不用会遮蔽标准库的`profile.py`。

以下为登记时的历史说明，当前以上方为准。

研究总目标未完成。先读本目录AGENTS和`protocol.md`，并读`reports/wetok_r2_result_20260914.md`（相对VAR_COMM）。R2四臂10000、46200质量主行、48400图像CPU审计、2400独立计时行已完整结束，原880742/880743/983552/998565/1004571等进程已退出，不据旧状态重启。

1. 先从实际旧Grid/R2/原7000父点/初始化/Adam及数据历史建立新控制资格，保留所有旧文件只读；不要从已优化的残差权重直接切换feature后短训。
2. 只实现full-grid prediction-vector对照，三次16读取、两次E、同融合/门控/参数；门控仍用残差能量，只有融合向量q vs y−q变化。不得改冻结旧类；新类/驱动放本目录。
3. CPU检查初始状态/初始函数等价、同参数/E调用、自己的接收历史、完整图像梯度路径、配对数据与断点恢复；再在GPU空闲时做实际小型profile，正常推进训练，不把检查扩成科学gate。
4. 新臂完整10000机会、原cal选择点与全部数据流；合格的R2 residual10000只读复用。新源码/协议在实际启动时绑定。当前没有新代码、checkpoint或GPU任务，不能把登记当成训练。
5. 完整结果后再解释向量残差是否超过通用重编码特征，保留强系统与成本；全局迭代不是next-scale，连续假设残差不是精确合法消息似然。未访问新holdout。
