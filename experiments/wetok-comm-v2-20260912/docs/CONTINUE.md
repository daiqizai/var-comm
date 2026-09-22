# 持续研究接续入口

**本轮geometry训练与匹配开发评测已全部完成。当前实际训练移至`../wetok-innovation-r1/docs/CONTINUE.md`，trainer3072510、watcher3077070（仍须实查）。本文件后续geometry7000“当前训练”条目为历史，不据此重开旧队列。**结果见项目reports/wetok_geometry_result_review.md；旧源码/结果只读。

总目标仍active：依据工作区next_scale_communication_research_brief_v2.md，形成同N/E、同骨干与强对照、独立验证支持的通信方法。当前仅有开发机制证据，不能把实验或准备完成当整个研究完成。

## 当前唯一训练：新geometry三臂

- 当前已从总4500末端恢复到总7000：trainer **2973427**，watcher **2974929**；须实际/proc核查，不只看旧PID/状态。旧3000/4500进程均已正常结束。
- 新204×30三结构(single_pass / multiscale_no_history / multiscale_conditioned)，continuous_mean输出。固定N3060 complex uses/E6120、WeTok原生32×16×16 ±1源、视觉冻结、原loss/宽192/2层骨干不变。
- 当前目标**总7000=2000表示+5000图像更新**。总4500与匹配控制image2500预算的校准/审计已完成，新LPIPS .203079/.204214/.203916，旧同结构.257791/.254603/.256377；只是calibration，仍没有新geometry条件独立优势结论。
- 状态：outputs/WETOK-GEOMETRY-20260913-TRAINING/status.json；pipeline：outputs/WETOK-COMM-V2-20260912-LOGS/geometry_total_0007000_pipeline.json。
- 日志geometry_train_total0007000.log / geometry_watch_total0007000.log；决定docs/geometry_continue_7000.md，完整接续docs/geometry_execution.md。

## 为什么换这一项变量

- 原接口5000完整评测/审计已结束，不自动扩大旧九臂。主1/4/7 continuous条件LPIPS .279289，对无历史-.007812、CI[-.009552,-.006112]，PSNR/DINO也有配对增量。
- 但仍明显输数字自适应.183763与感知Deep.205291；noiseless约.190与正确native约.0868差距仍大。5000结果在reports/wetok_interface_milestone5000_2026-09-13.md。
- TX只读诊断显示旧153×40的线性skip秩4896、额外通道方向能量弱。新204×30可行全行秩初始化；它不是整体容量证明，也不保证图像更好。
- 参数2927358对旧2895944，memory204对153；必须报告计算差。geometry效果不自动归为纯秩或next-scale创新，同geometry条件/无历史仍是结构控制。

## 合格控制与初始化要求

- 旧153×40三个continuous控制的完整2000表示+5000图像历史只读复用，不重复训练。真实初值、2000父模型、图像阶段fresh Adam、全部数据/增强/SNR/噪声和全校准选择机会已核查。
- 资格outputs/WETOK-GEOMETRY-CONTROL-QUALIFICATION-20260913/qualification.json，SHA在configs/geometry_study.yaml。不要把不同预算或不同历史的best当控制。
- 初始化必须用原configure_torch的**8 CPU线程**。2线程曾因QR末位差导致hash失败；已恢复8线程完全匹配，没改预期SHA或阈值。见docs/geometry_control_initialization_precision.json。
- 31项CPU检查及真实新geometry GPU梯度/功率/train-eval一致性profile通过，profile更新0。它们不是性能gate，也不是新geometry质量结论。

## 下一次实际步骤

1. 核查trainer/watch真实存活，等当前总7000完整校准与review_geometry；预算已与旧控制完整image5000匹配，不随意增加新geometry独有的选模点。
2. 完整校准/审计后，在GPU空闲时执行evaluate_geometry.py --total 7000 --execute及analyze_geometry.py --total 7000 --execute，再依据强对照/条件增量/成本决定后续；不要凭calibration约.204直接宣称追平development里的Deep。
3. 当前训练/profile绑定源码与configs不得边跑边改；尤其geometry_candidate、geometry_study、interface_study/model/native/objective/training/train_geometry、协议与资格绑定测试文件。新工具另立文件；错误保留现场，不靠改SHA/容差放行。
4. **evaluate_geometry.py/analyze_geometry.py已实现，正式新geometry GPU评测尚未做。** 接受总7000匹配预算；旧新六模型同场重测，保留四强参考及600支持点、native/noiseless、参数与在线时延；21000主行、1008配对+48交互区间。34项CPU测试和441条旧图PSNR复核通过，不冒充新模型结果。
5. 达到有意义的质量/资源取舍后仍需训练重复、鲁棒性与独立holdout；目前未完成，当前100图永久development。

## 历史结果与安全边界

- 旧A0/A1、bit权重六臂和接口九臂均保留原记录；不重开旧PID/旧停止命令。bit权重试验是1000处探索性停止，不伪造跑满5000。
- 接口5000已有33600主行+600 Deep补充，3024区间独立复算误差<4e-15；600补充GPU重放像素差0。统计审计不是第二次全量神经推理或独立test。
- Deep补充真实SNR5/6、NN条件4/7，0 header+3060 data；不把原5/6条件异常当新优势。HJSCC已有层级条件先例，当前未移植/训练。
- 不改旧权重/输出、不移动大资产，不提交/push，不租未授权机器、不停他人GPU任务。新输出仅在VAR_COMM。

使用私有解释器../backbone-eval-20260912/.venv/bin/python。当前7000训练已运行，不重复启动；随后先做匹配开发评测，不盲目扩大或重启旧控制。
