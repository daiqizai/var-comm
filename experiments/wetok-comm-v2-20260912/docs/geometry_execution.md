# Geometry实际执行入口

本轮在本机UTC+8日志2026-09-13启动；原接口5000的训练/评测/统计已完成，不再自动扩大相同九臂。

## 当前任务

- 最新已完成总4500/image2500的匹配全校准审计。新LPIPS .203079/.204214/.203916，旧同预算.257791/.254603/.256377；不能把共同geometry收益归为条件历史。当前从实际4500继续总7000，trainer **2973427** / watcher **2974929**，日志geometry_train_total0007000.log / geometry_watch_total0007000.log，pipeline geometry_total_0007000_pipeline.json。决定docs/geometry_continue_7000.md。

- 最新已完成总3000与同image1000控制的完整校准/审计；新geometry全校准LPIPS .206725/.206953/.207850，较同结构旧geometry改善约.0485–.0541。只是calibration证据，条件结构尚无独立优势，不与development强基线混排。
- 当前已从实际总3000末端继续到**总4500=image2500**，trainer **2941742**、watcher **2943268**；日志geometry_train_total0004500.log / geometry_watch_total0004500.log，pipeline geometry_total_0004500_pipeline.json。下方3000 PID为已结束历史。

- 新geometry：204×30，三结构single_pass/no_history/conditioned；continuous_mean接口、原视觉模型/loss/N/E不变。
- 控制153×40三结构复用真实完整2000表示恢复+5000图像训练；资格`WETOK-GEOMETRY-CONTROL-QUALIFICATION-20260913/qualification.json`。控制不是不同历史的旧best替代品。
- 新训练先2000表示步、边界fresh Adam，再5000图像步。当前首总3000=2000+1000，不是只训3000图，也不是总研究上限。
- trainer **2902777**，watcher **2907659**；必须核查实际/proc，不凭PID文字认定存活。日志`geometry_train_total0003000.log`和`geometry_watch_total0003000.log`。
- 当前状态：`VAR_COMM/outputs/WETOK-GEOMETRY-20260913-TRAINING/status.json`；流水线`VAR_COMM/outputs/WETOK-COMM-V2-20260912-LOGS/geometry_total_0003000_pipeline.json`。
- 本机00:53已完成2000表示更新并同时重置三臂Adam，随后进行image0全校准；新geometry仍没有开发集成绩。

## 已验证与边界

31项CPU检查通过；原shape控制参数/前向、表示loss/梯度及一次Adam更新与旧实现一致。真实GPU image梯度到新E/D、train/eval像素一致、功率误差0、冻结视觉参数未变，profile无更新。

控制初始化最初在2 CPU线程下哈希不匹配；原random输入相同，但QR末位不同。恢复原8线程后完全匹配历史hash，未放宽阈值或改预期SHA。新profile/训练均用configure_torch的8线程。该初始化环境是控制资格的一部分。

候选2927358参数、控制2895944，memory204/153，不能声称严格同参数或同FLOPs。初始线性秩不是完整非线性码容量；本轮只检验geometry实现包，条件机制仍须同geometry内对照。

## 后续正常步骤

1. 观察当前训练，首总3000自然完成后watcher运行review_geometry，按相同image1000预算对旧控制的0/1000候选池比较；不拿新image1000对旧最佳image5000作因果对照。
2. 根据完整校准决定恢复到总4500(image2500)、总7000(image5000)。只使用这些有合格控制的边界；不得随意增加新独有的全校准选模点。
3. 当前profile/训练绑定源码与configs不得边跑边改。新工具另立文件；不要为修复哈希直接改旧记录。非有限/物理/冻结错误保留现场并调查。
4. **新geometry开发评测/统计入口已实现但尚未正式GPU执行**：evaluate_geometry.py / analyze_geometry.py，只接受总7000匹配预算。旧新六模型同场重测、四强参考、21000主行+600支持点、1008配对区间+48交互区间；441条旧控制/强参考CPU复核和合成整链测试通过。不能把这些工程检查当新模型成绩。
5. 新geometry尚无正面性能结论，不访问新holdout；最终仍需足够训练、训练重复和独立验证。

当前全部34项CPU测试通过。总7000完成并通过校准审计后，才使用evaluate_geometry.py --total 7000 --execute和analyze_geometry.py --total 7000 --execute；它们不会接受用较短新训练预算去对比旧image5000控制。

当前进程已经运行，不重复启动。未来确认它结束并作科学决定后才使用：

```bash
../backbone-eval-20260912/.venv/bin/python -u -B scripts/train_geometry.py --until-total 4500 --resume --execute
```

完整比较协议：`docs/geometry_training_protocol.md`。其中末段排版损坏的“等…”句意为“不把kernel变大等同通信贡献”，SHA绑定原文保持不改写。
