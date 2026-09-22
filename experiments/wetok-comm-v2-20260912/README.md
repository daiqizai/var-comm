# WeTok原生固定预算通信：A0/A1

按工作区v2自主研究任务书执行，不受旧金额/半天/固定epoch上限约束；实际机器与数据权限仍严格遵守。

- 原生接口：32×16×16的±1特征，4组8-bit索引，8192rawbits；非VAR 4096类token。
- 首比较：single_pass、multiscale_no_history、multiscale_conditioned，同初值/2895944训练参数/数据/噪声/优化机会。
- 固定3060复信道、每符号平均能量2、总能量6120。无类别、caption和逐图mode，不硬加无用header；旧VAR的68+2992和Deep的0+3060按实际开销分别报告。
- 前两级4/8是连续通信状态，最后16级native硬符号。全部阶段读取完整y，不声称渐进到达。
- 冻结官方EMA Encoder、量化器、Decoder；图像梯度经过冻结Decoder，硬符号使用有偏identity ST，参数冻结不等于关闭输入梯度。

## 状态

当前见`docs/interface_execution.md`：原首5000已完成、bit权重续训已停止；接口对照九臂于16:06进入真实起点校准。19项CPU与真实九臂图像梯度profile通过，仍无新接口质量增益结论。旧status与命令为历史流程，不要据其重开已停止候选。

原生20k×2view/1k/100缓存及identity回放完成；CPU12项测试、真实GPU图像梯度与功率检查通过。
13:00已启动首个5000步里程碑；根据完整校准走势继续训练或修订，不把里程碑当作整个研究完成。实际状态和指标以outputs的JSON/CSV为准。

## 运行

使用已准备的私有解释器`../backbone-eval-20260912/.venv/bin/python`，不安装或修改全局环境。
所有数据/模型运行必须显式`--execute`；无标志仅显示计划。

```bash
PYTHON=../backbone-eval-20260912/.venv/bin/python
$PYTHON -B -m unittest discover -s tests -v
$PYTHON -B scripts/prepare_native_cache.py --execute
$PYTHON -B scripts/profile_training.py --execute
$PYTHON -B scripts/train_milestone.py --until-step 5000 --execute
$PYTHON -B scripts/evaluate_milestone.py --milestone 5000 --execute
$PYTHON -B scripts/analyze_milestone.py --milestone 5000 --execute
$PYTHON -B scripts/train_milestone.py --until-step 10000 --resume --execute
```

已完成的cache/profile不可重复覆盖；图像Decoder/模型/数据来源保持SHA可追溯。
已有在跑训练时，不启动重复GPU进程；监督器`finish_milestone.py`只接续本次里程碑的评测/分析，不替代之后的科学判断。

## 证据位置

协议`docs/protocol.md`；配置`configs/study.yaml`。训练与结果位于`VAR_COMM/outputs/WETOK-COMM-A0A1-20260912-*`；日志`VAR_COMM/outputs/WETOK-COMM-V2-20260912-LOGS/`。
数字对照是实际8PSK/CRC/卷积码软比特译码，不是真值修复或理想可靠传输；native完整重建参考也不是无线方法。
