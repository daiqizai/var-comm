# Bit监督权重对照：运行说明

2026-09-12约15:06启动六臂：三种receiver各自从原5000步模型与Adam状态，复制为joint_original与bit_support。仅bit BCE权重0.01→1，其它参数、源输入、native hard/ST、N/E、数据/噪声及学习率不变。
额外5000步为首研究里程碑，不是整个研发上限。初始父点较差，因此评测保留原实验选中的2000步历史最佳；只赢退化的续训不能称整体通信已提高。

## 实测与审计

- 16项CPU测试通过：同父模型/Adam值相等但存储独立、仅一个权重改变、完整27300行统计网格、原native/FEC与历史依赖测试。
- 首250追加步已保存，六臂global/Adam step5250，batch/增强/SNR/噪声指纹相同，代码snapshot未变。
- 378条旧系统与历史WeTok最佳图像引用在CPU复算，最大PSNR误差3.79e-6 dB；不是额外GPU指标复算。

## 入口

```bash
PYTHON=../backbone-eval-20260912/.venv/bin/python
$PYTHON -B scripts/train_bit_support.py --until-additional 5000 --execute
$PYTHON -B scripts/evaluate_bit_support.py --milestone 5000 --execute
$PYTHON -B scripts/analyze_bit_support.py --milestone 5000 --execute
```

训练中断且确认进程已退出后，才用相同配置加--resume。evaluate支持--resume及新attempt的--output-dir；analysis支持--evaluation-dir/--output-dir。不要覆盖冻结结果来消除工程错误。
当前训练/finisher初始PID为2473858/2483685，不能仅凭PID文件当作还活着；核实/proc、状态及日志。
输出位于VAR_COMM/outputs/WETOK-COMM-BIT-SUPPORT-20260912-*；六臂时延/质量及历史最佳对照均分列。

## 尚未完成

本轮仍运行中，未形成bit权重修订有效的结论。下一步须看完整校准、paired development及与原2000最佳/强系统的差距；不得因为源bits恢复更好而跳过图像指标，也不因负结果或达到5000步停止v2整个研究。
