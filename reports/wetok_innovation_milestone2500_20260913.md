# 接收创新量2500里程碑与当前推进

本机UTC+8 2026-09-13 07:28完成六臂各2500更新、完整校准和审计。**这是校准开发结论，不是新test或系统成功。**

## 结果与归因

1000图、五SNR与固定校准噪声；按预定完整校准LPIPS选模：

| 方法 | 选中step | LPIPS ↓ | PSNR ↑ |
|---|---:|---:|---:|
| single_pass | 1000 | 0.198920 | 21.220662 |
| multiscale_no_history | 0 | 0.200233 | 21.072752 |
| multiscale_state_history | 2500 | 0.201791 | 21.229134 |
| multiscale_prediction_features | 1000 | 0.201300 | 21.182018 |
| multiscale_innovation | 2500 | 0.200449 | 21.232548 |
| full_grid_innovation | 2500 | 0.196310 | 21.395159 |

多尺度创新量对同尺度state-history/prediction的LPIPS差为−0.001343/−0.000852，已有局部校准信号；但它仍不胜single，更弱于全网格迭代约0.004138。全网格对single约−0.002609，不能改称next-scale贡献，也尚未排除其中普通迭代与重编码特征的各自作用。

full-grid的0/1000/2500完整校准LPIPS为0.202972/0.199371/0.196310，走势支持完成原5000计划。所有臂从实际2500末端模型/Adam续训，不从选择的0或1000重启。

## 当前实际运行

- trainer **3205198**、review watcher **3205199**：六臂共同继续5000；07:44已保存2700。PID只作定位，须核查实际进程。
- 收尾监视器 **3211225**：等上述训练/审计完成且GPU可用后，自动进行27300主行＋600支持点评测、统计/CPU归档核验与完整校准绘图。不会启动新的训练分支，也不停止其它GPU任务。
- 当前总研究目标未完成，未有新holdout或训练种子重复。

## 新补齐的公平性与后续准备

1. no-history粗尺度仅用于训练辅助监督，推理时可删除，不影响最终图像。评测现在使用精简路径，并逐帧在计时外核对logits/符号/Decoder输入完全相同；CPU和真实GPU初值检查均通过。17项RX检查通过，训练/校准代码未改。
2. 父点图像阶段只覆盖15197张不同训练图，其余4803张只接受过表示监督。强Deep有30751643个通信参数、40000次ImageNet微调曝光与COCO历史；仍保留强参照，不称等参数/等训练历史。详见`wetok_training_sufficiency_audit_20260913.md`。
3. 同起点Joint/R-only控制已准备，7项CPU测试和真实GPU零更新检查通过；三Joint初始权重SHA与控制相同。实际图像损失的E/R梯度非零，未更新视觉模型或任何参数。初始波形最大末位差8.34e-7，在既有1e-6回放容差内，同y图像及no-history裁剪差均为0。
4. 新Joint训练**尚未启动**，须先完成当前5000控制及结果检查。后续仍需保留full-grid强参照；若Joint条件有增量，还需对应Joint普通全网格控制才能谈尺度增量。

## 资料

- 完整校准：`../outputs/WETOK-INNOVATION-R1-ANALYSIS/calibration_0002500/`。
- PNG/PDF曲线：`../outputs/WETOK-INNOVATION-R1-ANALYSIS/calibration_plots_0002500/`。
- Joint真实梯度/费用监控：`../outputs/WETOK-JOINT-SENDER-R1-PROFILE/profile.json`。三臂前/反向主体约0.321 GPU小时/1000更新，不含Adam、装载、校准。
- 分项梯度只在固定训练batch监控；其中MSE原始梯度范数高于当前加权LPIPS，但不由单批次或loss标量推断全部训练阶段谁主导，也未据此临时改权重。
