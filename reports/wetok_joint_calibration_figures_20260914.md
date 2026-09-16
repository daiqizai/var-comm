# Joint/R-only配对学习曲线

本次只读取经过审计的2500/5000模型快照及其中保存的训练/校准摘要，没有重跑模型或重新选checkpoint。绘图入口为`scripts/plot_joint_calibration.py`；两个里程碑分别保存，不覆盖中间历史。

## 建议查看

最新5000目录：`outputs/WETOK-JOINT-SENDER-R1-CALIBRATION-FIGURES-step0005000/`，各图同时提供PNG和PDF。

- `full_mean_quality.png`：三个基础结构的Joint/R-only全校准均值、全部初始点和实际选中checkpoint；各面板纵轴范围不同且图中已注明。
- `full_quality_by_snr.png`：1000张完整校准图的逐SNR PSNR/LPIPS曲线。
- `monitor_quality_by_snr.png`：固定100张校准子集的监控；**不参与选模**，不与完整校准点拼接成一条曲线。
- `training_components.png`：总loss、MSE、LPIPS、bit/state监督、诊断BER、梯度范数和每步耗时，均为尾随100更新均值；耗时没有承诺独占GPU。
- `quality_points.csv`、`full_mean_points.csv`、`selected_points.csv`：图中原始点及选择记录，可直接制表。

中间2500目录：`outputs/WETOK-JOINT-SENDER-R1-CALIBRATION-FIGURES-step0002500/`。此时no-history确实仍选择0，不能用5000结果抹掉这段历史。

## 解释边界

这些图只对照三个匹配基础结构，**不是所有强系统的排名**。5000校准Joint state LPIPS为0.194679，但R-only full-grid innovation仍更好，为0.187311；完整development必须继续保留它及数字/Deep参照。

星号表示由LPIPS规则选择的同一checkpoint；PSNR面板上的星号不是另按PSNR选模。所有均值都是校准，不是独立测试；不能把拟合曲线、训练loss降低或某次监控波动当成机制成立或失败。

5000末端仍有改善不代表已充分收敛，也不自动证明下一段训练有效。各图/CSV及源码快照SHA在各自`completion.json`中，原模型和原始校准CSV未修改。
