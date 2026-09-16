# R2实际7500里程碑：继续同预算充分性训练

2026-09-14（本机UTC+8）。四臂5000→7500训练、完整1000源图×五SNR校准于21:35结束，原reviewer于21:35:41完成CPU审计；本报告不是新的development或正式holdout结果。

## 核验与结果

- 四个实际7500模型与Adam均完整保存；三个111参数状态和创新量117参数状态的Adam step均为7500，global data step14500。实际`resume.pt`与封存7500端点SHA一致。
- 继承的5000模型/Adam、已有校准、源数据顺序、增强、SNR、噪声、冻结视觉模型及训练源码不变；完整回放新2500批次指纹审计PASS。未重置Adam、未重跑起点校准、未修改旧结果。
- 无类别/反馈/测试真历史；原生Fq输入与continuous_mean接收接口，3060 data+0 header complex uses，E6120，实噪声1/γ，仍为原20k训练/1k校准。

下表是同一完整校准人口和1/4/7/13/19 dB等权均值，**不能直接与development或主1/4/7 dB成绩混排**。

| 通信结构 | 5000 LPIPS ↓ | 7500 LPIPS ↓ | ΔLPIPS | 7500 PSNR ↑ |
|---|---:|---:|---:|---:|
| single | 0.197125 | 0.192001 | −0.005124 | 21.33326 |
| multiscale state | 0.194679 | 0.192695 | −0.001984 | 21.33615 |
| ordinary full-grid state | 0.194968 | 0.193974 | −0.000994 | 21.34563 |
| full-grid innovation | 0.184720 | 0.176768 | −0.007951 | 21.76040 |

四个LPIPS最优完整校准点目前均为7500；固定100图monitor未参与选模。R2运行至此记录5131.96秒≈1.426小时进程活动时间，含训练、校准和保存，不是GPU独占利用率积分。

## 决定及边界

四臂均改善，创新量仍有较明显改善，尚无充分收敛证据。按已登记方案从**实际7500末端模型和optimizer**共同续到10000，不从selected重新初始化，不只追加最优臂，不改变loss/lr/结构/通信预算。新数据global14500→17000。

本轮只检验匹配训练充分性；不能将更多训练的效果写成新机制。当前4/8/16是否提供独立优势仍待同机会development比较，普通全网格迭代不改名next-scale。系统级强对照、训练重复、鲁棒性与新的独立holdout尚未完成研究目标要求。

10000后的质量评测保留18个已封存旧方法，新增四R2模型；当前质量不记时延。独立15模型计时使用预先固定32图×五SNR×一个噪声，回放质量阶段真实波形与图像。共享仅用于显式准入的质量阶段，计时遇竞争等待，不干扰其它任务。

## 证据

- `outputs/WETOK-JOINT-SUFFICIENCY-R2-TRAINING/milestones/step_0007500.json`，SHA `229b51631a60aa12c8c188a646102ef20b0eae156937753c3a279ba636199d78`。
- `outputs/WETOK-JOINT-SUFFICIENCY-R2-ANALYSIS/calibration_0007500/completion.json`：实际模型/Adam/数据/功率/校准选择审计PASS。
- 同目录`curves.csv`保留全部完整校准历史，`summary.csv`保留选中模型逐SNR及聚合质量。
- `experiments/wetok-joint-sufficiency-r2/docs/evaluation_protocol.md`：最终10000比较和独立计时规则。
