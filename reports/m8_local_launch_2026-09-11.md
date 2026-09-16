# 固定m8 A/B本机运行：2026-09-11

## 授权与范围

用户明确说“我本机的显卡空出来了，可以开始了”。此次只启动已准备的固定m8配对实验，不启动全尺度、功率分配、parallel重训或旧工作区的暂停队列。
代码/输出在`/workspace/projects/var-next-scale-comm`；原模型、数据和历史结果只读复用，没有租卡、下载权重或访问新正式test。

## 冻结设置

- 本地配置：`configs/m8_local_20260911.local.yaml`，SHA256 `d34d8dede474d7a72604a9acbba1d53f8f8c782f7b82356fea5aa6d40f653725`。
- 各新增10000更新，有效batch4、microbatch2、fp32、AdamW LR 3e-5；同epoch2模型及step10000的Adam状态、相同数据/翻转/SNR/噪声。
- A：MSE + 0.01 LPIPS + 0.001 CE。B前2000：CE + 0.1 state；后8000：MSE + 0.01 LPIPS + 0.01 CE + 0.01 state。
- 固定255个前八尺度token、1508750通信参数、68 header + 2992 data = 3060 complex uses，平均复符号能量2；原AWGN约定不变。
- 冻结VQ/VAR/图像Decoder；自身恢复历史；state目标仅为实际发送真值token的逐尺度累计状态，硬选择/ST不变。
- 每500步固定100张校准图监控；0/2000/4000/6000/8000/10000步骤完整1000图选模。LPIPS为主，逐SNR PSNR相对共同起点退化不得超过0.2 dB；DINO只评测。

## 无更新GPU实测：通过

RTX 4090 D；启动时无其他compute进程。profile于2026-09-11 10:24完成（UTC+8），optimizer更新0次。
原hard图像/原loss最大差0，原loss梯度最大比较误差约3.49e-10；官方累计state目标差0，state对通信E/D梯度均非零。
prefix-only仅8个VAR尺度前向，图像Decoder调用0次；所有模型和optimizer状态未变。

| 阶段 | 有效batch4中位耗时 | 峰值tensor显存 |
|---|---:|---:|
| A原配方 | 0.398 s | 5.66 GiB |
| B prefix | 0.228 s | 2.24 GiB |
| B joint | 0.400 s | 5.66 GiB |

固定100图、5 SNR校准实测22.22秒。由此估算两分支训练2.12小时、全部计划校准0.86小时，合计约2.98 GPU小时。
这不是结束时刻保证：不包含optimizer.step、梯度监控、checkpoint/I/O、启动及最终development评测；实际成本以session journal为准。
原始证据：`outputs/VAR-M8-LOCAL-20260911-PROFILE/profile.json`。

## 启动与观察

2026-09-11 10:26:39（UTC+8）启动独立后台流水线，supervisor初始PID 1113754；训练初始PID 1113791。关闭本次终端不会因终端挂断结束任务。
顺序为train → evaluate → analyze，前一阶段成功退出且completion状态匹配才推进；异常立即停下，不自动改参数、放宽阈值或重试。
启动时先进行1000图、5 SNR的完整step0校准，此阶段不算新增训练；通过严格旧起点复现后才执行optimizer更新。
两分支初始模型SHA完全相同，Adam状态SHA完全相同，训练参数均为1508750；正式训练入口再次通过GPU工程检查。

**已确认进入真实更新**：完整step0校准的MSE/LPIPS相对旧结果差均为0；10:30:38（UTC+8）进入训练，10:33:23保存双方各250新增更新的共同断点。
两分支Adam step均10250，各曝光1000张训练图；首250个batch指纹全部配对，loss/梯度有限，teacher=0，B阶段无RGB loss，功率最大误差4.77e-7，冻结源码snapshot一致。
审计收据`outputs/VAR-M8-LOCAL-20260911-PIPELINE/first_update_audit.json`。这些是启动/执行正确性证据，不是训练方案有效性的最终实验结论。

实际状态以以下文件为准（相对代码仓库）：
- `outputs/VAR-M8-LOCAL-20260911-PIPELINE/status.json`：流水线阶段与进程。
- `outputs/VAR-M8-LOCAL-20260911-PIPELINE/train.log`：训练/校准日志；同目录保存evaluate/analyze日志，不污染冻结训练receipt。
- `outputs/VAR-M8-LOCAL-20260911-TRAINING/status.json`：最近保存的共同更新数或当前校准阶段。
- `outputs/VAR-M8-LOCAL-20260911-TRAINING/{continuation,two_stage}/training.csv`：每250个共同更新持久化。
- `outputs/VAR-M8-LOCAL-20260911-TRAINING/resume.pt`：双方共同断点；不使用单支未配对的临时状态。

完成后生成`outputs/VAR-M8-LOCAL-20260911-ANALYSIS/report.md`及代码仓库`reports/prefix_refinement_remote_result.md`。未生成完成receipt前，不宣称新训练有效或完成。
主要因果比较仍为B vs A，其次固定数字m8、数字自适应、感知DeepJSCC；按源图聚合噪声后做配对统计。

## 中断恢复

确认旧进程确实结束后，在代码仓库根目录用同配置和原asset绑定执行`remote_refinement.py train --resume --execute`；评测阶段则使用`evaluate --resume --execute`。
不能在训练中更改绑定的源码、协议、配置或SOURCE_MANIFEST。流水线已有训练completion时，不重新启动整个管理器，应直接续接未完成的后续阶段。
如用户要求停止，可向supervisor发送SIGTERM；它仅终止自己创建的子进程组，不停止其他用户任务。恢复时从最近共同断点重做未保存部分。
