# 接口对照执行接续

2026-09-12，UTC+8。本轮不是继续已停止的bit权重六臂。

## 已完成的工程步骤

- 19项CPU测试通过，包括原物理/原生位序/配对统计测试和新增hard/ST/continuous接口检查。
- 九臂真实WeTok Decoder无更新profile通过。相同权重下两个hard接口像素完全一致，train/eval实际输出一致；MSE与LPIPS梯度均到达通信E/D，视觉参数不接收梯度、不改变。
- 三结构各2895944训练参数；每图3060复信道、平均复能量2，continuous输入有界且不称native码字。profile新增optimizer更新0。
- profile实测每microbatch前反向约0.092–0.096秒；九臂×effective batch4，1000追加更新主体约0.944 GPU小时，5000主体约4.719 GPU小时，均不含校准/optimizer/I/O。完整起点/端点校准还需额外时间，首1000流程暂估约1–2小时，不能作为完成时刻保证。
- 16:13独立CPU逐行复核：新hard_identity/single_pass的1000图×5SNR起点校准，与原2000对应记录的PSNR/SSIM/LPIPS/BER/BCE/state误差全部为0；训练/profile绑定源码未变。回执`docs/interface_parent_replay_audit.json`。这不是新开发集性能结果。

## 当前真实进程

21:12新续训断点实核PASS：共同追加2700、global data4700、Adam2700，模型/optimizer存储独立、数据与噪声仍配对。回执`docs/interface_resume_to5000_audit.json`中的“新增development输出0”仅指该CPU审计本身，2500主开发评测已完成。

**20:57最新：trainer 2730582 / watcher 2733118，目标总追加5000。** 2500训练/校准/开发评测和统计均已完成；条件版在主区间LPIPS不胜无历史及强系统，负结果见项目reports/wetok_interface_milestone2500_2026-09-12.md。当前从实际2500模型/Adam继续原配置，不回退到selected checkpoint；旧PID均为历史。

18:24恢复检查通过：已共同保存追加1100步，global data step3100，Adam逐参数step1100；模型/Adam存储独立，数据/增强/SNR/噪声仍配对，source/profile绑定未变。回执`docs/interface_resumed_checkpoint_audit.json`，不是从selected=0重启训练。

**18:16已进入下一段续训：trainer 2614845，watcher 2616385，目标总追加2500。** 原1000任务与监督器已经正常退出；九臂完整校准/Adam/数据/选择审计PASS，三个continuous选追加1000，六个hard仍选合法追加0。继续从实际1000模型/Adam恢复，不回退到各自最佳checkpoint。新日志`interface_train_0002500.log`/`interface_watch_0002500.log`；pipeline `interface_0002500_pipeline.json`。决定见`docs/interface_continuation_2500.md`，下方16:06 PID为历史记录。

16:06已用独立session启动trainer **2518473**；校准监督器 **2518976**。启动日志为`interface_train_0001000_launch2.log`和`interface_watch_0001000_launch2.log`。原shell后台启动未留下存活进程、状态或更新，两个空launch1日志保留；不是模型/训练失败，也没有覆盖一次已执行训练。

16:09核查时处于**追加0的完整起点校准**，尚不能称已完成任何新增更新；首hard_identity/single_pass校准LPIPS0.287890、PSNR17.68266与原2000记录对应。必须以最新status和实际/proc为准，不凭此文档中的PID认定仍运行。

- 实时状态：`VAR_COMM/outputs/WETOK-COMM-INTERFACE-20260912-TRAINING/status.json`
- pipeline：`VAR_COMM/outputs/WETOK-COMM-V2-20260912-LOGS/interface_0001000_pipeline.json`
- profile：`VAR_COMM/outputs/WETOK-COMM-INTERFACE-20260912-PROFILE/profile.json`
- 首里程碑：`...-TRAINING/milestones/additional_0001000.json`
- 自动CPU校准审计/曲线：`...-ANALYSIS/calibration_0001000/`

## 下一步

最新下一步改为3000/3500/4000/4500监控、5000完整校准，见docs/interface_continuation_5000.md。2500的33600主行+600 Deep支持点已完成，27项CPU测试及独立3024区间复算通过；新holdout和训练重复仍未做。以下此前“GPU评测尚未执行”的条目保留为历史准备记录，以本节最新状态为准。

18:49更新：1500固定监控中continuous三结构的LPIPS为.260784/.267499/.261105，较1000同源分别回退+.003228/+.008029/+.005612。不要继续声称单调改善；保留既有最佳，仍按计划等2500完整校准，见`docs/interface_monitor1500_review.md`。

评测新增的CPU整链合成测试覆盖了实际写表、硬/连续图像分流、旧图不复制、源图级中断/续跑与完整manifest校验。并修正了续跑后首个新source的重新预热，增加source前后外来GPU检测和全部session已记录耗时统计。27项CPU测试通过；这些是工程证据，不是新的图像质量实验。未改当前训练绑定源码。

最新下一步是观察1500/2000固定监控及2500全1000图校准，保持九臂同机会。第一段1000全校准continuous LPIPS：single 0.259012、无历史0.261040、条件0.256377；它是校准结果，不能与development强对照混排，也未完成研究。

已补强最终评测的5/6dB Deep对照：固定NN支持5→4、6→7但物理噪声仍5/6，600行单列并计划GPU重放；旧16方法/主1/4/7不改。600张旧图的PSNR/资源检查及4次真实CPU模型重放通过，最大像素误差3.40e-6，未做600次GPU重放。`docs/deep_support_preflight.json`和`docs/deep_support_supplement.md`。

只读TX诊断发现线性skip秩4896、实际支付6120实坐标；16训练源上，当前continuous信号在channel_lift列空间外的平均能量比例约4.0e-5至5.6e-5。它不是整个非线性码的秩/容量界，也不证明“浪费20%带宽”，不能据此跳过充分训练或马上加功率控制。结果`WETOK-TX-SPAN-DIAGNOSTIC-20260912/summary.json`，当前不改packing。

17:09更新：共同500追加步的固定监控已完成，continuous三结构LPIPS约0.257，两个hard约0.301–0.335；但条件continuous对无历史均值差仅约-0.000944，尚无稳定结构增量结论。保持九臂继续完整1000检查点，不用monitor选模、不改当前配方。详见`docs/interface_monitor500_review.md`。

首1000全部结束后，监督器自动审计配对数据/噪声、fresh Adam步数、能量、完整校准选模，并生成完整calibration曲线；**这一步不访问development**。依据真实calibration走势决定同机会追加到2500/5000或另立修订，1000不是研发上限。

恢复命令（仅确认当前trainer已退出、已核对里程碑后）：

```bash
../backbone-eval-20260912/.venv/bin/python -u -B scripts/train_interfaces.py --until-additional 2500 --resume --execute
```

所有九臂必须共同恢复，不偷改单臂权重或预算；当前snapshot绑定的interface_study/model/native/training/objective等源码不可在运行中修改。需要修复时保留失败attempt，再登记新配置/脚本。

**最终development评测/分析入口现已实现，但尚未执行新模型GPU评测。** `scripts/evaluate_interfaces.py`直接decode各自receiver_features，不将continuous硬化；`scripts/analyze_interfaces.py`完成源图级统计及图表。16方法、33600无线行、63比较/3024区间的完整网格由CPU合成测试验证；441条真实旧对照的图像/资源/噪声/PSNR复核通过，最大PSNR误差约3.81e-6dB，回执`docs/interface_reference_preflight_v2.json`。旧图只读引用，不大批复制。

须先审查完整calibration再决定何时运行最终评测，不自动在每个监控点访问development。命令为：

```bash
../backbone-eval-20260912/.venv/bin/python -u -B scripts/evaluate_interfaces.py --milestone 1000 --execute
../backbone-eval-20260912/.venv/bin/python -u -B scripts/analyze_interfaces.py --milestone 1000 --execute
```

此处1000是命令参数示例，不是现在批准跳过完整校准、抢占正在训练的GPU。后续2500/5000应填实际已完成且审计通过的里程碑。完整校准review不能冒充最终development结果；本轮没有新测试集访问。GPU执行器仍需实际运行验证，不能把CPU合成数据当实验成绩。

16:44已审计真实共同100追加步：global data step2100、fresh Adam每参数step100、九臂model/Adam存储互不共享、完整数据/翻转/SNR/噪声轨迹相同，最大能量误差5.96e-7；所有训练/profile源码绑定未变化。回执`docs/interface_first_checkpoint_audit.json`。这只是配对训练工程证据，不是质量增益。

研究目标仍active，尚没有经强对照和独立验证成立的通信新方法。
