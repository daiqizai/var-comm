# 固定发送器接收创新量：当前接续

## 已完成并冻结（2026-09-14）

5000更新、完整校准、27300主行＋600支持点评测及CPU分析均已完成，旧进程25991/25992/25993已退出。不按下方历史状态重启或重复评测。结论`reports/wetok_innovation_result_20260914.md`；当前真正接续为`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`。原多尺度创新量不再追加该候选，full-grid仍是强参照，研究目标未完成。

总研究目标未完成。前geometry实现包确实改善同N/E质量，但新geometry的条件历史在主区间损害LPIPS/DINO，仍未胜强系统。

## 当前实际任务

- 2026-09-14 11:10已保存恢复后的4100更新，GPU任务与两个监视进程均存活。不要把旧7:58状态当成仍在运行，也不要从0/2500重复训练。

- 2026-09-14已确认上次流程中断，旧3205198/3205199/3211225不存在；本机10:10:44重新启动。已从经审计的实际3000断点恢复，trainer **25991**，review watcher **25992**，finisher **25993**。须查实际/proc，不凭PID文字认定存活。恢复依据`docs/recovery_20260914.md`。
- 状态`VAR_COMM/outputs/WETOK-INNOVATION-R1-TRAINING/status.json`；pipeline `VAR_COMM/outputs/WETOK-INNOVATION-R1-step0005000.pipeline.json`。
- 当前日志在`outputs/WETOK-INNOVATION-R1-RECOVERY-20260914/{trainer,review_watcher,finisher}.log`；旧原名日志只到中断前，已保留。2500全校准/审计PASS，实际3000模型/Adam/数据噪声接续到5000，不从选中checkpoint重启。
- 收尾进程 **25993** 在等待恢复后的训练与审计，之后自动执行已登记的RX评测/分析/绘图。状态`WETOK-INNOVATION-R1-step0005000-finish.pipeline.json`，子阶段日志`WETOK-INNOVATION-R1-step0005000-finish.stages.log`。**不要另起重复评测或在其前启动Joint训练**；遇其它GPU工作会等待，不停止它们。脚本`finish_registered_trial.py`不会自动开始下一训练分支。
- 六臂共享冻结204×30 single发送器数值及同一个实际y，每batch实际发送/噪声只计算一次。只训练R和新融合模块，视觉与E参数冻结；E(mu)的输入梯度保留。
- 新额外参数仅6417；基础R可训练1898080，带特征/门控者1904497。full-grid与multiscale创新量参数相同，但full-grid查询计算更多，不能称等FLOPs。

## 已验证

CPU三项模型检查与真实GPU profile通过：同波形、parent/零投影输出一致、E参数无梯度而输入梯度保留、train/eval一致。profile临时非零投影检查后恢复原初值，optimizer更新0。每1000更新RX主体估计约0.635 GPU小时，校准另计，不是完成时刻保证。

一次CLI导入检查因profile.py遮蔽标准库失败，已改profile_receiver.py；无训练结果被覆盖。当前配置/训练/profile绑定源码已冻结，不边跑边改。

## 下一步

1. 核查实际进程。5000结束后watcher自动review.py，检查固定E、真实共享y hash、全部采样/Adam和完整校准选择；monitor不能选模。1000/2500均已通过，不重复执行已完成步骤。
2. 等已排队的5000评测/分析完成，依据实际图像质量和计算量决定后续，不因代理残差变化或校准最优就宣布成功。
3. 重点看innovation对prediction_features（排除只是多用E）、对state_history、以及对full_grid_innovation（排除只是普通迭代）。同时保留有力single/no-history续训控制。
4. 本实验最终开发评测/统计执行器已实现，17项CPU/合成整链检查通过；441条旧图和21个父点波形的只读检查通过。新接收器的真实GPU development评测仍未执行、已排队。no-history图像推理省去无用4/8辅助读取，逐帧核对原输出不变；条件读仍保留。所有额外E计算计时，保留geometry强父点/数字/Deep和固定支持点。
5. 仍不访问新holdout；最终目标还需要合理强对照、训练重复、鲁棒性与独立验证。

入口scripts/train.py、profile_receiver.py、review.py、watch.py。用既有私有venv，不改全局依赖、不动旧冻结结果、不租新设备、不停他人任务、不提交/push。

## 已准备的后续执行器

- `scripts/evaluate.py --step 5000 --execute`：仅在对应训练结束、全校准审计通过、GPU无其它计算后执行；总27300主行与600个Deep固定支持点。完整校准选中步骤可以早于可用更新预算，必须照实披露。若在较早注册里程碑停止候选，可用对应step，不按development改选模。
- `scripts/analyze.py --step 5000 --execute`：核对全部文件、实际保存的s/y/AWGN、RGB/PSNR、连续特征与诊断BER，生成1968主配对区间和108支持点区间，训练/时延成本、native/noiseless诊断与结果摘要。此CPU复核不是新训练重复或新holdout。
- `scripts/plot_calibration.py --step 1000 --execute`：审计后画训练各项与完整校准曲线；监控子集单独画、不跨人口当作改善或用于选模。
- 预检证据`docs/reference_preflight.json`：441条旧参考PSNR最大误差3.93e-6 dB，21个父点信号平均复能量误差最大4.77e-7。不是新机制的性能结果。
- 本机05:35观察到86–87°C及间歇软件温控降频；未改时钟、功率或风扇，不提高GPU训练负载。新评测会按每帧轮换六臂测量顺序、保存每源前后硬件状态，不把带节流的时延说成硬件性能极限。准备记录`docs/evaluation_preparation.md`。

## 首1000实际结果

完整校准选中LPIPS：single .198920；no-history .200233（选中0，实际末端已训练1000）；state .201941；prediction .201300；innovation .202062；full-grid .199371。创新量此刻未优于预测特征/state/single/full-grid，不把较差初始化的修复当成功。

1000的完整/监控曲线PNG/PDF与loss/梯度/速度图已生成并查看，`outputs/WETOK-INNOVATION-R1-ANALYSIS/calibration_plots_0001000/`，已完成receipt，不重复运行同一步绘图覆盖结果。报告`reports/wetok_innovation_milestone1000_20260913.md`。下一段预计约1.35–1.45小时；仍无新holdout/训练重复/最终强系统确认。

## 2500与后续准备

2500选中LPIPS：single .198920(step1000)、no-history .200233(step0)、state .201791、prediction .201300(step1000)、innovation .200449、full-grid .196310。多尺度创新量优于同尺度两个对照，但仍弱于single和全网格；不能据此声称next-scale成功。报告`reports/wetok_innovation_milestone2500_20260913.md`及2500校准曲线均已生成。

父点训练曝光审计在`outputs/WETOK-TRAINING-SUFFICIENCY-20260913/`：图像阶段20k曝光仅覆盖15197个不同源，其余4803源只经过表示监督。为隔离E/R联合优化，`experiments/wetok-joint-sender-r1/`已准备同起点控制，7项CPU及真实GPU无更新检查PASS，图像梯度确实到达E/R；新训练未开始。该三基础结构用于训练充分性，不足以独自排除普通迭代；保留full-grid强参照，必要时补Joint对应迭代控制。

Joint后续完整评测已准备并通过16项CPU测试：九模型新测、全部十六方法、33600主行＋600支持点，不能只比较弱R-only；详见该目录`docs/evaluation_preparation_20260914.md`。真实reference资格要等当前原RX5000评测/分析结束后才能生成，不要跳过finisher抢先训练。

**冻结边界**：当前训练/profile、Joint初始化/profile，以及收尾监视器捕获的RX评测/分析/推理代码与`evaluation_protocol.md`均已绑定SHA。不可边跑边改这些文件或更改SHA放行。报告/接续文档可更新；若确有新代码缺陷，保留原证据并明确另立修订。
