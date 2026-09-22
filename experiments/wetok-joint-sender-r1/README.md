# Joint E/R与冻结E的同起点控制

**已于2026-09-14 12:39激活训练，首1000里程碑正在运行。** 原RX-only已完成。先读`docs/CONTINUE.md`核查实际PID与状态，不启动重复队列。

问题/设计见`docs/protocol.md`，动机见`../../reports/wetok_training_sufficiency_audit_20260913.md`。保持现有204×30结构、loss、N/E及数据，仅让通信E也接受图像梯度；不把这一训练因素当新的论文机制。

已实现：

- `src/joint_sender/model.py`、`runtime.py`：同权重初始化，精确微批次梯度分解；图像梯度经R、信道回到E。
- `scripts/train.py`：阶段保存/恢复、完整校准选模；启动须已有R-only完整5000控制和真实无更新GPU profile。
- `scripts/review.py`、`watch.py`：实际E/R/Adam/数据噪声核验，比较相同新增更新机会下的校准选择，不拿Joint1000对控制5000选中点。
- `scripts/check_initialization.py`：真实父点的CPU权重核验已PASS；三结构初值SHA都与旧控制相同。Joint可训练参数2927358，R-only为1898080。
- 7项CPU测试PASS，包括frozen模式与旧R-only数学结果一致、图像到E/R梯度、整batch/微批次梯度等价，以及实际训练器中断续跑与连续更新权重逐位相同。
- 07:32真实GPU无更新profile PASS：图像-only损失的E/R梯度非零；同y的初值图像和no-history裁剪差为0；初始波形末位差8.34e-7低于既有1e-6容差。权重未变，optimizer更新0；分项梯度与费用在`../../outputs/WETOK-JOINT-SENDER-R1-PROFILE/profile.json`。
- 2026-09-14评测/统计执行器已补齐：Joint三结构与全部六个冻结接收器同机新测，并保留七项系统参考，共16方法/33600主行＋600支持点；2016配对区间、144结构交互、162支持点区间。16项CPU测试通过，涵盖实际评测器的中断续跑、不同Joint波形、全部强接收对照和旧图指针复用；不是已完成新模型GPU评测。

尚待：

- profile已完成且绑定源码，不重复运行覆盖。三臂前/反向主体约0.321 GPU小时/1000更新，Adam、装载与校准另计，不是总工期保证。
- 原R-only5000及其完整评测、分析、reference资格均已PASS，Joint已按`docs/activation_20260914.md`从相同原父点开始。后续仍先完成自身完整校准审计再评测。
- 待原RX全流程完成后运行`scripts/qualify_references.py --execute`，封存实际5000控制及其评测/分析；目前不能用不存在的终点占位。新Joint自身训练、全校准审计完成后才运行`scripts/evaluate.py --step 5000 --execute`及`scripts/analyze.py --step 5000 --execute`。新holdout和训练重复均未执行。

原RX队列25991/25992/25993已正常结束，不再重启。当前Joint trainer/review为188608/188609；最终比较不能忽略更强的full-grid参照。

FP32梯度标记影响矩阵乘法路径的已定位末位差见`docs/numerical_gradient_note.md`。没有改旧Encoder/接收器、冻结数据或校验值。
