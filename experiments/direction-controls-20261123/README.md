# 研究机制对照与 5A/5B 审查

本目录包含三组配平训练对照、两项冻结权重干预和来源复核代码。HoloRate、SemSIC 和 5B 各训练两臂。研究结论、逐来源得分、均值和区间位于[研究结果目录](../../reports/directions_20261123/)。

## 已用版本与存放位置

`study.py` 内的 `SHA` 固定为 `ec608c7d89d78704b4efa7d32004fafec85f6766`，是 3B/4B 训练实现的来源。该提交位于 Direction 4 研究分支。脚本使用现有 `liulu` 运行环境的 Python 3.10、FP32 和关闭 TF32 的路径，完整权重、输入图像和外部模型按既有 PFS 布局读取。执行前已核对冻结的 Swin 权重。历史 GPU 注意力训练算子按原设置允许确定性警告，匹配种子不保证重新训练逐位复现。

| 内容 | PFS 位置 |
|---|---|
| 研究整理、统计和运行代码 | `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/analysis/var_comm_directions_20261123` |
| 运行日志、进程退出记录、生产版本快照 | `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/direction_controls_20261123_v1` |
| 六个训练臂的权重，按 holo、noma、dir5_train_control 分开 | `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/direction_controls_20261123_v1` |

## HoloRate 与 SemSIC 训练对照

1. `study.py` 从 Git 对象读取所需源码并按字节复制到独立的运行目录，应用显式记录的实验差异。运行后产出 `protocol.json` 和 `source_identity.json`。新输出以普通文件写入。
2. `qualify.py` 在 CPU 上检查 HoloRate 的装包、逆变换和整块擦除，检查 SemSIC 两个接收分支的 LMMSE 次数以及本用户去噪器和门控的有效梯度。
3. `train_pair.py holo` 和 `train_pair.py noma` 分别训练各自的两个配平模型。每臂共享本方向的新种子、优化器、样本顺序、训练目标和验证集选择规则。HoloRate 每臂 5,000 步，SemSIC 每臂 4,500 步。权重保存在上表的专用目录。
4. `after_training.py` 按实际退出记录接续 `evaluate_pairs.py`，在 common500 和 Kodak24 上逐臂复用同一噪声、衰落和图像组合，复核旧 raster/单级接收器的记录。原测试表缺逐帧值，这两份旧权重的重放用于获得新老模型的逐图配对。每项登记的 500 图和 Kodak 结果独立汇总。

## 5A 与 5B

`capture_dir5.py` 把生产版本的代码、轻量表格和日志存成普通文件，并计算检查点哈希。`seal_dir5.py` 在两组 v2 测试完成后捕获第二版。`audit_dir5.py` 验证公开的差值和均值相减一致，并保留缺少区间的字段。v1 与 v2 表分目录保存。v1 的两份 GenHolo 检查点已被生产流程更新，因此 v1 只能追溯到原评测表记载的哈希。v2 核对磁盘上的实际权重。每个版本身份文件也记录捕获时可见的 Git 提交和未归档输入的逐文件哈希。

`dir5_channel_check.py` 验证酉变换、单音多普勒泄漏和符号预算。`dir5_waveform_check.py register` 登记固定的 5A 开关对照，运行时为每份权重、每种波形只生成一次相同信道观测，供两种可靠度处理使用。`dir5_receiver_check.py register` 登记 5B 的 1/256、10 dB 开发集干预，保持同一有噪观测、码本、解码器与后处理权重，测量在推理时移去校正项的变化。两项干预使用原生产验证噪声，检查未干预的结果能否复现原验证分数。

5B v2 正式结果和冻结接收干预出来后，另行登记了 1/256 的训练对照。运行 `dir5_training_pair.py prepare` 固定源文件与实际潜变量缓存的哈希。`dir5_training_pair.py run` 在同一张 A800 上顺序核对缓存、训练两臂并执行 `dir5_training_evaluation.py`。两臂均使用相同的 8×8 源表示、冻结 VQ 训练目标、后处理、初始化、6000＋1500 步和选点规则。对照臂从第 1 步起把码本修正的两个权重置零。测试沿用 500 张图和 Kodak24 的三个原信道抽样，固定 10 dB、8 个块全部到达。原生训练模型还在同一个接收观测上关闭修正一次，以分开读出训练作用和推理作用。

## 阅读与复核

`audit_history.py` 比较原六分支指定结果 CSV 的 Git 对象与 PFS 字节。`finish_checks.py` 从新结果的逐帧、逐角色记录重新计算全部来源均值和 5,000 次直接配对重采样区间。`results_digest.py` 用完成后的 CSV 生成机制结果表。`prepare_release.py` 向主线的验证视图复制轻量结果，记录每份交付文件和留在 PFS 的四份逐帧原表的 SHA-256。

`dir5_training_audit.py` 从额外的 5B 逐次输出重算 36 个均值和配对区间，并把实际设备用时合入总账。各项用时与退出状态以 PFS 中的进程记录为准。若要在另一台机器独立复现，可将 `study.py` 的研究、临时和权重根目录指向新的普通目录，并准备相同来源的源码、权重、划分与指标实现。冻结干预还依赖生产快照中的原始函数。
