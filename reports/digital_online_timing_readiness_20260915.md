# 数字VAR在线计时补充：输入已核验，完整计时尚未执行

日期：2026-09-15，本机UTC+8。此项是收敛报告后的CPU只读补充，不是新架构、新训练、GPU推理、质量实验或新holdout验证。原收敛报告revision 2及全部旧结果不改写。暂停仍有效。

## 1. 实际完成的工作

执行`scripts/audit_digital_timing_readiness.py`，读取已有开发传输及R2独立计时记录，没有加载PyTorch或视觉模型。完成回执时间为2026-09-15 11:38:22 UTC+8。

- 沿用R2预先固定的32个源图索引，即`rint(linspace(0,99,32))`，没有按失败、质量或速度重选。
- 五个SNR为1/4/7/13/19 dB，固定原seed2001；核验m7/m8/m9/adaptive四个数字方案，共640条既有记录。
- 32张原图文件SHA、数字归档RGB对应的预处理像素SHA、波形/接收记录/图像档案SHA均通过。
- 由原发送波形和原噪声生成规则在CPU重建接收观测，640条`received_sha256`全部逐字节匹配。不是重新执行FEC、信道译码或模型推理。
- 每条数字波形均为3060×2的原float64 QPSK坐标，所有实分量绝对值为1，总能量6120；没有改资源或精度后再宣称回放相同。
- 与R2原15模型×32图×五SNR的2400条计时记录核对完整矩阵和源图身份；输入源码及相关文件共记录155个SHA。
- 检查存档输出合法性以及header失败灰图、CRC失败候选保留但不标为可信的规则。没有利用原图选择输出，也没有重新计算图像质量。

**核验边界：** R2这里只复核原SHA绑定的计时记录和源图ID，未重新推理或独立重建R2输入张量。数字RGB也是既有归档，不是本次重新渲染。不能把640条记录说成640次新传输，也不能把核验通过当成完整计时已完成。

## 2. 新查清的计时端点差异

| 现有入口/列 | 起点→终点 | 边界限制 |
|---|---|---|
| 数字旧`receiver_seconds` | CPU接收波形→CPU译码prefix | 不含VAR补全和图像Decoder，原报告已经注明 |
| 数字`complete_image` | CPU prefix→CPU RGB数组 | 包含生成和图像解码，也包含返回CPU；不含PHY译码 |
| R2 `receiver_seconds` | GPU接收波形→GPU RGB张量 | 包含图像Decoder，但H2D在计时前，D2H在计时后 |
| R2 `online_TX_seconds` | GPU视觉编码时间＋GPU通信编码时间 | 两段计时之和；中间`indices_to_features`及主机/设备搬运未计入，不是连续端到端时间 |

证据位置：

- `scripts/evaluate_progressive_channel.py:238`：PHY计时结束，图像生成在后续代码中。
- `src/var_comm/progressive.py:250`：图像返回显式包含`.cpu().numpy()`。
- `experiments/wetok-joint-sufficiency-r2/scripts/evaluate_timing.py:153`：接收波形已放入GPU。
- `experiments/wetok-joint-sufficiency-r2/scripts/evaluate_timing.py:173`：RX计时结束；第176行才把图像复制回CPU。
- `experiments/wetok-joint-sufficiency-r2/scripts/evaluate_timing.py:134`：两段TX计时之间执行表示转换；第184行相加报告。

因此，不能简单把数字`receive_whole + complete_image`包上计时器，再与R2旧39.40ms直接做严格快慢排名。后者是包括视觉Decoder的设备驻留接收计算，仍是有效的内部比较，但不等于CPU波形到CPU图像的完整系统时延。本次不估计这些遗漏操作有多大，也不通过减去估算值修补旧计时。

## 3. 所有失败记录保留

下表仅为固定32图×五SNR×一个噪声的计时输入子集，每方案160条。它不是原100图×三个噪声的主质量统计，不用于更新方法排名或SNR策略。

| 数字方案 | Header失败 | Body CRC失败 | 协议接受 |
|---|---:|---:|---:|
| m7 | 0 | 3 | 157 |
| m8 | 0 | 32 | 128 |
| m9 | 0 | 75 | 85 |
| adaptive | 0 | 14 | 146 |

“协议接受”不是原token绝对正确的证明。误接受计数字段仍在逐帧表中。当前子集没有header失败，不伪造测量样本补入；CPU回归测试单独覆盖header失败必须输出灰图以及禁止免费类别的分支。失败样本今后计时也不能丢弃。

## 4. 若获准补测，所需的是现有模型同端点计时

建议采用共同系统边界，而不是另建通信机制：

- **TX：** 已驻留CPU的共同裁剪RGB→CPU信道波形，一段连续计时；必要归一化、H2D、视觉编码/量化、表示转换、通信编码/FEC、输出搬运均在其中。
- **RX：** 已驻留CPU的原接收波形→CPU RGB，一段连续计时；计入必要搬运、PHY或通信网络、VAR补全及图像Decoder。使用各自原质量实验的数值精度，不擅自重算后换精度。
- 统一不计磁盘IO、模型加载、预热、AWGN构造及离线指标计算；批量1，受控空闲GPU，CUDA同步，全部失败仍计入。
- 保留原类别获取假设、68/2992数字header/data账本、3060/6120总资源、原采样/argmax及失败规则；不提供真实prefix，不重新选模。
- 正式测量必须重放实际在线TX，不能把源token缓存当免费编码；RX不能复用跨帧图像/latent/KV计算。缓存只在计时外用作一致性参照。
- 报告CPU端点系统时延；如另报设备驻留计算，独立标明，不混进同一列。R2至少冻结residual作学习参照，不能只补数字时间后与旧不匹配端点相减。

本次仅形成上述依据与输入表，**没有实现或排队新的GPU计时driver**，没有新的延时数值。这不是新的科学gate或架构搜索；本项CPU准备到此结束，继续推进应是获得执行范围确认后的实际同端点测量，不是继续扩写预检。

## 5. 交付与验证

- 只读核验脚本：`scripts/audit_digital_timing_readiness.py`。
- CPU回归检查：`scripts/check_digital_timing_readiness.py`，10项通过；覆盖矩阵缺项/重复、SHA篡改、波形/噪声/精度变更、合法token、失败保留和可信状态隔离。
- 产物根目录：`outputs/DIGITAL-ONLINE-TIMING-PREPARATION-20260915/`。
- `input_audit/archived_frame_inputs.csv`：640条可定位旧观测/输出的输入清单。
- `input_audit/timing_boundaries.csv`：四种现有计时入口的端点和源码证据。
- `input_audit/retained_failure_counts.csv`：固定子集的完整失败分层。
- `input_audit/completion.json`：状态`OFFLINE_INPUTS_VERIFIED_TIMING_UNMEASURED`、155个输入SHA及输出SHA。
- `cpu_checks_001.log`、`input_audit_001.log`：实际CPU执行日志。

复核不触及训练权重；数字VAR仍为候选主系统，R3仍是唯一保留的学习待验证候选，R2等冻结为对照。没有以本次工程核验宣称通信方法或总体研究已经完成。
