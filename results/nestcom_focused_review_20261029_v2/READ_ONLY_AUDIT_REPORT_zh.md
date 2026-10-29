# Focused V2：正式帧能量、失败状态与实现配置只读审计

本审计仅读取已闭合正式记录及实际执行证据，没有运行模型、PHY、指标或 bootstrap；不改变旧实验结果。审计基准为 main / `93205a2e007ac587e28246cd399a531347a4f3f7`，现场核对该 HEAD 未变、已跟踪工作区干净。暂停支线 `nested_residual_rd_20261028_v1/STOP` 的 SHA-256 仍为 `452e3eb7391fe6bd55b430bea01c1a443deb40be9571c8141d52bebd67c53f6f`。完整环境见 `environment_and_scope_guard.json`。

## 1. 范围、关联与计数

能量和逐阶段失败审计覆盖：

- N1024：当前修复闭合版 common500 的 WHOLE、PARTIAL，3 个 SNR × 500 源 × 3 原信道重复，共 9,000 个逻辑结果。
- N512：当前冻结正式 common500 的 WHOLE、NEST、ADA，3 个 SNR × 500 源 × 3 原信道重复，共 13,500 个逻辑结果。

合计 22,500 个逻辑结果，对应跨方法去重后的 20,613 个物理记录路径。N1024 的 Ada、ARPC、BPG、Swin 和 Latent JSCC 不属于本次原始能量审计；18 行冻结策略表包含 N1024 Ada，并不表示审计过该方法的原始能量。

关联使用当前 closed delivery 的完整 source ID、原 source_index、noise_seed、实际端点、码长、physical key/path 和记录 SHA-256。N512 formal plan/RX plan、冻结策略及协议均核对其绑定；N1024 使用 repair_v1 的9000行交付和各自实际物理记录。每方法每 SNR 均为 500 个源、1,500 个逻辑帧。三重复的实际发送端点对每个源一致，既包括成功帧，也包括接收失败帧。

`formal_event_mapping_n512.csv` 与 `formal_event_mapping_n1024.csv` 保存逐逻辑关联；`formal_event_mapping_index.json` 保留原始行位置与拆分前后哈希；`source_identity.json` 保存原始输入 hash。物理去重按路径，不按相同结果或相同数值删除事件。`energy_summary.csv` 同时给出 logical 和 within-method physical_unique 两种权重；相同物理事件被两个方法复用时，仍各计入该方法的正式分母。全局20,613不能代替任一方法的统计分母。

## 2. 已有功率约定及实测记录

当前代码已有公开 scrambling：LDPC 编码与 RV0 rate matching/interleaving 后，对发送比特 XOR 由 protocol/session/frame counter 生成的 PCG64 掩码，再调制 QAM；接收端软解调后按同一掩码翻转 LLR 符号，再进行 LDPC 解码。不能将现有实现描述为“未扰码”。

功率约定是星座平均每复符号能量 Es=2，不做逐帧归一化。有限16/64-QAM码字允许实际能量波动。此审计读取正式 raw transmission 中记录的 E_frame、header/body energy 和 waveform hash，并逐记录验证 E_header+E_body=E_frame、E_header=136、rho=E_frame/(2N)、符号数=68+(N−68)。没有重生成波形或更改噪声。

所有 mean/SD/min/max 见 `energy_summary.csv`；SD 是总体标准差（ddof=0）。E_frame/N 的目标为2，rho的目标为1。N512、4 dB 的三种方法全部4,500逻辑帧均为QPSK，记录的 E_frame 严格等于1024，E_frame/N=2、rho=1、SD=0。

同源、同原重复的 PARTIAL（N512为NEST）对 WHOLE 的平均能量比，定义为 10 log10(mean(E_P)/mean(E_W))：

| N | 4 dB | 10 dB | 19 dB |
|---|---:|---:|---:|
| 1024 | 0.000000 dB | +0.001287 dB | +0.001023 dB |
| 512 | 0.000000 dB | −0.006272 dB | −0.001060 dB |

对应原始均值、逐对能量差分布、rho差值和 padding 关联见 `paired_energy.csv`。10/19 dB 的“逐对能量差 vs padding比例差” Pearson r 依次为 N1024 −0.01438/−0.02586，N512 +0.03319/+0.03351。QPSK能量恒定，其相关系数不定义，CSV留空。

这些是已评估正式帧的描述性审计，不是新置信区间或新的功率鲁棒性实验。平均能量差量级小且符号不一致；不能将此扩展成任意有限码字严格等能量、任意数据下无偏或已排除全部功率因素的结论。padding分母是LDPC信息容量k；源码利用率分母是k−29，两者不能混用。

## 3. 失败状态与完整分母

`failure_summary.csv` 分列 TX_NO_FIT、收到NULL、header reject、body CRC reject、body parser reject、source decoder reject 和成功payload解析。

- N1024 WHOLE/PARTIAL：各SNR各1,500帧的所有拒收/源解码失败计数为0。
- N512 NEST：4/10/19 dB的body CRC拒收为4/1/0。
- N512 WHOLE：7/0/0。
- N512 ADA：5/0/0。
- N512全部正式记录的header/body-parser/source-decoder拒收、TX_NO_FIT和NULL计数为0。

N1024还逐一读取9000个实际源解码证据：5439条内联证据、3561条hash绑定的修复接收结果，均为 ARITHMETIC_SOURCE_DECODED，且没有用TX真值或TX CDF补接收机。见 `source_decoder_audit.json`、`n1024_source_status.csv`。N512源解码状态与正式frame_scores逐行核对。失败记录仍在原逻辑分母中；本审计不重新评分灰图。

## 4. N512、10 dB相同目标下的实际端点

冻结目标均为(7,0)，16-QAM、rate2/3，但实际装入结果不同。500源在三个重复上的端点一致。

| 方法 | 实际端点及源数 | fallback | 尺度内端点 | 平均token数 | 平均源码利用率 |
|---|---|---:|---:|---:|---:|
| WHOLE | (6,0):498；(7,0):2 | 498/500 | 0/500 | 91.256 | 79.2951% |
| NEST | (6,0):53；(6,16):378；(6,32):59；(6,48):8；(7,0):2 | 498/500 | 445/500 | 107.896 | 93.4625% |

因此两者均99.6%触发目标以下回退；NEST89.0%的源使用尺度内端点。相同冻结目标不等于相同实传前缀。见 `endpoint_distribution.csv`、`n512_10_endpoint_explanation.csv`。这些是TX实际端点，不能因接收失败把端点记为0。

## 5. 可复核配置

`frozen_policies_compact.csv` 给出N512与N1024、WHOLE/NEST/Ada的18个冻结目标和MCS。目标与实际端点分开保存。

`configuration_audit.json` 保留配置、源码定位及SHA：

- NeST：VAR-d16，词表4096，Cvae=32，10尺度边长1/2/3/4/5/6/8/10/13/16；flash/fused关闭，当前正式RX为class1000、greedy、noCFG。
- 正式模型身份来自已闭合 `actual_native_identity.json` 中实际加载state hash，列出VAE、VAR、适配decoder。此审计未重新读取或加载大权重。
- 当前NeST正式RX记录为Python3.10.12 / torch2.11.0+cu128 / NumPy2.2.6 / CUDA12.8 / cuDNN91900；FP32、matmul与cuDNN TF32关闭、deterministic开启，matmul precision highest。CPU只读审计环境单独记录，不冒称当前审计使用GPU。
- native logits送入float64 softmax构建CDF，CDF总量2^24、最小频数1；先分配floor(p*(2^24−4096))+1，再将全部剩余整数质量加给最大概率token。并列遵循NumPy argmax的首索引。不是largest-remainder算法。算术区间为32 bit，源解码不加入生成CFG。
- 适配decoder每批输入确为50%连续F、25%全量化Fq、25%插值 F_bTX+alpha(F−F_bTX)。F_bTX经原10尺度流程补全到最终latent；其前8尺度前缀与源类别标签进入训练缓存生成，当前正式NeST接收则使用class1000。不存在显式coarse-only训练分支。E2应解释为该固定适配接收机的干预，不能声称对直接粗尺度解码做了公平专门训练或排除全部decoder训练因素。

## 6. 已有计时尝试日志

已有NEST软件计时日志确实保存逐次实际码长检查。本审计没有新计时。只取固定16张开发图、每源3次measurement，排除3次warmup，各SNR48测量调用。不是common500、也不是64源。

| SNR | 平均尝试次数 | 中位数 | p95 | min/max |
|---|---:|---:|---:|---:|
| 4 | 4.7500 | 5.0 | 5.0 | 3/5 |
| 10 | 4.7500 | 5.0 | 5.0 | 3/5 |
| 19 | 4.3125 | 4.5 | 5.0 | 3/5 |

p95按排序位置0.95(n−1)线性插值。三个measurement重复的各源尝试端点列表一致。`timing_attempt_calls.csv` 保存原key、实际尝试端点、prior_scale_calls和已记录时间；`timing_attempt_summary.csv` 汇总。尝试次数是对实际独立终止码流的长度比较，不等于额外prior前向次数，也不能替代实测时间。不能由此宣称Quarter最优或NeST比未同协议计时的Ada更快。

## 7. 复现与验收

在唯一远端checkout使用frozen_ldpc_py311，输出必须是全新目录：

```text
python experiments/nestcom-focused-review-20261029/read_only_audit.py --output /new/read_only_output
python experiments/nestcom-focused-review-20261029/read_only_configuration_audit.py --output /new/configuration_output
```

两脚本只使用标准库读取既有JSON/CSV/源码和Git状态。原始input路径和hash见两份source_identity JSON；图像数组、权重、缓存和大运行日志未加入本交付。代码分析结论在配置JSON的行号/源码hash中可追溯。父任务的no-fit协议修订与定点验证在独立 `nofit/` 中，不与本审计0新增PHY的账目混为一体。

本子任务状态：done。全部统计来自现有正式记录，未恢复暂停支线、未新建调度、未commit或push；Git发布由主任务统一处理。


出版打包说明：原始只读审计清单为 12,389,213 字节，超过项目 10 MB 单文件上限，因而仅按预算拆分 CSV。22,500 行的所有字段不变，统计未重算。原审计验收回执保留为 `audit_completion_original.json`，本次文件布局及原始大清单 PFS 位置见 `publication_packaging.json`。
