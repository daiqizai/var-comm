# H initial_true200 CPU核心：待执行登记封存

本目录只提供纯CPU核心和使用假PHY的测试。没有CLI、调度器、GPU评分、真实译码、部署或任务启动。没有改动已冻结的H协议、h64模块和预算ledger。

## 输入与接口

- `load_source_assets`读取源检查点SHA已绑定的source200 NPZ中的`m6_bits`至`m9_bits`，以及原S1 NPZ中的`tokens`。不读取图像、像素或概率。检查source ID/索引、档案SHA、四个码长与独立往返标记；上游仍须核验stage completion及其完整绑定。
- `select_payload`接收一个已经冻结的whole候选。H-R只用raw；H-A逐个m比较原缓存码长，算术只有严格更短才使用，相等选raw。若装不下，target m逐级降至m6。最终选择实际公开catalogue中的m/mode/profile ID；不构造额外免费控制字段。
- `run_frame`注入已登记的CPU backend、header和原H ledger。该函数不创建ledger、不改配额、不选候选、不捕获失败后重试。只允许`initial_true200`，每次真实header/body回调沿原`receive_frame`计费。

候选格式沿预筛接口：`candidate_id, arm, snr_db, target_m, K=0, q, nominal_rate, selection_rank, slot`。整份shortlist须为`H_EXPECTED_PSNR_SHORTLIST_FROZEN`且`ready_for_real_calibration=true`，200个source ID次序固定，每臂/SNR最多两个whole候选。partial另走后续登记，不在此核心混入。

## 必须先封存的工程确定化

这些细节具体化原协议，并未被本模块当作已经获执行登记的事实：

1. 噪声身份为`[h64_catalog.PROTOCOL, "initial_true200", source_id, integer_snr, integer_seed]`。使用UTF-8、ASCII转义、紧凑逗号/冒号、禁止NaN的JSON；SHA256前16字节按little-endian整数初始化PCG64。
2. 生成一次float64、形状1024×2的标准高斯。所有臂/候选共享该源/SNR/seed的高斯，不把candidate加到种子。以float64做`wave + 10**(-SNR/20)*noise`，最后整帧转float32。没有逐帧功率归一化。
3. 公共frame counter为`((slot*200+source_index)*3+noise_seed_index)`，seed次序6101/6102/6103；预筛冻结的全局slot为0–15。这个公开计划有9600个互异counter，与源内容、payload长度、噪声实际值无关。scramble session固定`H_INITIAL_TRUE200_V1`，防止和资格/测表会话混用。
4. 事件ID含固定阶段、candidate ID、SNR、源序号、seed；同一事件重放只可复用原ledger的完整结果，不新增译码或退还费用。不同SNR的同candidate ID不会碰撞。
5. 接收器用完整、原ID不重排的已准入公共catalogue。真实错收但被接受的header决定实际RX布局；未知ID或失败由原接收规则拒绝。原图token、TX profile和TX算术概率均不进入`receive_frame`。

`ENGINEERING_CHOICES`是以上选择的机器可读形式。执行前由主控的新登记生成contract，要求：

| 字段 | 内容 |
|---|---|
| status | `H_PAYLOAD_CPU_ENGINEERING_SEALED` |
| execution_registration_sha256 | 新执行登记原文件SHA256 |
| core_source_sha256 | 本模块原文件SHA256 |
| protocol_canonical_sha256 | `h64_catalog.digest(完整H协议JSON)` |
| catalogue_canonical_sha256 | `digest(已准入完整catalogue JSON)` |
| shortlist_canonical_sha256 | `digest(冻结shortlist JSON)` |
| engineering_choices_sha256 | `digest(ENGINEERING_CHOICES)` |

这些canonical哈希不能替代执行登记中的原文件SHA。主控仍需绑定h64依赖源码、backend/header实现、source200和S1检查点、预算登记、预筛完成凭证与所有输入；封存前不运行真实帧。

上层driver必须实际读取并核验新execution registration文件SHA，不能只接受一个格式正确的SHA字符串；必须核对原预算registration文件SHA、既有ledger的budget binding、branch=H及完整phase_limits逐项相等，不能只检查initial_true200剩余额度或重新建一个等额ledger。新执行登记不替换既有ledger的预算绑定。上层还必须验证H资格完成凭证及其所有绑定，核对当前backend实现身份/版本/源码、decoder设置和实际layout与获准入资格一致；完整公共catalogue也须来自同一已验证资格。仅有catalogue的ADMITTED标签或core contract不能替代qualified backend核验。这些启动责任属于尚未实现的上层driver，本核心不会伪称已完成。

## CPU输出的边界

输出包含原始实际header/body outcome、CRC接受/解析状态、实际decoded bits和payload、收发波形SHA、噪声SHA、全帧能量、实际profile、所有TX回退尝试以及仅用于评测的正确性标记。它们是私有接收记录，不应原样作为公开载荷上传。

`logical_packet_events`及`packet_event_ids`只描述此轨迹引用的header/body逻辑事件。复用完整事件时仍引用原1或2个事件，但新增译码和新增计费均为0。真实累计费用与状态必须来自既有ledger，不能对这些轨迹字段求和当作译码次数；并行worker下也不能将一次全局counter差值归属于单个worker。测试单独核验首次调用的2次计费、复用时0次新增计费和0次新增body调用。

raw解析成功可直接给出实际收到的token。算术CRC和L/padding通过只标`ARITHMETIC_CANONICAL_RX_REQUIRED`，`gray=null`；后续独立RX必须用收到的m和实际payload调用冻结`h64_source.decode_prefix`。不能拿TX缓存token/CDF填充，不能把实际错收流替换为clean图，也不能提前用gray代理为真实图像评分。CPU阶段没有图像指标或系统成功结论。

最多16候选×200源×3噪声=9600轨迹；header全部尝试，body仅在header接受后译码，上界19200调用，仍属于既有20000的`initial_true200`分仓。整个header/body都是已付1024符号；未尝试body译码不退还信道资源。
