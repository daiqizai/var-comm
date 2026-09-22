# 视觉骨干选型预注册（本机日期2026-09-12，UTC+8）

## 0. 授权、范围与资源

最后用户消息覆盖远端要求：先等本机已有任务自然结束，再启动本轮。旧2×2训练及自动校准/评测不被打断，不在阶段切换空隙抢GPU；之后不再排新原版VAR训练。
本轮只选型，无信道、无优化器、无训练或模块搜索。候选只有XQ-GAN配套MSVR10P2-4096+VAR-d17和WeTok指定完整codec；不接入FlowAR/FlexVAR/新通信E/D。
任务命名含liulu全拼、小写/数字/中划线；不创建或改变公司开发机、PFS、BOS、队列。原远端认证前连接超时，无远端文件或进程改动。所有执行已改到本人本机隔离目录。
单GPU、batch1、2 CPU线程、nice10、下载单流≤8MiB/s/低I/O优先级；既有权重不复制。候选原始训练checkpoint下载合计约12.84GB（含上游optimizer，但不实例化optimizer），不是下载全数据集。
监视器无CUDA上下文，最多等待24小时；连续3次/60秒GPU空闲（无compute PID、util≤5%、memory≤512MiB），旧完整pipeline终态+PID身份退出后才启动。评估最多4小时。
有外来GPU任务或探针异常，只停止自己验证过的进程组；不重置GPU，不停止旧/他人进程，不自动重试模型实验。无调度器条件下不能原子预留GPU，明示短时竞争残余风险。

## 1. 数据与统一协议

固定旧100张ImageNet development，永久不称为新test。清单：
`/home/liulu/projects/VAR-MAP-GATE0/results/vae_reconstruction/imagenet100_m6_to_m10_rate_sweep/selected_samples.json`
SHA256 `33d2a4f13eb97bb1d47fca25df04d9ca4b7ebf0fdcefe7d9497c5a6e4830c243`。
每个输入文件与预处理tensor记录hash，所有模型同一顺序、同一RGB张量。
沿用历史 `var_decoder_tuning.preprocess_path`：短边256、尺寸round、PIL bicubic/antialias、中心裁剪256，映射[-1,1]。不替换成候选论文自用resize/center-crop协议。
使用历史 `evaluate_var_decoder_only.calculate_metrics`：PSNR、pytorch_msssim SSIM、LPIPS-Alex、DINOv2 ViT-S/14 cosine；保留准确float输出计算，PNG仅供视觉展示。DINO评测尺度/归一化原封复用，不把两个不同DINO版本混排。
FP32、不用autocast、不启用TF32；冻结eval模式。新全重建与旧100图已存结果做协议一致性检查，不凭论文数值认定改善。

## 2. 模型、来源与表示账本

| 臂 | 用途 | 全表示rawbits | class |
|---|---|---:|---|
| old-official | 原VQ+官方decoder，完整及m8/m9参照 | 8160 | 直接解码无需class |
| old-fidelity | 同原tokenizer、已有fidelity decoder；只完整 | 8160 | 不需要 |
| xq | **生成checkpoint中实际配套的tokenizer**，完整及prefix/completion | 静态配置6864，必须输出索引复核 | 仅completion提供同等真实类 |
| wetok | ImageNet/downsample16四码本完整codec，EMA推理 | 静态8192，必须索引复核 | 不需要 |

XQ官方源码：`lxa9867/ImageFolder` commit `137869c6e60b1c48edc2a33f543a28b565a10632`。
公开配对：`qiuk6/XQ-GAN/MSVR10P2-4096/best_ckpt.pt`及`VAR-d17-MSVR10P2-4096/ar-ckpt-last.pth`；HF revision `67d51a379d592d228fe96ddd587aa84682ac53a0`。
README/config/API快照与Git blob核验在本目录。HF官方站直连不稳定，记录使用hf-mirror解析并转向官方HF CAS的传输，按公布LFS SHA256验证，不把mirror域称作者。
配置明确 `[1,1,2,3,3,4,5,6,8,11]`、PQ=2、每分支4096码、12bit/index；空间合计286，索引572，而不是286个index、也不是以8192联合logit维度计13bit。
原VAR尺度 `[1,2,3,4,5,6,8,10,13,16]`，m8/m9/full分别255/424/680索引及3060/5088/8160bits。
XQ真实自然档位：第8尺度后101位置×2分支×12=2424；第9尺度后165×2×12=3960；全部286×2×12=6864。
这轮**只用XQ第8/9自然prefix**和full点；5088附近不存在相同自然档位，展示离散率质点，不裁剪坐标、不改名m8强行等bits、不进行插值“等资源获胜”声明。

XQ官方inference.py读取 `generator['trainer']['vae_local']` 与 `var_wo_ddp`。必须完整strict加载，核对d17 head(8192,1088)、位置长度286及两个4096码本。
同时把独立tokenizer checkpoint model/ema与嵌入的配套tokenizer逐张量比较；若不相同，显著披露并始终使用**实际配套版本**，不能把不匹配的standalone模型接入生成器或谎称权重相同。
构造时抑制冗余pretrained下载并立即strict恢复全部checkpoint张量；未运行的semantic loss constructor仅临时取rank0/world1，不创建DDP。覆盖范围写入metadata，不改变前向网络/数值结构。

WeTok官方源码 `zhuangshaobin/WeTok` commit `caa2ad7e709cdabe8432bead448f7514def13919`；配置 `configs/WeToK/Inference/ImageNet_downsample16_imagenet.yaml`。
HF `GrayShine/WeTok` revision `85fc6eb084d458b8d4fa3a32d541379e95b2bf87`，`ImageNet/downsample16/WeTok.ckpt`。
配置n_embed256/num_codebooks4，16×16×4×8=8192bits。24.50dB是论文协议报告，不预设同一100图的提升，不接LlamaGen/通信。

## 3. Completion不可泄漏

接收器API只收到真实**截断后的prefix**（XQ两个分支各截到同一自然尺度），不得收到源图/未传true suffix。生成suffix按自身历史闭环，不能teacher-force后续真值。
直接prefix用零**残差**在完整11×11潜变量上解码；不把未传index设0（那是非零码字），残差层选择仍使用原训练10尺度分母。
所有forced prefix逐index不变，保存输入hash/实际使用hash/输出token hash；full-token complete=direct作为工程检查。
主比较：旧与XQ都采用**真实类条件、无CFG混合、确定性argmax**。这与历史已用源图补全口径一致。
固定辅助敏感性：仅XQ作者sampling设置cfg=3.25、stage-ramp `t=3.25*stage/9`、topk750、topp0.95、3预定种子0/1/2；全报告、不best-of、不按development选采样配置。
注意“主argmax cfg1”是无CFG混合，不是作者的额外ramp参数1。两种约定在每行中明示。
类标签沿用双方相同pre-shared/oracle类别的无信道假设，条件补全另报如果发送1000类固定长度需10bit；raw表示不含该10bit。不得给新骨干独享免费类标签优势。
这里无FEC/CRC/header/信道/能量实验，不能称3060 complex uses正式等资源结果；后续通信贡献仍必须在同一骨干内建立数字/学习配对。

## 4. 有界执行与计时

每模型先首2张smoke，通过索引/strict配对/输出/前缀守恒检查后才做完整100张；smoke不是科学样本，不加入均值。模型分独立进程，避免两个源码的 `models/dist/datasets` 命名冲突。
每图编码、直接解码、生成+解码分别CUDA同步计时，batch1，剔除文件I/O与指标网络；peak allocated/reserved及常驻基线分开记录。模型权重载入时间另报，不混入稳态codec时延。
先保存全精度输出，再释放codec模型、加载统一指标网络，因此模型峰值不包含LPIPS/DINO权重。算法中接口校验/前缀验证成本如计入时延则在字段说明中明示。无不同硬件旧时延混排。
所有代码/配置/来源/资产SHA在ready receipt锁定；工程异常停止并保留日志，不静默换checkpoint、放宽shape/assert、加载部分权重或改采样参数继续。

## 5. 输出与迁移判据

100图逐图四指标、原始bit、sidebits、耗时/显存、mean/median/尾部、逐图胜率、按图bootstrap 95%区间、离散rate-quality图。采样先按图汇总各seed，不能把300条当300张独立图。
完整重建对官方及fidelity双参照，prefix/completion同时对直接prefix、同候选full和旧3060/5088工作点。选代表性困难样例和细节对照，不只报均值；人工观察只能解释，不用于逐图oracle输出选择。
支持迁移：在相近实际源表示预算下稳定改善PSNR且LPIPS无明显恶化，或相近源图配对质量下实际表示更小，且收益在真正使用的prefix+completion点仍存在。没有额外武断5%门槛。
若只有full改善，列为完整codec改善而非生成式通信迁移通过；若只有FID/DINO好、PSNR/LPIPS没改善，不建议迁移。
XQ使用DINOv2骨干/语义引导，DINO不是独立语义证据，不单独决定迁移。WeTok不承担主线接口迁移。
本脚本只产出结果与描述性证据，**不自动改主骨干、不训练新通信E/D、不把旧通信checkpoint直接复用于新表示**。

## 6. 实际checkpoint兼容性补记（GPU评估之前）

XQ公开独立tokenizer包含ruamel.yaml配置对象。torch2.11内建weights_only即使列入白名单，也因CommentedSeq APPENDS不兼容而失败；这是CPU反序列化工程问题，不是模型质量失败。
已实现只接受这两个**确切文件大小+LFS SHA**的受限loader：固定PyTorch2.11.0+cu128 tensor-safe globals及7个审阅过的Namespace/YAML数据类、固定pickle protocol2，拒绝扩展registry/未知global，不动态导入checkpoint指定模块。自定义受限Unpickler必须配合weights_only=False兼容标志，**不能声称仍为内建weights_only，也没有退回默认/不受限pickle**。WeTok仍使用内建weights_only=True。小型恶意global拒绝/合法元数据回转测试通过，真实独立tokenizer的CPU元数据读取通过，未建立CUDA上下文。
计时边界细化：codec API内的参数验证、内存prefix/hash核验计入时间（并非纯GPU kernel计时），文件I/O与指标网络不计入。prefix复用一次完整tokenization，因此不是最小化prefix编码器的时延宣称。old-official和XQ在各自进程会常驻配套生成器，full臂的绝对显存也包含它；同时报告每操作增量，不将此解释成仅tokenizer最小显存。fidelity和WeTok不常驻生成器。
