# 外部方法盘点与本轮执行选择

方案标识 EXTERNAL-COMPARISON-20261004；盘点实际执行于 2026-10-03。用户后续指令优先于原附件：重训 SwinJSCC MSE，接冻结 ADM 的 HiFi-DiffCom；其他新外部实验按下表处理。

## 盘点范围与证据

已遍历 `/home/liulu/projects`、实际存在的 Hugging Face 缓存和 Torch 权重缓存。共枚举 1,506,723 个文件，记录 7,710 个权重文件、18,144 个方法名相关结果文件，并检索代码、配置与较小文本证据。权重计数含实验断点和重复副本，不代表独立模型数。

跳过 `.git`、环境包、逐源指标大文件的正文，以及所有 holdout 目录和名称；不跟随目录符号链接。唯一额外目录链接是环境 `lib64→lib`，与模型资产无关。发现一个失效的 Firefox 临时锁路径，未影响方法或权重盘点。完整枚举与源版本保存在 `outputs/EXTERNAL-COMPARISON-20261004/inventory_v2/`，提交压缩后的可复用资产清单与检索回执。

## 可用方法

下文根目录为 `/home/liulu/projects/VAR_COMM`，旧外部实验根目录为 `experiments/external-baseline-positioning-20260916`。

| 方法 | 已有资产及结果 | 资源和公平性边界 | 本次处理 |
|---|---|---|---|
| SwinJSCC 官方 SA+RA base | `vendor/SwinJSCC`，源版本 `a6d0e6da53548976acbe9317839a077ef31f190f`；已有 MSE/PSNR 权重；旧 100 图×5 SNR×3噪声结果与浮点 RGB 缓存齐全 | 旧正文 C32/64/96 对应 N4096/8192/12288；付费通道掩码和功率后总 N4498/8754/12952。作者目录采用 DIV2K/CLIC，公开权重的精确训练源和历史不能由评测配置推断 | **目标预算需重训**。复用官方结构和代码，在原 ImageNet20k/校准1k、256×256上随机初始化一个双预算模型 |
| 项目此前自行训练的 SwinJSCC SA | 另一个项目的 `EXP-S34A-SWINJSCC-BASE-SA-EQUAL-BUDGET-001` 和 `CM-SA` 目录有 best/latest 等权重 | 这些是项目训练结果，不是新任务所需的官方 SA+RA 双低预算训练。不能因文件名相同就当作对应低码率模型 | 保留历史记录，本轮不截短或挪用 |
| HiFi-DiffCom + ADJSCC | `vendor/diffcom_code` 版本 `a8cc4d63304f5bb514d69576e689ff2412510161`；已存在无条件 ImageNet256 ADM；旧运行与缓存入口存在 | 上游实际默认算子是 ADJSCC，Swin 支持仍是 TODO。旧 C2 正文 N4096，与目标 N1024/2048 不同。功率/排列假设须付费或明确登记 | **复用冻结 ADM，新增 Swin 后验算子适配**。必须先过完整图像梯度、前向一致性和真实采样检查 |
| ADJSCC | 同 DiffCom vendor；C2/C4/C6 官方权重、旧开发集结果和缓存已有 | 256×256正文 N4096/8192/12288；旧作者假设与付费元数据视图分开。无可直接冒充 N1024/2048 的已训权重 | 按用户最新指令不新增实验 |
| SGD-JSCC | `channel-adaptive-semantic-drift-controlled-diffusion-jscc/third_party/SGDJSCC`，版本 `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`；原 500 行结果及5个重建张量存在 | 源256×256切四块128×128；N9856，包含边缘信息；每块免费文本描述。原单噪声口径及信道设置也与当前主表不同 | 相关工作讨论，**不放入同资源主表、不新增重训** |
| 项目 DeepJSCC-3060 | 旧 checkpoint 与结果证据存在 | 项目自己训练、N3060，不能称官方预训练；不满足当前两个低预算点 | 按用户最新指令不新增实验 |
| DiffJSCC | 有历史方案、停止记录、相关源码/配置引用 | 文本命中数不等于已完成的可用模型；本轮未把匹配记录当作目标预算权重 | 按用户最新指令跳过 |
| DiT-JSCC | 本地为文献/方案引用；官方仓库实时核验仅 README，版本 `d72503a831da8da0a45abf5bad16c2f951dd11cc`，没有 releases | 截至核验时没有可执行公开实现或公开权重；不能据论文带宽范围声称现成可跑 | 仅相关工作 |
| NTSCC、WITT、MambaJSCC | NTSCC/Mamba 命中相关工作；WITT 另有 SGD 包中的嵌套编码模块 | 未发现经登记、已完成、能直接覆盖目标预算的独立实验。嵌套模块不能当成完整基线 | 不新增实验 |

## 已确认的冻结生成模型

权重路径：`experiments/external-baseline-positioning-20260916/checkpoints/256x256_diffusion_uncond.pt`。

- 模型：公开无条件 ImageNet 256×256 ADM。
- 大小：2,211,383,297 字节。
- SHA256：`a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1`。
- 本轮不训练扩散参数。作者配置中的 `N=1.0` 是采样步数乘数，不是通信资源 N。
- Swin 新适配使用真实接收波形、译出的付费掩码与功率；不得读取发送端真值。

## 来源与复现

[SwinJSCC 官方仓库](https://github.com/KeYang8/SwinJSCC)、[DiffCom 冻结源配置](https://github.com/wsxtyrdd/diffcom/blob/a8cc4d63304f5bb514d69576e689ff2412510161/configs/diffcom.yaml)、[DiT-JSCC 官方仓库](https://github.com/semcomm/DiTJSCC)、[ADM 官方权重](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)。精确源码/API核验见实验目录中的 `verified_primary_sources.json`。

新训练、码率和 SNR、能量、元数据、选模、停止和失败处理以同目录 `PROTOCOL.md` 与 `swin_training_protocol.json` 为准。LPIPS 微调属于 MSE 主比较后的可选扩展，尚未启动。
