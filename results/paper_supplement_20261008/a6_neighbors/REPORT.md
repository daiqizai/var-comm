# A6：最近邻机制与可运行性核对

核对日期：2026-10-09（Asia/Shanghai）。主结果仍是 `252176e041758ecb2d3e81b6fde5b587e7e17bb7`。

**首轮文献和作者代码核对已完成；外部 native 实测均为 NOT_RUN。** 本目录的 `comparison.csv` 给出 18 项差异，`availability.json` 逐项记录证据、运行缺项和唯一候选。没有下载模型权重、导入 CUDA 模型、运行推断、训练或改写原科学结果。

## 可写入论文的区别

ARPC 已经研究“发送多尺度前缀、生成未传尺度”和 AR 熵编码，因此不能把这些一般思想作为本文独有新颖性。作者实现使用 group-masked bitwise quantizer 与 Infinity，默认 demo 依赖 caption/T5。本文应定位在冻结 raw12、多尺度内部分发送、实际有限码长接收及逐帧预算下的取舍，并用现有消融支持。旧完整尺度或算术增强结果仍是内部参考，不能重命名为 ARPC 外部复现。[作者代码](https://github.com/Joanna-0421/ARPC/tree/f7480889175271a58c7e5f8e48e0b58f247ccf63) · [正式论文](https://tianweiz07.github.io/Papers/26-iclr-3.pdf)

Ada-TokenCom 与本文同样涉及前缀生成和 MCS 选择，但原文的主协议使用长期平均资源约束、Type-I ARQ、类别条件，质量以成功重建子集计算。本文 N1024 逐帧上限、单次 AWGN 和全帧统计不能直接与其图表数值合并。若另做适配，条件信息、反馈与失败处理都要单列。[原文 II–IV 节](https://arxiv.org/html/2608.28086v1)

## 唯一可继续考察的 native 候选：ARPC

作者 release 的源码与推断入口可见，README 给出 `var_codec.pth`、`vae_d16_reg.ckpt` 两个作者权重链接；网页返回成功并显示文件名，**不等于已下载、完整校验或加载成功**。尚未获取权重字节数或校验和。代码需要 `weights/flan-t5-xl`，使用 `T5EncoderModel`；不能把该组件漏出存储/显存核算。

- 论文 Table 7 报告 AR 约 2.2B 参数，VAE 发送端 44.9M、接收端 65.0M；这些是作者报告，非本机实测。Table 7 的合计不作为当前 demo 完整驻留内存估计。
- 作者环境声明 Python 3.10.16、Torch 2.5.1、Torchvision 0.20.1、Transformers 4.51.3、Triton 3.1.0 和 Flash Attention，并包含 Linux/CUDA 依赖；本机 Python 3.12 没有 Torch、Transformers、Triton、Flash Attention，静态报告也未找到 conda/CUDA 编译工具。远端环境本次未复核，不能由本机结果推断远端不可运行。
- 默认 `demo.py` 设置 `pn='1M'`，方形输入为 1024×1024。**不是仅支持 1024**：`dynamic_resolution.py` 中 `0.06M` 对应 256×256，论文 Table 6 也列出该分辨率。现成码长统计却硬编码除以 1024²；改尺寸前须显式校正分母并核实检查点，不能把默认 demo 当现成 256×256 对照。
- demo 传递 Python 嵌套 bit 列表和 help flags。码长累加不包含 caption 或可独立解析的分包/长度容器，没有实际付费头部、LDPC、MCS、N1024 和噪声接收路径。故当前同协议适配标 `UNSUPPORTED`，含义是现有实现未覆盖，不是判断无法开发。
- `reciever.py` 的最终有效 `decoding` 接受名为 `gt_ls_Bl` 的参数，但静态语法检查显示函数体没有读取它；`demo.py` 给 `decompress_cfg` 传的是解出的 `dec_idx`。不能仅凭变量名指控真值泄漏。正式自检仍须断开源 token 输入，并验证算术往返、截断边界和概率一致性。

模型大小、环境和上述接口证据分别来自 [论文 Table 7](https://tianweiz07.github.io/Papers/26-iclr-3.pdf)、[environment.yaml](https://github.com/Joanna-0421/ARPC/blob/f7480889175271a58c7e5f8e48e0b58f247ccf63/environment.yaml)、[demo.py](https://github.com/Joanna-0421/ARPC/blob/f7480889175271a58c7e5f8e48e0b58f247ccf63/demo.py) 与本目录保存的源码。

## 第一个 case 的最小入口与停止点

入口是作者 `demo.py` 中 `load_tokenizer → load_visual_tokenizer → load_transformer → compress → decompress`。作者 fixture 首项是 `data/DIV2K.json` 的 `0801.png` 及对应 caption；文件路径是作者机器的 `/gemini/code/Dataset/...`，图像本身未包含在本目录。直接执行原 demo 会遍历该 manifest，**本轮没有执行**。

后续若实际开展，只选择 ARPC 一个方法：先准备隔离的作者环境、完整源码、验证过的三个权重组件和输入；绑定旧校准集的前 8 个固定源（不从 holdout 选图），首先仅运行第 1 个 source。保留 native 预处理和条件，计一次收发耗时、峰值显存和模型文件实际字节；只有成功后才扩到已固定 8 个。若用现有 256 图放大运行默认 demo，应称放大后的 native 接口诊断，不称原生 1024 内容或等预算通信结果。当前校准源/原生 caption 尚未绑定，native 固定小样本状态为 `NOT_RUN`。

还缺：权重实际可下载和加载凭证、CUDA/扩展兼容、校准图与 caption 绑定、独立可解析码流往返、真实单 case 成本。付费无线适配另缺 caption/容器计费、合法 FEC/头部/失败规则、固定策略与真实接收验证。没有这些证据，不启动文档中建议的 900 帧比较，也不填入任何零分或猜测耗时。

## Ada-TokenCom 可用性边界

已读取全文和作者出版页，并查询作者 GitHub/准确论文名；本轮未定位到 Ada 专属作者仓库、权重清单或可执行入口，所以 native 标 `NOT_RUN`。这只是截至本次核对的可发现性结论，不声称代码永远不公开。`liqiao19/TokenCom_Code` 的 README 明确对应另一篇 2025 年论文及 MaskGIT，不能借其名字充当 Ada-TokenCom。 [作者出版页](https://liqiao19.github.io/publications/) · [TokenCom_Code 身份](https://github.com/liqiao19/TokenCom_Code)

## 文件和验证

`comparison.csv` 为机制矩阵；`availability.json` 为状态及缺项；`STATIC_VERIFICATION.json` 为作者提交绑定与静态检查；`primary_sources/` 保留原文、作者代码和抓取回执。原文 PDF 约 20.7 MB，为本地阅读缓存，不据此修改发布过滤器或默认入库。A6 这一首轮完成的是最近邻表与可复现性边界，不是外部性能复现。
