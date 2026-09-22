# WeTok辅助完整重建候选：源码/公开权重核查

本项只核查并实现完整codec参照，不接入通信、不训练、不运行GPU。
实际GPU重建和同图指标仍须由受控评估入口完成；以下论文数字不是本地结果。

## 精确选型

- 作者代码仓库：`https://github.com/zhuangshaobin/WeTok`。
- 固定代码提交：`caa2ad7e709cdabe8432bead448f7514def13919`。
- 固定配置：`configs/WeToK/Inference/ImageNet_downsample16_imagenet.yaml`。
- 作者README链接的权重仓库：`https://huggingface.co/GrayShine/WeTok`。
- 固定HF提交：`85fc6eb084d458b8d4fa3a32d541379e95b2bf87`。
- 精确文件：`ImageNet/downsample16/WeTok.ckpt`。
- 文件大小：3,763,537,210 bytes；SHA256：
  `e40a0bcdefd8509b2201e93a9f5007d38c8fba9423c17a98b0e45f9e8763e696`。
- canonical下载地址：
  `https://huggingface.co/GrayShine/WeTok/resolve/85fc6eb084d458b8d4fa3a32d541379e95b2bf87/ImageNet/downsample16/WeTok.ckpt`。
- 当前实际可用的mirror resolver：
  `https://hf-mirror.com/GrayShine/WeTok/resolve/85fc6eb084d458b8d4fa3a32d541379e95b2bf87/ImageNet/downsample16/WeTok.ckpt`。

代码由官方GitHub REST contents接口按固定ref下载原始字节；未下载图示/其他模型。
官方HF域名直接连接超时，因此文件列表/大小/LFS SHA来自上述镜像的固定revision API，
身份由作者README交叉核查。权重下载由父任务单流、限速、串行安排；本子任务没有下载大权重。
没有把镜像LFS元数据伪装成已经逐字节核验过下载文件；adapter加载时必验实际size/SHA。
所有小文件SHA与Git blob哈希见`wetok-source-lock.json`；HF tree原始收据在vendor目录。

## 论文数字与公开配置的对应

作者论文`https://arxiv.org/html/2508.05599v1`的ImageNet-50k重建表中，
WeTok 16×16、stride16、256输入、隐式`2^32`词表这一行报告PSNR24.50dB、rFID0.61。
这里明确引用v1，未声称为最新修订。它与相邻32×32/stride8或GeneralDomain结果不是同一配置。
作者推理shell明确把`ImageNet_downsample16_imagenet.yaml`与
`ImageNet/downsample16/WeTok.ckpt`配对。源码配置实际为：

```yaml
n_embed: 256
embed_dim: 32
num_codebooks: 4
use_ema: true
ddconfig:
  z_channels: 32
  ch: 256
  ch_mult: [1, 1, 2, 2, 4]
  num_res_blocks: 4
```

因此目标256图像的encoder输出为`B×32×16×16`，每个位置分4组，每组8bits。
完整表示为`16×16×4×8=8192 raw bits/image`。源码LFQ实际返回扁平整数索引，
batch1是`[1024]`，逻辑布局为`[1,16,16,4]`；不能误计为256个8-bit索引。
论文中的隐式`2^32`不是显式分配一个4294967296项大码本。
配置的`ddconfig.resolution:128`在对应Encoder/Decoder仅存为属性，未用于裁剪，
数据配置和本次入口固定256，不能因为该属性把输入改成128。

## 推理实现与不扩大环境依赖

作者`VQModel`会初始化Lightning和对抗/感知loss，`env.sh`还包含TensorFlow、cuBLAS安装等；
本项**不运行env.sh、不修改全局环境、不下载其训练用loss权重**。
`../scripts/wetok_adapter.py`导入未修改的固定作者源码：

- `src/WeTok/modules/diffusionmodules/improved_model.py`中的`Encoder`和`Decoder`；
- `src/WeTok/modules/vqvae/lookup_free_quantize.py`中的`LFQ`。

运行依赖仅PyTorch、einops和PyYAML，实验私有venv已具备。
配置不启用`gan_decoder`、`use_GFQ`或`token_factorization`，adapter对此断言；
不能将其它WeTok变种静默混入此profile。

作者重建入口在`model.ema_scope()`中调用encode/decode。
adapter按作者`LitEma.copy_to`规则，把每个参数名去掉`.`后映射到`model_ema.*`；
逐个要求存在且形状相同，再strict加载全部encoder/decoder参数，不允许残留随机权重、
缺失EMA或默默退回普通非EMA权重。实际codec结构总参数量460,294,179；该值仅meta结构计数。
使用`torch.load(weights_only=True,mmap=True)`，如果真实checkpoint包含不支持的对象则失败，
不会自动降级到不受限pickle；届时必须单独记录检查结果。

`encode(x[-1,1])`只返回整数索引和固定布局元数据，不保留原图或连续latent。
`decode`从索引通过作者`LFQ.decode`恢复±1离散latent，再解码成clamped float RGB01；
完整评测避免PNG往返改变指标。
作者LFQ在eval中仍保留STE数值表达式，可能与严格从索引恢复的±1存在微小fp32舍入差；
adapter逐图记录`official_quantized_vs_index_roundtrip_max_abs`并要求≤1e-5。
没有prefix/生成补全API，传入prefix即报错。

所需父配置keys：`wetok_source`、`wetok_config`、`wetok_checkpoint`，均为绝对路径。
所有code/config/checkpointSHA随metadata保存，且decode不加载任何生成器。

## 已做与未做

已通过4项CPU工程检查：
1. 固定作者源码和配置的SHA；
2. LFQ实际扁平索引、四分组、8192bit及离散latent回转；
3. EMA权重映射、缺失权重硬失败；
4. meta设备构造结构和参数计数（未分配真实模型权重）。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tests/test_wetok_adapter.py
```

**尚未加载公开大checkpoint，尚未做真实图像推理或GPU计时/峰值显存测量。**
后续仍须在同一100张development和统一PSNR/SSIM/LPIPS/DINO上对比。
论文24.50dB不能直接减去旧项目22.83dB当作收益；8192只是无信道raw源bits，不是复信道次数。
