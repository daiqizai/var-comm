# 固定3060接收器结果：可用图与原图入口

这些图全部来自已完成并冻结的RX-only结果，没有新推理、训练或test访问。全部方法总预算为3060复信道使用、能量6120；模型规模、训练历史和header占比不相同，不能称等参数比较。

## 建议使用的展示图

- `../outputs/WETOK-INNOVATION-R1-PRESENTATION-20260914/strong_comparison_training_support_anchors.png`：五个共同训练支持SNR上的PSNR/LPIPS/DINO；另以叉号标明Deep在5/6 dB的既定支持点策略，不把旧未训练条件输入的异常当新方法优势。
- 同目录`deep_conditioning_diagnostic.png`：单独保留旧Deep条件输入在5/6 dB的诊断，不修改任何历史指标或主排名。
- 同目录`fixed_source_000.png`、`fixed_source_037.png`、`fixed_source_099.png`：固定源、固定噪声2001、1/7/19 dB对比，不按质量挑选案例。全部有PDF版本。
- `../outputs/WETOK-INNOVATION-R1-FIGURES-20260914/primary_paired_lpips.png`：主1/4/7的成对LPIPS区间，阴性结果未删掉。

## 单张原图

`../outputs/WETOK-INNOVATION-R1-FIGURES-20260914/originals/{000,037,099}/`保存各方法独立256×256展示PNG及源图。`image_manifest.json`逐图记录原float32 archive、索引和SHA；指标来自float32而不是PNG。

展示版拼图直接粘贴原256×256 PNG，没有缩放或修改图像内容；另用CPU逐块比对72个展示图块，与原PNG完全一致。第一版拼图行标题拥挤，已保留原记录并另生成上述清晰排版，不覆盖指标或旧图。

`all_snr_strong_comparisons.png`是保留原七SNR条件的完整诊断图，含Deep的离支持点异常，不建议脱离说明单独用于方法排名。正式汇报优先用展示目录的支持SNR视图与补充策略标记。

这些仍是development材料，不是新的独立验证，也不把全网格迭代改称next-scale贡献。
