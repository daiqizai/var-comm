# 可信前缀量化条件：无训练接收端修正开发 gate

日期：2026-09-06。状态：在本轮真实图像结果产生前登记。
用户授权范围：评估给定方向并推进20张自检、100张无训练开发实验。
不重启已终止的 diffusion/HDA/scale7 分支，不覆盖旧结果、不训练网络、不接入信道。

## 判断与必要修正

这个方向值得作为可证伪的小实验，而不是已成立的新方法。新增机制只看同一组
已接收 token、共享类别和模型、同一个 VAR 起始 latent；能否改善图像由配对指标判断。
“更符合量化cell”不等于“更接近原图”，更不能直接宣称PSNR/LPIPS/DINO有保证。

现状与所贴建议有两处重要差异：

1. 现有补全为无CFG、确定性argmax，换随机种子不会产生独立样本。主消融保持argmax；
   另将 `top_k=900, top_p=0.95, cfg=0` 的三种子补全明确标为随机生成开发诊断。
   三种子固定使用argmax开发集全局选出的参数，不每图、每种子重新选参。
2. 2026-09-04 已完成 ImageNetV2 2,000 query 和完整10,000 gallery实验。
   该2,000张不得按建议直接重复当“全新正式测试”。未来必须另做跨项目历史使用审计。

已有的 `m7/m8/m9 + FEC` 冻结为基线。当前 source class 是两边共享的真实类别，
本轮无噪声，不把“已知正确前缀”推广为真实PHY可检测可靠性或信道纠错能力。

## 实现合同

- `F = vae.quant_conv(vae.encoder(x))`；解码为 `decoder(post_quant_conv(Z))`。
- 以**固定收到**的前面 token 贡献计算残差；重编码失败后绝不采用预测token更新条件。
- Down 使用官方 `area`，尺度贡献先 `bicubic` 上采样，再使用完整十尺度索引的 `phi`。
- 同时实现欧氏距离和归一化相似度；接收码字不参与“最近竞争码字”搜索。
- FP32，禁用autocast及TF32。数值容差按归一化margin `1e-5`；输出逐尺度违反率、
  平均正margin和最大正margin。主违反率为逐尺度等权均值，不按token数隐式加权。
- 只优化Z，所有共享网络冻结；前后完整VAE/VAR state SHA和梯度边界验证。
- 接收端模块接口不接受x、F、未发送token；这些只在独立评测/缓存路径出现。

欧氏一致性loss为收到码字距离与最近其他码字距离的正差，除以codebook平均平方范数；
anchor MSE除以codebook平均平方坐标。归一化相似度分支使用竞争者减目标的正cosine差。
只惩罚越界，不把已可行latent拉向码字中心。位置内平均，再跨尺度平均。

## 20张自检

使用原100张manifest前20张，m8/m9各检查F、full-VQ、VAR；F必须容差内零违反，
full累积必须与官方完全一致，新receiver-only补全必须重放历史argmax latent（最大差≤1e-5）。
学习率候选依次0.01/0.003/0.001；固定lambda=1、20步，选择所有20图两个m下
最终总objective不高于初始值的第一个候选。选择不调用原图指标。

## 100张开发消融

全部沿用原ImageNet-100与官方模型。每个m比较：

| 组别 | 参数 / 作用 |
|---|---|
| 原版VAR | 完全相同的argmax起点，重新统一FP32评测；不直接混用旧FP16缓存指标 |
| prefix线性混合 | alpha=0.1/0.25/0.5，全开发集固定选一个 |
| 仅最后收到尺度 | lambda=0.1/1/10，T=5/10/20 |
| 全部收到尺度 | 相同lambda/T，候选机制 |
| Full-VQ | 使用完整真实token的**更多信息参照**，不是等码率方法 |
| Full-VQ同样修正 | 从Full-VQ出发，用同一r1:m约束与同一参数网格；报告相对Full-VQ差 |

一个20步轨迹保存第5/10/20步，禁止看单张原图指标挑步数/输出。
每个m在DINO/PSNR护栏内按全开发集LPIPS均值选参数；若无eligible参数，记录最小LPIPS
但必须标记护栏失败。完整网格全部保存，不只展示最好的几行。

主表：违反率、PSNR、LPIPS-Alex、DINO、配对95%CI和新增延迟。原始结果逐图逐设置保存；
三种子先按源图平均配对差，再按图像bootstrap10,000次。开发选参后的pointwise CI
是探索性区间，不是独立验证或经过多重比较控制的显著性结论。
额外延迟区分约束构建、优化/混合、共同generation/decode；质量评测不计入receiver延迟。

## 停止门槛

两个m均要求：LPIPS相对降低≥5%、配对差CI上界<0、DINO平均下降≤0.005、
PSNR平均下降≤0.2dB，且LPIPS均值优于最佳简单混合和最佳最后尺度约束。
若联合约束与最后尺度无可靠额外差异，不能宣传多尺度联合收益。
若full-VQ相似改善，优先解释为通用tokenizer后处理，不声称VAR特有。

不通过时停止这个latent-cell修正候选，记录负结果；不自动启动LoRA、Decoder训练、
后续真实尺度辅助译码或CRC/PHY实验。通过也只能记为开发可行，尚需独立校准和新正式测试。

## 复现

```bash
python3 scripts/evaluate_var_prefix_consistency.py selfcheck
python3 scripts/evaluate_var_prefix_consistency.py development
python3 scripts/evaluate_var_prefix_consistency.py seed_diagnostic
```

三个阶段仅写各自新输出目录；下一阶段验证上游receipt、源码/配置和所有产物SHA。
目录存在即拒绝覆盖。运行依赖全部使用本地官方checkpoint，无模型/数据下载。
