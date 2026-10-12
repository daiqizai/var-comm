# NeST-Com v5 修改记录

基于用户提供的 `nestcom_wcl_draft_v4.tex`，证据版本保持 `260a2393c6c4e25b4647c525ec88406df432d4cd`。

## 篇幅

TeXcount统计正文及图注等非标题文本，由 **2450词**减至 **1940词**，减少 **510词（20.8%）**。参考文献和数学符号不计入这个口径。编译PDF为5页。

## 直接删除的典型句子

| 原文 | 处理 |
|---|---|
| These works establish the foundation, rather than the contribution, of the present interface. | 删除。前文已有准确的相关工作描述。 |
| Global optimality is not claimed. | 删除。优化式已限定有限候选集合。 |
| capacity optimality is not claimed. | 删除。保留模型规模、训练步数与校准平台事实。 |
| Fig. … shows a metric-dependent tradeoff, not blanket superiority. | 删除前置辩解，直接报告各指标的实测胜负。 |
| The small 13-dB gain should not be confused with a large perceptual advantage. | 删除。保留LPIPS差值−0.00247。 |
| Thus raw partial transmission is not universally quality-optimal … | 删除。保留强熵编码在19 dB反超0.258 dB的结果。 |
| This qualitative comparison does not establish a test-set ranking, speedup, or isolated VAR-versus-diffusion advantage. | 删除。段落只描述定性比较的接收实现和来源。 |
| These values belong to the separate 100-image cohort, not Fig. … | 删除。数据集身份已在段首和图注交代。 |
| This is not a fading-channel test … | 删除。正文已给出AWGN模型和SNR失配操作。 |
| the full system is not untrained. | 删除。保留预训练骨干和Dc适配过程。 |

## 改成正面定义的内容

- 将“不是任意中断LDPC码字的独立恢复”改为“发送端在编码每帧前选择(m,K)”。
- 将“不声称每一帧严格等能量”改为固定标称星座能量及组成相关帧能量的描述。
- 通过“LDPC硬判决”“固定空类别条件”“从译出的token重建CDF”定义实际接收信息，不反复声明没有使用真值。
- 通过ConvNeXt原图—重建top-1预测一致性的准确定义说明指标，不反复否认其为语义正确率。
- Swin仍在改善时预算停止的事实只保留一次；删除相邻的收敛性否认与重复的19 dB范围提醒。
- 图注只保留读图所需的条件、比较定义、版本和置信区间。
- 将106个整尺度候选的补充复核放到对应消融结果处。
- HiFi图注仍注明来源21、92是查看早期图库后选择的开发样例，不将它们改写为随机测试样本。

## 结果与接口

公式、数据表、引文和参考文献逐项对比均未改变；曲线与系统图复用v4图像文件。PSNR/LPIPS等负结果、SNR失配退化、两类源码接口的失败规则、头部开销与功率定义仍在正文中。

删去没有实测数值的重复计算范围段；统一计时待办改为源文件注释。旧Kodak表继续留在已有的 `SUPPLEMENTARY_NOTES.md`，没有在主文重复归档过程。

## 文件与待补资产

- `nestcom_wcl_draft_v5.tex`：当前正文。
- `nestcom_wcl_draft_v5.pdf`：当前5页PDF；图4仍为明确标注的占位。
- `audit/v4_to_v5.diff`：完整逐行差异。
- `audit/editorial_changes.json`：逐项修改前后文本。
- `audit/editorial_validation.json`：内容不变性检查、字数和编译检查。

作者信息、图4实际PDF、统一收发计时仍按原计划补齐；导入图4的方法见README。
