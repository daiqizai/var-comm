# 接收接口与反向近似：受控训练协议

2026-09-12，在新profile/训练前登记。替代已探索性提前停止的bit权重续训，不修改其失败现场。

## 通信问题与可否定假设

原native学习映射在noiseless/19dB仍损失大量源信息；2000步表示恢复后，图像联合训练引发logits膨胀和校准恶化。单独bit权重0.01→1未在首1000追加步遏制该现象，已共同停止六臂。这不是对所有权重的否定，也不是已经查明唯一根因。

H1：无界logits上的identity-ST不能合理调节图像梯度；有界均值的ST可能减少失控。H2：无法确定的原生bits被强制硬化后，Decoder看到严重受损的码字；真正训练连续特征接口可能得到更好的失真取舍。两项均需实际图像质量验证，不是新通信贡献的预先宣称。

## 三接口 × 三结构

设a为现有接收logits，h=sign(a)（零按-1），u=tanh(a/2)。u与当前bit-logit参数化相符，但不假设其概率已被校准。

| 接口 | 训练/评测Decoder输入 | 反向到a |
| --- | --- | --- |
| hard_identity | h | identity-ST |
| hard_bounded | h | du/da = 0.5(1-u²)的ST |
| continuous_mean | u | u的通常反向传播 |

hard_bounded仍是有偏近似，不是argmax可微化；与identity的差异同时含梯度尺度与置信度相关衰减，不能单独声称只验证了后者。两个hard接口在相同权重/观测下必须像素完全相同。continuous始终连续，不把soft训练/hard评测混用，不称合法WeTok token、不当无需训练上界。其符号错误率仅事后诊断，不是实际数字恢复率。

每接口都训练single_pass、multiscale_no_history、multiscale_conditioned。所有结构的N/E、源输入、数据曝光与可用观测相同；无历史控制仍执行4/8/16三个网格。每臂2895944参数。三阶段读取完整y，不冒称渐进传输。

## 共同历史与不变项

从每结构原2000步表示恢复checkpoint复制三个接口。该端点没有保存完整Adam，故九臂**统一新建**同设置AdamW；绝不猜造原moment。原配方5000继承Adam实验不能与本次视为相同训练历史。主要接口因果比较只在本轮共同重置的同结构内进行。

保留原joint权重MSE=1、LPIPS=0.01、bits=0.01、state=0.01，不再同时加入新loss/退火/扩模型。LR=1e-4，betas与weight decay沿用base，clip=1、fp32、micro1/effective4。从global step2000起配对数据、翻转、五SNR与噪声。两中间状态仍为自己恢复的连续4/8状态；RX输入只含y和名义SNR。

官方WeTok视觉Encoder/量化器/Decoder、数据、源8192 raw bits、0 header+3060 data complex uses、平均复能量2/总6120、每实维噪声方差1/gamma全不变。无类别/caption、反馈或额外信道。最终Decoder冻结但保留输入梯度。不添加功率分配。

## 训练、校准、选模与成本

先做九臂无更新工程profile，验证hard前向一致、image梯度到E/D非零、连续输入范围、能量与冻结权重。首追加1000为检查点，初步计划5000；按完整校准走势决定继续，不以里程碑作研发上限。

完整1000图校准在追加0/1000/2500/5000以及明确追加的里程碑；每500步固定100图监控。九臂同图同噪声，最小五SNR完整校准LPIPS选模、平局取早；monitor不能选模，DINO不训练/选模。PSNR逐SNR报告，与同接口single_pass差超过-0.2dB时明确提示，不用失败fallback伪造结构差值。

允许保留真正的追加0全校准候选，但必须报告选中步数；原2000最佳另列，不能只胜退化父点。非有限loss/梯度、资源/接口或冻结校验异常时停本轮并记录，各臂以最后完整配对checkpoint为准。若校准继续恶化再作明确科学决策，不能偷偷更改正在运行的权重/梯度。

预计九臂5000更新主体约4–5 GPU小时，另加完整校准约1小时，实际profile后更新估算；首1000应约1–2小时。这是费用/时间估算，不是新研发上限，未租新机器。

## 最终判断与比较

同结构hard_bounded−hard_identity检查反向近似；continuous_mean−hard_bounded检查训练后的最终接口；同接口conditioned−no_history检查条件历史。在同骨干single_pass之外保留原2000历史最佳、WeTok数字8PSK/FEC、旧数字m8/自适应与感知Deep的同N/E结果。

最终评测原100 development×七SNR×三个旧配对噪声，统一PSNR/SSIM/LPIPS/DINO、严重失真与收发时延；源图先平均噪声/SNR，再paired bootstrap。noiseless诊断与正确native重建单列，不作无线方法排名。连续输入增加幅值/饱和率诊断，禁止按真图决定采用hard/soft。没有新holdout和额外测试曝光。只有通信质量/资源/可靠性改善才能算系统进展，修ST或换loss本身不算论文创新。
