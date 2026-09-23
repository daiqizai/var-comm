# VAR_COMM：迁移后执行审查与研究推进建议

固定审查提交：`9e8bee77c946fb8ee6618cf4c1ae6cf2bd10d6dc`。
日期：2026-09-23。此文是审查与建议，不是已经修复或运行新方法的回执。

## 1. 审查范围

通过GitHub连接固定读取最近的增强模型、latent定义、训练与数据接口、数字PHY包装、统一部署、计时、Decoder适配、线性测量、预测创新、策略统计及已发布结果。重新核对HDA-DeepSC、Hybrid Semantic-Complementary Transmission公开全文，并检索用户保存的SharpCast、SK-Cast、DAC-Mobi原文相关章节。

本次在本地运行10组CPU函数摘录、结构和数学探针，结果见`probe_results.json`。探针包括成功复现缺陷，不是“10项工程全部PASS”。没有加载用户真实权重/图像，没有GPU重推理，没有重新计算全体LPIPS、DINO或配对bootstrap。直接git clone因本地网络DNS失败，故不是本地完整clone后跑过全仓测试；仓库记录的57项通过仍属于原执行记录。

旧的`source_export_002`只用于定位历史文件，不拿旧字节替代最新实现。本次结论以固定commit中读取的内容为准。

## 2. 总体判断

1. 原阶段A/B核心可以继续作为研究资产，未在本轮重点阅读中发现真实F直接绕过信道进入接收器、增强段不经噪声、或基本能量二倍因子错误。
2. 已有1024增强是有价值的工作点，但不是已证明全指标领先：修复配对文件对同N4084 raw数字质量策略支持约+0.562 dB PSNR；LPIPS差约-0.000303，区间跨0；DINO约-0.01916。
3. 后续脚本仍有确定问题：线性控制失败误删基础投影、W校准只用最后一个噪声、提前写COMPLETE；预测创新结尾缺`arm_key`、实际TX条件与配置不符。
4. 原来的“只换残差公式”建议需要降级。在当前线性首层接口下，`[F-Fb,Fb]`与`[F,Fb]`有精确重参数化，不自动形成新的编码能力。
5. 下一步不宜再并行堆6种模块。先收口这些局部实现，补同F/同Dc/同总N的纯连续控制；随后只保留一个清楚的通信改进主线。

## 3. 各方向执行状态

|方向|代码/结果判断|处置|
|---|---|---|
|raw/算术数字m7/m8/m9|可保留为强系统参考；不等于全m10/所有FEC最优|沿用校准冻结动作，明确模式范围|
|原Stage A连续Decoder|连续表示参考有价值，不是有限带宽无线成绩|保留checkpoint，不重新训练|
|原Stage B 512/1024/refiner|有限增强输入、实际RX基础、原图损失与latent损失有对应实现；能量/梯度局部测试合理|保留，1024为主候选；512是有取舍的资源点|
|Decoder适配|optimizer已逐臂deepcopy；真实算法为交替通信更新/Dc更新|保留为后续适配分支；不要描述成单次同图联合更新或普遍更强Decoder|
|数字/连续折中分配|此前三点比较已有结果|不要反复重跑这一个点；下一步才是校准冻结的选择和必要m变化|
|计时v3|零基础/双噪声已经改掉；还缺全方法一致性smoke及少量必要搬运清理|有限补验，不因此重训|
|真实投影测量|幅度已通过32-use控制包计费，但回退与校准仍有代码错误|先修该分支；随机A不能代表原论文最优投影|
|预测后残差|当前校准/恢复框架比旧版完整，但结尾会缺键；完成记录只有校准选择，不是新development胜出|修selected；明确接口后再决定匹配实验|
|显式分组功率|未见独立完成证据|列NOT_ESTABLISHED，不当成已经试过且失败|
|coset/扩散|未作为当前匹配分支验证|暂不扩张|

## 4. 当前代码中的确定问题

以下E表示`experiments/var-latent-enhancement-20260917/`。

### N01 预测创新收尾缺少arm_key

文件：`E/mechanisms/src/latent_mechanisms/predictor_innovation.py`，`main()`末尾。

`selected[n]`由只含step/utility的`state['selection']`构造，加入checkpoint与步数，但未加入`arm_key`；之后却执行：

```python
arm.load_state_dict(payload[r['arm_key']], strict=True)
```

本次原构造表达式探针得到`KeyError: 'arm_key'`。中间独立selected JSON写过arm_key，不代表这个新建内存字典带有它。

修复：唯一映射`original_residual_control -> control`、`predicted_innovation_candidate -> candidate`进入同一个选择记录生成函数；训练保存、恢复、评测共用记录，并核对对应checkpoint SHA。

重跑：已有合法训练checkpoint不用重训。补选中checkpoint的校准/开发评测即可。当前完成回执没有`selected_checkpoint_evaluation`，说明不能把那份历史回执当成当前新尾部已跑过的证明；需补当时执行源码身份，不指控已有5000步训练未发生。

### N02 预测创新TX条件与配置不一致

配置`predictor_innovation_v2_config.json`称两臂都读取`(Fb_TX,P(Fb_TX))`。实际`matched_batch()`只传：

```python
model.receive_training_residual_sample(residual, ptx, prx, ...)
```

Encoder接收`[residual,ptx]`，没有单独的Fb_TX。两臂确实共享Prx起点、P已使用相同canonical状态13dB/[1,1,8]，这两点修复有效；但原始Fb条件丢失。

若研究“完整pipeline比较”，允许保留并按实际接口命名。若隔离“残差定义”的作用，应使两臂拥有相同的Fb/P条件与相同预测后起点，使用同等E/D并明确原样/重参数化暖启动。不要让候选和控制接收不同信息后，将差值唯一归因于残差可压缩性。

### N03 线性测量控制包失败后，源测量支路破坏基础

文件：`E/mechanisms/src/latent_mechanisms/linear_measurement.py`，`receive_projection()`和`main()`评测部分。

控制包失败时`receive_projection()`返回零measurement。main仍计算：

`delta = estimate - Fb_RX @ A` （source臂）

于是：

`F_hat = Fb_RX - W * (Fb_RX @ A) @ A.T`。

这不是“没有增强观测就保持基础”，而是删去一部分已有信息。例：基向量[1,2,3,4]、A投前2维、W=.6，无有效观测时变成[.4,.8,3,4]。residual臂返回0修正，导致控制失败时两臂不对称。

修复：对“已经构造完整的delta”乘`control_ok`，或者失败直接回退原基础。资源仍全额计费；不能删除失败样本。分别测试header失败、范数包失败、合法zero-norm码和CRC误接受等可区分状态。

### N04 线性W校准只累计最后一个seed

同文件main的`for method`与`for ni,seed`同缩进，位于seed循环之后。数据batch走3次，W拟合却只读取最后一次的`noise/Fb_RX/status`。结构探针按当前5SNR、3seed、250个batch得到2500次方法累积，不是预期7500次，实际只使用seed2003。

修复：将方法处理和num/den累计放入噪声循环中；每个SNR/臂记录实际源图数、seed集合、有效控制包计数、num、den。W全局汇总后一次除法/clip，不回到batch比值平均。旧W和对应新方法输出应重新拟合/评测，不重训A/B。

### N05 线性脚本提前发出完成回执

`for source_index in range(100)`内每处理一源就写`LINEAR_MEASUREMENT_DEVELOPMENT_COMPLETE`、`sources=100`。首轮仅30行即显示完成，进程中断会留下误导回执。

修复：循环中写RUNNING进度与原子partial CSV；只有在全部源、方法、SNR、seed唯一覆盖与行数检查后，原子发布completion。现有v5确有3000行，不能仅因这一风险宣称它不完整。

### N06 配对程序仍需最后一道身份检查

`followup/.../policy_development.py`的增强聚合已拒绝重复完整键，并平均3噪声；这是旧72差值根因的实际修复。但是数字选择路径另写了`source_seed`聚合，对同键多行仍取均值；双方最终未显式核对seed_keys和preprocessing集合。

修复：两个方法先用同一校验器按完整键唯一对齐，再共同聚合。不要左右各自实现一套不同严格程度的逻辑。现有修复后的配对数值可按其真实数据范围报告，不把这个防御性缺口当成历史已混算的证据。

### N07 计时主错误已修，验收仍不足

`timing_clean.py`现在使用实际base、canonical channel_apply、单次噪声，正确性比v2明显改善。但`verify_reconstruction()`仍只核4个source0数字工作点，没有覆盖refiner/512/1024；数字clean_digital先输出CPU，再转GPU，最后又转CPU；只有一个warm-up路径，缺逐方法稳态检查。

修复：每个计时方法与正常质量路径逐项比对波形/观测/Fb_RX/RGB；移除无意义CPU-GPU-CPU往返；各法充分预热并记录重复与轮换顺序。不要以小于毫秒的差推断机制速度优势。原计时可作指定脚本测量，不应直接升级成完整最优系统成本证明。

### N08 恢复/实际执行绑定仍需范围化补齐

预测创新与适配入口存在读取latest路径却不完整核对SHA、配置与源代码绑定的问题；适配初始模型读取所选记录后未在此入口显式对比实际父文件hash。当前正常跑通并不证明故障输入会被拒绝。

对这两条续训脚本复用原B较完整的run registration与resume保护，不重新造一套平台。实际执行代码与历史completion不完全对应的项明确保留状态。共享编译/熵码此前修复按对应测试保留。

## 5. 现有结果该怎样解读

来源：`results/latent_followup_runs/integrity_fix_v1/paired_vs_enhancement_repaired.json`，100 development源、5SNR、3噪声，source-level bootstrap。这里读取已发布区间并核对符号/均值，不宣称本地独立重算所有bootstrap。

原1024增强减去同N4084 raw质量策略/Dc：

- PSNR +0.562036 dB，CI [+0.487914,+0.635246]；
- LPIPS -0.000303，CI [-0.001985,+0.001424]；
- DINO -0.019159，CI [-0.024425,-0.013957]。

这支持PSNR增量，未确认LPIPS优势，并显示DINO代价。不能称全指标支配；CI跨0也不等于已建立非劣性，需事先定义容差才可作非劣检验。

原512增强对同N3572 raw质量策略/Dc：PSNR+0.263914；LPIPS+0.009159；DINO-0.030959。它是取舍点，而非另一条共同赢家。

Decoder适配公开均值22.2190/0.131421相对同机会续训控制22.0985/0.135576有正面迹象，但其历史optimizer影响、相应数字Decoder参照、运行身份要按已核实范围解释。不将新Dc固定m9的低SNR均值冒充完整最优数字策略。

预测创新v3完成回执提供的是5000更新与4000步校准selected，utility约0.02758638/0.02756025，相对仅约0.095%的校准目标差；没有已核对的新development质量/成本结论。当前main尾部bug先修，不能据此说预测创新已稳胜或已失败。

线性v5有3000行和范数费用，但本轮未下载并独立聚合完整CSV，不给其编造总体均值；控制失败和W覆盖错误影响范围需要重评。

## 6. 很关键的结构事实：简单换残差不一定产生新方法

当前EnhancementEncoder第一层为线性Conv，输入`[(F-Fb)/s,Fb/s]`。令两块核为Wr、Wb：

`Wr*(F-Fb)/s + Wb*Fb/s = Wr*F/s + (Wb-Wr)*Fb/s`。

把首层第二块核换为Wb-Wr，改读`[F/s,Fb/s]`，后续trunk、projection、normalize均相同，则整个Encoder前向相同。本次float64随机Conv验证最大差1.33e-15。

这说明两种接口的函数类在这里相同，不保证优化轨迹相同。残差表达可以提供有限模型的优化偏置，但不能仅凭“相减以后范数小”声称降低了条件信息量或产生独立通信机制。

因此不建议再把“带同样基础条件的直接F”当成必须长训的重大新分支。真正不同的是：

1. 去掉TX端VAR补全，只把数字前缀先验留给RX；
2. 将全部总资源交给纯连续F传输，判断数字基础是否值得；
3. 改变实际传输方向/功率/保护的分配；
4. 从无效观测无法提供的信息中，采用明确的纠偏与回退规则。

## 7. 最缺的一条强对照：同F、同Dc、同总预算的纯连续通信

`F`为32×16×16=8192实坐标，N4084提供8168实信道坐标。这个总预算已接近直接传一个完整连续latent的坐标数。

这不是说直接传F无损或必胜：仍要付噪声、功率归一化、幅度信息及有限映射损失。但它说明不能只与数字m9比较，就证明3060数字+1024连续是最好的分工。

新增强对照应复用同F、同Dc、同训练集和信道，把全部4084 uses用于合理训练的连续E/R，不用VAR补全、不用类别字段。实际输出必须是8168实坐标，不能对仅支持512/1024的旧head作未经训练截断。使用固定预算读出/训练中已定义的掩码或恰长投影，归一化只按实际发送坐标计费。

这一个控制将很快区分：混合的优势是否集中在低SNR/较小预算，或者连续基础就更划算。它是必要的科学定位，不是要求所有方法全SNR都赢。

## 8. 文献怎样指导下一步，而不是堆模块

### HDA-DeepSC

公开v2的§III-A/III-C把数字基础、连续特征残差和融合解码联动，并采用分阶段训练/联合优化。它支持当前HDA骨架，但不证明现有VAR固定前缀+小Encoder最优，更不要求现在重训VAR或加入扩散。

可借鉴：把纯数字、纯连续和混合端点一起放到资源—失真曲线上；在已有冻结codec上只训练通信模块与必要Decoder适配。损失中若某项只依赖冻结F和Fb，它对通信E/D梯度为0，不应机械照搬。

### SharpCast / SK-Cast

SharpCast保留结构、数字化高能内容，再为模拟部分分配资源；SK-Cast区分残差压缩丢弃误差与信道噪声，并根据条件选择模拟映射/数字基础。

可借鉴：当前已测(4016D,0A)/(3504D,512A)/(2992D,1024A)，先在校准上选定总N/E内的策略。它是已有点的系统收口，不是又训练一个调度大网络。确需增加m7/m9混合时，它们的残差分布变化，必须有相应训练，不能把m8权重未经适配直接拿去代表新模式。

若尝试增强组功率，需保留原Encoder自然幅度分配、同分组固定能量、SNR条件分配对照，噪声不随分组功率被错误缩放；逐组归一化会丢掉相对幅度，必须将这个设计差异计入解释。

### Hybrid Semantic-Complementary Transmission

原文用源图和本地重建差得到逐实例低秩补充，传输投影与投影矩阵并在RX纠偏。当前随机QR矩阵不是其逐图最优投影，更不是VAR先验导出的重要方向。

当前A有1984列、总维8192；随机正交子空间在对随机A取期望时仅捕获1984/8192=24.21875%的固定向量平方能量。不是实测图像提升比例，也不是神经恢复上限。

可先用训练集残差二阶矩选择共享投影/PCA（均值处理需明确）；与相同维度的随机A比较，沿用已计费范数包。先固定长度+同物理资源检验，最终评价经过同Dc的图像，而非只评价解释方差。

### DAC-Mobi

其利用接收侧相关图像解开模余数歧义，提示“源端不需要生成完整预测图，RX先验仍可帮助恢复”的路线。模解包对边信息误差敏感，当前1dB基础不可靠，故不优先直接移植coset。

更直接的迁移是轻量TX条件增强：TX用F和真实前缀累计latent编码，RX用实际VAR补全结果恢复。TX不再运行一次完整VAR。此为建议的新实验，不是原论文已验证的当前系统。

## 9. 最合理的下一步次序

### 步骤一：仅修闭环错误，不重新组织目录或全量A/B训练

N01/N03/N04/N05修完，新增对应反例测试；selected加载与norm失败回退先CPU再少量实权重检查。补P选中模型评测，重拟合W并重评线性。计时全臂一致性检查只覆盖实际需要的方法。

### 步骤二：用现有1024做一次误差来源诊断

固定模型、来源和条件输入：交叉比较实际/正确基础，以及实际/去掉增强噪声。无噪声只取消物理噪声，神经SNR条件留在已训练支持点，不喂无穷大数。D_c(F)另列连续表示参考。

该2×2表回答剩余差距主要随基础错误、增强信道噪声还是有限编码而变化。oracle只诊断不进主排名；不能把dB差看作独立物理误差百分比。

### 步骤三：建立纯连续F同预算控制，冻结一套实际资源选择

先N4084，不平铺所有预算/新骨干。复用已有数字/混合工作点在校准冻结策略，并与纯连续控制同源评测。如果纯连续高SNR更好，不是研究自动失败，而是要把混合有价值的区域、基础服务和成本说清楚。

### 步骤四：只增加一个真正改变计算或通信的主候选

优先选“TX不做VAR生成、RX使用真实前缀的next-scale补全作为边信息”的条件连续编码：

- TX：F与真实前缀累计表示F_prefix（保持官方十尺度量化映射，只累积前m项）；
- 数字链：原m8不变；
- 增强：E(F,F_prefix)输出1024 symbols；
- RX：R(y_A,Fb_RX,SNR,status) -> Dc。

对照当前完整TX VAR残差1024，并提供等额训练机会/明确暖启动。若质量接近而TX计算确实下降，属于可解释的质量—资源—计算收益；若显著变差，保留完整TX prior，不强行套低复杂度故事。

PCA投影作为便宜机制对照/次级分支即可。分组功率等上面定位清楚后再单独尝试；coset、扩散、大视觉骨干暂不扩张。

## 10. 结果表和发表边界

不要求全PSNR/LPIPS/DINO、全SNR共同胜出。需要清楚回答：额外信道为什么用于此分支更划算；数字基础是否有不可替代作用；代价增加在哪里；已发表HDA和渐进压缩与当前设计差别是什么。

旧100development及已看过的1k holdout不能重新命名未见测试。方法与选择规则收口后，才使用新独立图像验证泛化。当前不需要再开无边界“找问题”的实验队列。

## 11. 主要源文件与论文索引

固定commit下重点读取：
- `phase_b/src/latent_enhancement_b/model.py`, `data.py`, `train.py`
- `src/latent_enhancement/latent.py`
- `evaluation/src/latent_enhancement_eval/runner.py`, `deployment.py`
- `followup/src/latent_followup/decoder_adaptation.py`, `timing_clean.py`, `policy_development.py`
- `mechanisms/src/latent_mechanisms/linear_measurement.py`, `predictor_innovation.py`
- `mechanisms/predictor_innovation_v2_config.json`
- `results/latent_followup_runs/integrity_fix_v1/paired_vs_enhancement_repaired.json`
- `results/latent_followup_runs/predictor_innovation_v3_requalified_20260921/completion.json`
- `results/latent_followup_runs/linear_measurement_v5_requalified_20260921/{completion.json,projection.json,per_frame.csv}`（per_frame本轮仅抽读首个源图，不冒充完整聚合）

论文：HDA-DeepSC arXiv:2405.12580v2；Hybrid Semantic-Complementary Transmission arXiv:2507.17196v1；用户Library的SharpCast、SK-Cast、DAC-Mobi PDF。论文机制与本报告提出的迁移假设分开，不宣称复现了整篇原系统。
