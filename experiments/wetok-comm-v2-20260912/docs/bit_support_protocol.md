# 训练目标修订：native bit-preservation单变量对照

本方案在原5000步里程碑、完整校准、统一development与无噪声诊断完成后登记。它是新的训练配方假设，不改原结果，也不承诺提高质量。

## 已定位的问题

原三臂的完整calibration LPIPS从2000步约0.287恶化到5000步的0.422/0.596/0.443，bit错误也增加；选模因此都保留2000步，不是warmup失败回退。
同图development的正确native重建LPIPS0.08681；选中模型的noiseless/名义19dB输出约0.210，仍有约10.1% bit错误。19dB无线输出与noiseless相近，说明当前不能把额外损失主要归因于AWGN，更不应先加功率分配。
2000与5000共同起点的固定训练探针均显示image梯度可能远大于0.01权重的bit梯度。5000探针有很大的梯度长尾，尤其条件版；这只定位风险，不能由单batch推断总体，不能把提高bit权重当保证。

## 假设和唯一变化

假设：冻结native Decoder前的真实源bits约束过弱，使image-ST优化偏离有用的原生表示。仅把joint的bit BCE权重0.01提高到1，保留MSE=1、LPIPS=0.01、coarse-state=0.01。
这是bit-preservation监督强度的整体检验，不改ST、量化规则、输入Fq、WeTok、网络、N/E、SNR、功率、前状态或数据。若λ=1仍不足以控制图像梯度长尾，结果允许为负；不得事后偷偷放宽或再加入梯度模块。

## 公平训练

对每个single_pass/no_history/conditioned，复制同一个5000步模型及原Adam state，分别执行原joint与bit_support。共六臂，optimizer必须deep-copy，不能共享moment或step tensor。
共同起点由原里程碑的不可变optimizer硬链接与SHA确认；不从仅存model的2000 checkpoint猜造Adam状态。LR均为1e-4，数据/翻转/SNR/噪声采用原global step5000起的相同序列。
首新增5000更新/臂只是修订对照里程碑；需要延长时双方同机会。微批1、有效batch4、clip1、fp32及冻结视觉模型均不变。

## 选模、对照和停止解释

所有六臂从实际共享5000起点开始新的候选集合；追加0/2500/5000处完整1000图校准，固定100图监控不能选模。最小全SNR校准LPIPS、平局取早，PSNR与其它指标按v2报告取舍，不用失败fallback。
原实验选中的2000步三臂另作历史最佳参照。因此若新配方只胜已退化的5000续训，未超过原2000最佳，必须说明只是修复退化，不称整体通信已经进步。
主因果比较是每个结构内bit_support对原joint；之后再看各配方内条件对无历史。同N3060/E6120的WeTok数字、旧数字自适应和感知Deep仍保留。
没有新holdout，没有新类别/反馈/额外观察。用PSNR/SSIM/LPIPS/DINO、严重失真和时延判断，不以bit错误单独宣布成功。

## 计算计划

按已测微批1的joint吞吐，六臂新增5000更新约3.2 GPU小时，另加完整校准/监控与评测，粗估约4–5小时。是运行计划而非研发总上限，也不授权新付费机器。
若验证无收益，优先根据noiseless映射、Decoder输入分布和梯度证据决定native/continuous或发送映射等单独修订，不循环叠加无依据模块。
