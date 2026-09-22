# 接口对照最终评测与统计账本

2026-09-12，首次新模型development推理前固定。补充interface_protocol，不改运行中的网络、loss、选择规则或训练数据。

## 方法与样本

九个训练臂分别使用相应里程碑完整calibration LPIPS选中的模型。评测前必须取得该里程碑的校准配对/Adam/选模审计；不按development调整选中step。

保留七项冻结参照：原2000最佳三结构、同WeTok数字8PSK/FEC、旧数字固定m8/自适应、感知Deep。总16方法×100源图×7SNR×3噪声=33600无线行。仍为永久development，不是新holdout。

新方法原生源8192 raw bits，0 header+3060 data，总能量6120。旧数字分别列68+2992，Deep为0+3060；各方法总资源不变。旧数字的coded bits按原实际发送账本继承，不能把raw bits或连续坐标当成互换单位。AWGN噪声用已校正的big-endian SHA种子，与旧控制逐源/噪声严格配对。

## 接口与图像保真

hard_identity与hard_bounded评测均为native±1；continuous_mean直接将tanh(logits/2)交给冻结Decoder，**不再sign/argmax**。记录接口标签、输入幅值/饱和率和特征误差。continuous的thresholded-logit BER仅诊断，不称恢复了同样的数字源bits。

保存所有实际无线输出，不因质量差、CRC失败或header失败而删除样本；旧数字灰图及失败候选照原规则保留，不用真值修复。模型不新增接收观测、类别或后缀。正确native重建和noiseless nominal19映射另列，不混入无线排名。

新图保存lossless float32。旧七方法逐图校验原archive及图像SHA，只读引用既有绝对/工作区相对路径，不大批复制历史图像。统一GPU评测器重新计算新图及旧图的PSNR/SSIM/LPIPS/DINO；旧指标必须在预定数值容差内复现，不能为使检查通过而放宽容差或更改旧结果。

## 统计与时延

每个进程session的首个未提交source重新预热视觉/通信模型，续跑不能把冷启动混入该source时延。每个source前后核查无外来GPU计算，出现竞争只停止自身评测、保留已提交source，不结束他人任务；未提交source后续完整重放，不按图像好坏剔除。session耗时包含异常/续跑的已记录时间，非正常终止留下的未完整计时明确作为下界。

新补充：已核实旧5/6dB固定支持Deep对照存在，并在本轮新模型development推理前登记600行单列表。按同source/raw_noise/实际SNR重新生成，验证旧像素与指标，见`deep_support_supplement.md`。原16方法/33600行及主1/4/7不变；另给162个补充配对区间。质量图用菱形标出合理支持点，不以原Deep的未支持条件异常制造优势。

主区间仍1/4/7dB，并完整报告1/4/5/6/7/13/19dB。每图先平均三个噪声，再平均指定SNR；仅对100个source-image向量paired bootstrap，不能把33600行当独立样本。固定10000 bootstrap重采样及既有seed。

63组对照：每结构三个接口差；每接口三种结构差；每新臂对同结构原2000最佳及四个系统参考。六指标（PSNR/SSIM/LPIPS/DINO、相对native的LPIPS额外损失、严重失真率）×8个SNR组，共3024配对区间。主机制结论仍看同接口conditioned−no_history，不能以跨接口差冒充条件结构增量。

新模型计时包括在线视觉Encoder/量化/源特征转换、通信Encoder，以及完整接收器和冻结图像Decoder。固定首样本预热；GPU串行无其它计算任务时评测，报告mean与P95。旧系统历史时延只列参考，不与新实测做严格时延排名；若后续声称省时延，必须再统一重测所有相关强对照，当前不凭空补数。

训练GPU小时单独取真实日志，纳入共同2000历史并区分后续配方预算；最终比较不只按选中step少计全部已投入训练。当前只准备执行器，不因评测代码完成而声称方法或整个研究完成。
