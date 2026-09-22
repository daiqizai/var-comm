# A0/A1初始通信比较：执行前协议

依据工作区`next_scale_communication_research_brief_v2.md`（SHA256见配置），先研究新codec的重建优势能否在有限AWGN资源中保留，不将更好的视觉模型本身作为通信创新。

## 问题、机制与否定性对照

问题：8192个原生源bits在3060复信道使用下并未可靠到达，接收表示与冻结Decoder之间可能产生大量额外失真。
A0：全局联合编码完整WeTok量化后Fq，单次从完整y恢复32×16×16的native符号并解码RGB。
A1：相同全局发送接口，4×4→8×8→16×16条件读取；前两层是对正确Fq面积下采样的连续通信状态，最终才硬符号化。自身恢复状态可改善对相同y的解释，但并未增加源信息或新波形。
匹配控制：同参数single_pass、multiscale_no_history、multiscale_conditioned，共用编码/读取模块形式、相同训练机会。无历史控制同样读取三个网格并接受相同状态监督，但其上下文只从y取得。若条件版不优于无历史版，不能把多算几层包装为next-scale收益。
全部阶段看到完整y，不声称逐包渐进、可提前停止发送或新观察到达。暂不加功率分配、全尺度VAR、反馈、Decoder微调或其它视觉骨干。

## 视觉接口与梯度

复用指定WeTok ImageNet/stride16 EMA checkpoint和已核验官方模块。实际Fq为32×16×16的±1；16×16位置×4组×8bits=8192rawbits。按代码实际2^arange(8)的位序读写组索引，不分配2^32分类头，不复用VAR的4096类CE。
最终native sign前向严格±1，反向使用明确的identity straight-through估计；它是有偏梯度近似，不把sign变成真正可微。图像loss通过冻结Decoder到通信E/D，不能调用带inference_mode的旧decode包装器训练。
本指定配置gan_decoder=False，为确定性Encoder/LFQ/Decoder链，不添加生成噪声。冻结模型参数、检查全部EMA加载及输入梯度；原source/vendor/checkpoint及旧输出不改写。

## 每图无线账本与信息边界

新WeTok臂不需要类别、caption或逐图模式：固定协议无header，3060 data=3060 total uses，平均复符号能量2，总能量6120，AWGN每实维方差1/gamma。
不为了沿用旧68-use header硬加无用类别；与旧VAR的68+2992和Deep的0+3060分列，但总N/E相同。固定frame大小、分辨率、码率及名义SNR为预共享条件，不通过隐藏mask/归一化系数/逐图seed输送源信息。
发送端只有Fq及名义SNR；接收端只有y及相同SNR，不接触源图、Fq真值或接收噪声。全局功率归一化不把其逐图系数免费送给RX。
对应WeTok数字参考采用固定8PSK+原卷积码族，8192payload+CRC16+tail6全部进入有限coded预算9180bits=3060×3。8PSK每个符号恒能量2，无需传逐图QAM幅度归一化系数。失败候选仍按合法native bits渲染，CRC只标可靠性，不用真值修复；这不是最优数字方案宣称，后续可扩强FEC/熵编码参考。

## 数据、训练和资格

原20000图训练、1000图独立校准、100图永久development；检查ID不交叉。原图只读，缓存原生组索引与两种真实翻转编码，不复制大型原图/权重。首次默认fp32、microbatch1/effective4，优化实现须先记录验证，不追求占满显存。
三臂通信参数同一初值，Adam从头，相同数据/翻转/SNR/噪声。先2000步bits BCE+0.1 coarse-state，再MSE+0.01LPIPS+0.01bits BCE+0.01coarse-state。state只监督4/8连续目标，不重复加入一套完整Fq损失或DINO。
初始5000更新是检查里程碑，不是训练上限；之后按5000步完整校准检查走势，自主延长且比较臂获得相同更新机会。学习率共同调度，不针对development改权重。校准仍改善则继续；过拟合或充分收敛后分析并登记下一假设。
完整校准在表示阶段结束和每个里程碑；每1000步固定100图监控。选平均LPIPS最低完整候选，平局取早；本轮不通过PSNR门槛排空对照后用warmup代替。PSNR各SNR均报告，超过A0下降0.2dB时必须标为取舍而非无代价质量增益。
不合格/异常模型如实报告，不以不可部署或弱fallback制造结构收益。原B及新旧VAR数字结果只作不同表示/历史的系统参考。

## 评测与归因

统一PSNR/SSIM/LPIPS/DINO；主1/4/7dB，其余点全部保留。每源图先平均三个噪声及声明SNR再bootstrap。严重失真事件预定为LPIPS相对该图正确native重建增加≥0.15，仅事后评测，不用于RX选优。
native完整重建是正确Fq参考，不是已免费无线到达，也不是理论最优质量上界。额外区分nominal19dB条件/noiseless learned channel的映射损失与真正19dB噪声损失。
先比较同WeTok条件/无条件，再对WeTok数字、旧数字m8/自适应和感知Deep。只胜旧VAR不能归为通信创新。Deep5/6既有条件支持异常单列，SGD高预算不进入等资源优胜排名。
报告完整TX/RX时延、冻结codec规模、缓存开销、训练GPU时间与不同模块成本。新holdout在方法/选择规则/强对照冻结后再启用。

## 原论文与实现

WeTok原论文为arXiv:2508.05599；2026-09-12网页显示v3，但本轮仍锁定已实测的源码caa2ad7e709cdabe8432bead448f7514def13919与指定checkpoint，不将后续版本混入。源码Apache-2.0；HF权重卡无独立许可元数据，不据此推断权重商用/再分发权限，本轮只复用已授权本地研究资产。
本轮A0是工程基线，A1尚为候选，不宣称多尺度条件结构/JSCC/Transformer首次提出；形成机制后继续按v2核查最近通信原论文及对照。
