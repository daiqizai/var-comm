# 固定m8两阶段prefix恢复：已授权执行协议

2026-09-08用户确认上一轮所列定义、权重及约束并要求“开始吧”。本文件在新增训练前固定协议；不改写任何旧结果。
当前执行状态以`PROGRESS.md`及新训练目录的`status.json`为准，排队、实现和自检不等于已经训练完成。

## 起点与公平性

两个分支都从`VAR-PREFIX-JSCC-TRAIN-001/next_scale/checkpoints/epoch_02.pt`出发。
其SHA为`4af01e2a2fc048f5fcd27d1b5747d885b13b0c1e22baa33594dfcd0df8f8f659`；
同目录`resume.pt` SHA为`89fa576564ddac911da811747135ebd8a4a5a9bd0bc2bad186f563d372fec4c3`。
已只读核对108项模型状态逐张量一致，104项optimizer状态均在10000步，Adam一二阶矩有限。
resume内selections还指epoch1，是最终校准前的旧字段；不使用它重新选起点。

独立复制两个模型和optimizer，保留参数顺序、一二阶矩及步数；新增各10000更新，结束时optimizer步数应为20000。
固定lr=3e-5、batch4/microbatch2、FP32、clip1，保留原AdamW betas/weight decay，不重启warmup，不在阶段切换重置optimizer。
每个分支新增40000图像曝光。20k训练集及真实重tokenize翻转不变，按原order/channel种子加原数据epoch索引2、3继续采样。
两分支逐更新使用完全相同的图像、翻转、SNR与噪声；记录批次指纹。整个新增预算teacher=0，不复活真实历史warmup。

固定m8、255×32码本embedding输入、1,508,750通信参数、全局2992个复数据符号及真实68-use header，总3060。
功率、噪声、class/mode side information、hard argmax/ST、CE位置平均、clamp及后缀生成规则均不改。
VQ/VAR/官方Decoder/LPIPS权重冻结；仅通信模块更新。不改变温度、采样、模型容量、FEC或资源分配。

## State监督定义

令F[k]为完成第k尺度后32×16×16的累计latent，由冻结码本、原上采样及quant_resi逐尺度相加得到。
L_state = mean_k(mean_elements(((F_pred[k] - F_GT[k]) / embedding_std)^2))，k=1...8。
标准差是已有训练集统计的32维embedding_std，按通道广播；不重新估计统计量，不用逐图真值归一化。
预测状态使用自身hard-token前向和原ST反向；目标状态仅由真实r1:8在no_grad内构造，不含真实r9/r10或完整VQ latent。
目标只进入损失，不进入接收器上下文；接收API不接收真实token或teacher状态。
state项按八尺度等权平均；CE仍按255位置平均，不增加尺度CE重分权。
header失败时CE/state项为0，但该图仍占batch分母；图像输出仍为固定灰图，MSE/LPIPS不排除失败样本。
校准逐尺度误差记录实际候选的未屏蔽诊断值及header标记；不把header失败时的候选称为协议接受的恢复。

## 两分支固定loss

| 分支/阶段 | 新增更新 | loss |
|---|---|---|
| continuation | 1—10000 | MSE + 0.01 LPIPS + 0.001 CE |
| two_stage / prefix | 1—2000 | CE + 0.1 state |
| two_stage / joint | 2001—10000 | MSE + 0.01 LPIPS + 0.01 CE + 0.01 state |

阶段A只运行前八尺度，无需生成r9/r10或调用RGB Decoder；但前八尺度条件VAR仍保留输入梯度，不用整段no_grad截断。
所有图像训练和校准均使用实际hard前向。阶段B对原图计算image loss，不对无误码VQ重建计算image loss。
这些是预先固定的起点权重，不声称最优，也不根据loss值、开发集成绩或梯度监控自动修改。

## 工程自检与监控

在固定的两个训练样本上检查：原三项loss与原实现的hard输出/梯度一致，新增state目标与预测形状/标准化正确，
state对通信E/D梯度存在，prefix-only模式没有运行后缀/Decoder，冻结权重未变；自检不更新保留模型。
可在CPU先执行工程自检，GPU训练前再在实际设备复核；CPU自检不是GPU训练结果。
新增step0及step2000，用相同固定训练样本/噪声记录各项对全部通信encoder/reader参数的未加权和加权梯度范数。
监控不进行optimizer.step，不改变训练采样器状态，不作科学性能门槛。记录额外计算开销。
匹配更新数与样本预算，不声称匹配FLOPs；阶段A计算较少，单独记录训练、校准、监控耗时及显存。

## 密集校准与选择

保留独立1000张校准图，1/4/7/13/19 dB，原seed2026090714的固定噪声，不访问development选择模型。
新增step0以及1000、2000、...、10000校准；每次记录各尺度CE、TER、state误差和hard图像MSE/PSNR/LPIPS、header标记。
step0两分支模型/optimizer逐张量相同，允许共用一次确定性校准前向的结果；后续校准各自实际执行。
PSNR先由每图MSE计算，再平均；不得用-10log10(mean MSE)冒充mean PSNR。
对每个分支，在全部已登记候选中选择五SNR平均LPIPS最低且每个SNR平均PSNR不低于共同step0超过0.2 dB的checkpoint。
step0为合法候选，相同LPIPS取更早一步；若后续无合格改善则保留step0，不硬选更差结果。
无论中间性能如何，都执行完新增10000更新；保留阶段边界2000、终点10000与被选中步骤，说明被选中checkpoint所属阶段。
起点校准与旧epoch2的同图MSE/LPIPS逐项复核。DINO不加载到训练/校准代码，不参与loss、guard或选模。

## 最终开发对照和统计

全部训练完成且checkpoint选择封存后，再评测原100张development，1/4/5/6/7/13/19 dB，原三个seed2001/2002/2003。
新增两分支与未续训起点、固定数字m8、原数字自适应同时报告PSNR/LPIPS/DINO；强数字对照直接复用冻结的同噪声原结果。
不改变原数字输出/FEC，不把新校准选择规则追溯应用到旧研究。所有失败样本保留。
先按source image平均三噪声，再做10000次paired bootstrap，seed2026090715；主1/4/7 dB也先在每图内平均。
最终分别回答修订是否胜过等预算续训、是否胜过固定m8和数字自适应。未赢强对照不得包装为系统成功。
保留真实训练曲线与11个真实校准点，不用训练移动平均冒充校准；无新正式测试集。

## 输出与执行安全

新配置`configs/prefix_refinement.yaml`；新增代码只写VAR_COMM，不修改冻结训练/评测脚本。
输出分别为`VAR-PREFIX-REFINEMENT-TRAIN-001`、`VAR-PREFIX-REFINEMENT-EVAL-001`、`VAR-PREFIX-REFINEMENT-ANALYSIS-001`。
原输入、源码与新运行脚本记录SHA；已完成产物禁止覆盖。训练中断使用本轮独立resume，不写回旧目录。
启动需要至少11000 MiB空闲GPU显存；不会终止、重配或挤占其他用户进程。若GPU不足，明确标记等待，不声称已训练。
