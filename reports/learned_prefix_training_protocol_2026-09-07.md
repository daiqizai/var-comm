# 固定m8：全局prefix-JSCC与逐尺度条件读取训练协议

2026-09-07，用户明确要求真正训练。状态：已授权，工程梯度自检通过后直接执行，不设无训练性能门槛。

## 数据与训练预算

沿用既有ImageNet20,000训练图和独立1,000校准图，最终100图永久作为development。
新缓存只存官方完整十尺度tokenizer的前八尺度索引（255个）；原图与旧checkpoint不搬迁。
训练水平翻转由冻结VAE真正重新tokenize，不能只翻转旧token网格。按原图及翻转像素SHA检查，三角色没有精确内容交叉。
固定归一化统计只用训练prefix的码本embedding；不使用旧fhat或DINO特征作为训练输入。

两模型都从相同通信参数初始化，交替处理完全相同的图像、翻转、SNR和噪声，每个2个完整epoch、40,000图像曝光。
有效batch4，microbatch2；AdamW lr3e-4余弦至3e-5，weight_decay1e-4，betas0.9/0.99，clip1.0，FP32、dropout0。
SNR逐图均匀取1/4/7/13/19 dB。固定MSE＋0.01 LPIPS＋0.001 token CE，token CE按255位置取均值，仅为辅助。
不把DINO用于loss、超参数或checkpoint选择。无论初始/第一epoch质量如何，两版本都执行完固定预算。

## 全局发送与对应接收器

发送器只查冻结码本e(r1:8)，255×32 embedding经全局稠密混合与Transformer残差成为187×32=5984实数，
一次发送2992个复符号；不是把token整数当连续数，也没有独立数字尺度包。
全部data统一逐图功率归一化，mean complex power=2，real noise variance=1/gamma；归一化因子不提供给接收器。
另有原68-use class/mode/CRC/tail header，总3060；data不再追加FEC/CRC。
接收端使用真实译出的class，CRC/格式失败灰图；mode必须为8。训练同样模拟header，失败样本图像损失仍计入且无数据梯度。

- parallel：从整份y一次输出255个prefix概率，之后同样VAR生成第9/10尺度。
- next_scale：每尺度读整份y，同时读当前自己估计的状态及冻结VAR的完整下一尺度概率，再输出当前尺度。
  融合是依赖接收特征/状态/不确定性的可训练gate和特征网络，不是固定MAP权重。

两者使用相同有效通信模块和参数量1,508,750，baseline的融合输入来自y自身的初步码本亲和与特征，不填闲置参数。
两个发送器均参与联合训练，训练后波形可以不同；比较的是结构参与联合训练的系统增量，不是固定y的纯接收器消融。
所有receiver步骤复用一次收到的全局y，不重复计免费信道；没有中间CRC可靠门槛。

## 真正的图像梯度

VQ encoder/码本/quantizer尺度映射、VAR、官方图像decoder全部冻结参数并保持eval；通信模块单独train。
训练不调用旧no_grad推理入口。维护有梯度的VAR输入/KV和非原位累计latent，冻结参数不关闭输入梯度。
硬选择前向严格argmax：embedding=hard_embedding+(soft_embedding-soft_embedding.detach())。
这只是有偏ST梯度，不把argmax称为真正可微；第9/10尺度使用相同ST，评测前向仍相同argmax规则。

next_scale的GT历史概率在前40%更新从1线性退至0，剩余60%（含整个第二epoch）完全使用噪声下自己的历史。
即使teacher warmup，最终图像分支也重新使用自己的预测prefix，不能把GT历史直接当作重建prefix。
校准/评测始终无teacher，receiver接口不需要源tokens；源tokens仅为发送输入、辅助监督或明确训练teacher。

工程自检的临时通信模型在两张校准图上检查梯度后直接丢弃，不保留任何更新权重；正式模型从相同固定随机种子重新构建。
自检未依据图像质量选择架构/参数，检查了image-only MSE/LPIPS对E/D梯度、suffix通过VAR到prefix的梯度、冻结权重、
训练/评测硬前向一致、官方重建一致及功率。正式保留模型的所有梯度更新仅来自训练20k。

## 选择、评测与强对照

只在独立校准1000上按五SNR平均MSE＋0.01 LPIPS选择epoch1或epoch2，不含token CE或DINO；不从开发100图选checkpoint。
两模型完整训练结束并封存后，评价100图×3原噪声×1/4/5/6/7/13/19 dB，报告PSNR/LPIPS/DINO及token准确率。
主汇总SNR1/4/7，先平均每图噪声与主SNR再按source-image bootstrap（10000次，seed=2026090715）。
同时列固定m8数字链、旧阈值整帧自适应，以及现成本地感知DeepJSCC。

DeepJSCC固定使用旧lpips_0p01、校准选中epoch2的checkpoint，不用开发100图挑模型。
它有COCO预训练加同20k/2epoch感知微调历史，并且输入完整图像；不能称与新通信模块相同从零训练成本。
它不使用class/mode，故允许全部3060 uses发送数据，不强迫为不需要的header浪费68 uses；也不免费给它类别。
其旧通道方法存在半方差口径，本轮显式取active coordinates、归一化、加sigma=1/sqrt(gamma)噪声，
不调用旧半方差transmit。总功率/信道次数与新方案相同，波形不同。

最后仅做一次7 dB/seed2001的固定错配message检查（保持实际header），检验模型是否利用源相关数据；只作诊断，不选择输出或重训。
这是一次固定预算训练实验，不声称收敛至最优、全新正式测试或新颖性成立。记录损失曲线、teacher比例、功率、参数与源文件SHA。
本轮不重启旧单点state纠正，不训练tokenizer/大VAR/图像decoder，不新增网络/尺度/超参数搜索。
