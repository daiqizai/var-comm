# 固定m8全局prefix-JSCC训练结果

2026-09-07，Asia/Shanghai。两版本完整训练、独立校准选择、10500行开发评测及复核均已完成。
结论：**next-scale条件读取在相同通信参数/训练预算下有明确增量，但当前学习通信链仍未超过原数字自适应系统。**
这是首次本轮训练的有限预算结果，不是全局最优、收敛充分或新颖性声明；不延续旧无训练候选的停止结论。

## 1. 这次确实训练了什么

- 只训练通信Encoder、Decoder和融合模块，每版本**1,508,750**个参数，初始state SHA完全相同。
- 两模型交替处理相同图像、翻转、SNR和噪声，各2个完整epoch、10000次optimizer更新、**40000次训练图像曝光**。
- 训练20k、独立校准1k、最终development 100，均沿用旧划分；原图/翻转像素SHA没有精确内容交叉，训练覆盖1000类。
- VQ encoder、码本、量化器/尺度映射、VAR和官方图像decoder参数冻结；训练只查前八尺度255×32码本embedding。
  不是把整数token直接当连续输入，也没有使用旧fhat或DINO特征。
- 全部prefix联合映射到5984实数（2992复符号），统一逐图归一化平均复符号功率2。
  原header占68 uses，总计3060；real noise variance=1/gamma，data没有额外FEC或CRC。
- 基础版一次读取y恢复255个位置；next-scale版每尺度都读完整y，并结合自身估计状态与冻结VAR概率。
  融合权重和特征处理可训练；两个发送器随各自接收器共同训练，训练后波形可以不同，不能称为固定y的纯receiver消融。

网络初始为全局稠密正交混合/回投影加Transformer残差；两个版本都有相同有效融合模块，没有填闲置参数凑规模。
基础版融合来自y的初步码本亲和与特征，而不是免费使用VAR的prefix预测。

## 2. 图像梯度与自身历史

训练损失固定为**MSE＋0.01 LPIPS＋0.001 token CE**，token CE按255位置取均值，仅作辅助。
DINO没有加载进训练loss或校准选择。

训练前向不调用旧no_grad推理入口，使用非原位累计latent、有梯度的VAR输入/KV和硬前向ST：
`hard_embedding + (soft_embedding - soft_embedding.detach())`。
前向仍是argmax选中的真实码本embedding，第9/10尺度同样用argmax；ST是有偏梯度近似，不把argmax说成真正可微。

工程自检确认：MSE和LPIPS分别对E/D有非零梯度，suffix能通过冻结VAR传回prefix；训练/评测硬前向像素差0。
临时自检模型的更新全部丢弃，正式训练从固定种子重新构建，没有保留自检权重。
正式保留模型的梯度更新仅来自训练20k，VAE/VAR/LPIPS前后SHA完全相同。

next-scale的teacher概率前4000更新从1降至0；实际7996/40000次样本使用训练teacher上下文。
**最后6000次更新、整个第二epoch均完全使用噪声下自己的估计历史。**
即使warmup，最终图像也从自己的预测prefix生成，不直接渲染teacher GT状态；校准/评测无teacher。

## 3. 校准选择

只在独立1000图、1/4/7/13/19 dB上按MSE＋0.01 LPIPS选择checkpoint，不含token CE或DINO。

| 版本 | Epoch 1校准目标 | Epoch 2校准目标 | 选择 |
|---|---:|---:|---|
| 基础parallel | 0.023304 | 0.02289528 | Epoch 2 |
| next-scale | 0.020843 | 0.02077703 | Epoch 2 |

两个版本都执行完整预算后才开展development评测。训练阶段约8956.89秒（约2小时29分钟），峰值GPU allocated约6.16 GiB。
两epoch和这一个loss/结构配置不足以证明收敛至最优，也没有根据开发结果继续调参。

## 4. 预定主区间结果

主区间为1/4/7 dB；100图各先平均三个噪声和三个SNR，再做10000次source-image paired bootstrap。
每方法900条传输记录不是900张独立图片。完整七个SNR结果保存在`summary.csv`。

| 方法 | PSNR ↑ | LPIPS ↓ | DINO ↑ |
|---|---:|---:|---:|
| 基础VAR-prefix JSCC | 17.42431 | 0.362668 | 0.650595 |
| **next-scale条件JSCC** | **18.14048** | **0.330314** | **0.718470** |
| 固定m8数字链 | 18.65316 | 0.214927 | 0.844922 |
| 原整帧数字自适应 | 19.62665 | **0.183763** | **0.886743** |
| 感知DeepJSCC | **24.37834** | 0.205291 | 0.544427 |

**next-scale−基础版：**

- ΔPSNR=**+0.71616 dB**，95% CI=[+0.55341,+0.89445]。
- ΔLPIPS=**−0.03235360**，95% CI=[−0.04079135,−0.02418770]，相对改善**8.921%**。
- ΔDINO=**+0.06787497**，95% CI=[+0.05135241,+0.08440882]；DINO未直接优化。

因此，这次确实观察到**next-scale结构参与联合训练的增量价值**，不是只写了一个新模块或只提高token正确率。
但相对原数字自适应，next-scale仍低1.48618 dB、LPIPS高0.14655、DINO低0.16827，三指标均未胜出。
相对感知DeepJSCC，next-scale的DINO高0.17404，但PSNR低6.23786 dB、LPIPS高0.12502；这是权衡，不能说全面胜过DeepJSCC。
相对相同源prefix的固定m8数字链，也未取得总体优势，不能把差距全部归于数字自适应在高SNR发m9。

## 5. 代表性SNR与剩余差距

| SNR | 基础版 P / LPIPS / DINO | next-scale P / LPIPS / DINO | 原数字自适应 P / LPIPS / DINO |
|---|---|---|---|
| 1 dB | 16.57277 / 0.437801 / 0.480196 | 17.34518 / 0.391760 / 0.604419 | 18.01946 / 0.236670 / 0.848680 |
| 4 dB | 17.53569 / 0.348224 / 0.695598 | 18.28919 / 0.316873 / 0.751230 | 19.69081 / 0.177149 / 0.889749 |
| 7 dB | 18.16448 / 0.301979 / 0.775991 | 18.78707 / 0.282309 / 0.799760 | 21.16969 / 0.137471 / 0.921798 |
| 19 dB | 18.63208 / 0.283202 / 0.798395 | 19.04164 / 0.262843 / 0.818651 | 21.24373 / 0.135896 / 0.923859 |

19 dB固定m8数字链的LPIPS为0.177141，仍明显好于next-scale的0.262843。
这说明当前差距不能只归咎于“冻结m8表示不够”：学习通信映射及其训练/硬选择仍留下很大差距。
没有仅凭这一结果断言具体优化原因，也没有声称高SNR误差都是不可消除的表示上界。

7 dB prefix token准确率：基础85.58%，next-scale81.09%；19 dB为92.15%和87.75%。
next-scale在总体token准确率更低时图像三指标仍更好，进一步说明不能用token准确率代替图像目标。
本轮未单独证明这种差异的逐尺度因果分解。

## 6. 强DeepJSCC对照与5/6 dB条件检查

固定复用本地感知DeepJSCC的lpips_0p01、独立校准选中epoch2 checkpoint，不从development挑权重。
它有COCO预训练加同ImageNet20k/2epoch微调历史，约30.75M参数，输入完整图像；
不是与新通信模块同样从零训练，也不是声称复现某篇官方DeepJSCC论文。
它不需要class/mode，所以允许全部3060 uses发送数据，不强制为不需要的header支付68 uses；不免费给它类别。
显式使用real noise variance=1/gamma，未调用旧半方差transmit方法。

完整七点曲线发现：该checkpoint的原始SNR条件通路在未训练的5/6 dB异常。
代码是高频Fourier特征＋MLP，不是简单查表；本次不能把它在未见条件上的退化算成VAR通信优势。
因此在主结果冻结后追加一次只读强对照检查：**真实SNR仍5/6，NN条件按最近已训练支持点固定为4/7**，
不训练、不看原图选条件、不改变噪声/功率/预算。原始结果保留，1/4/7主区间完全不变。

| 真实SNR | Deep原始条件输入：P / LPIPS / DINO | 固定支持条件：P / LPIPS / DINO |
|---|---|---|
| 5 dB | 11.26609 / 0.736786 / 0.055080 | 条件4 dB：24.80432 / 0.195358 / 0.606406 |
| 6 dB | 23.64389 / 0.307243 / 0.501187 | 条件7 dB：25.01476 / 0.173682 / 0.618134 |

补充控制下，next-scale在5/6 dB仍然PSNR和LPIPS更差、DINO更高，没有总体胜利。
这是post-evaluation工程对照，不包装为事前注册的新方法或已证明最优的SNR插值。
未做Fourier频带消融，不能将异常归因于某个已单独验证的频率。
原`training_and_quality.png`保留原始条件曲线，其中5/6 dB的Deep异常须结合本节阅读，不能据图宣称这些点获胜。

## 7. 不是只靠类别猜图

固定7 dB、seed2001，将data消息改为下一个源图的编码，header仍来自当前源图；只作诊断、不选输出、不重训。
真实body→错配body，基础版LPIPS上升0.46465，next-scale上升0.47839；next-scale的PSNR下降9.43906 dB、DINO下降0.72026。
这表明输出显著依赖源相关接收数据，不是忽略y只依靠class/VAR生成。
错配同时引入类别与body矛盾，因此不能把它当作已经隔离类别信息的纯实例信息量测量。

## 8. 独立复核与交付

- 训练预算/选择审计：每版本10000更新、40000曝光，同初始化、相同header实现/更新进度；最后6000更新无teacher。
- 所有epoch的校准目标独立重算，只有MSE+0.01LPIPS，未包含DINO/token CE；选中checkpoint SHA核对。
- 10500行物理预算/噪声、7348张独特图像全部重算指标。
- 4200张学习prefix图像用旧官方接收规则从已预测hard prefix重建，像素差0；80个学习receiver重放、40个Deep重放和200个错配消息重放均一致。
- 最大PSNR/LPIPS/DINO复核差3.785e-6/1.193e-7/6.239e-7，配对区间最大差3.553e-7，物理噪声最大差6.267e-7。
- 两项主审计均通过；上述数量不把后来600条Deep支持条件检查冒充事前主审计的一部分。

主要路径相对VAR_COMM：

- 训练协议：`reports/learned_prefix_training_protocol_2026-09-07.md`。
- 数据绑定：`outputs/VAR-PREFIX-TRAINING-DATA-001/`；旧图像和大模型未搬迁，新缓存约31 MiB。
- 训练曲线、checkpoint与选择：`outputs/VAR-PREFIX-JSCC-TRAIN-001/`。
- 完整结果：`outputs/VAR-PREFIX-JSCC-EVAL-001/summary.csv`、`paired_quality.csv`、`shuffled_body.csv`。
- 汇总与图：`outputs/VAR-PREFIX-JSCC-SUMMARY-001/primary_summary.csv`、`message_dependence.csv`、`training_and_quality.png`。
- 审计：`outputs/VAR-PREFIX-JSCC-TRAIN-AUDIT-001/audit.json`、`outputs/VAR-PREFIX-JSCC-EVAL-AUDIT-001/audit.json`。
- 支持条件检查：`outputs/VAR-PREFIX-DEEP-COND-SANITY-001/`。
- parallel选中checkpoint SHA：`c7fc03f107baca049c0974318eb8b2a6eea8078d6693a408c27d2f9e4193005f`。
- next-scale选中checkpoint SHA：`4af01e2a2fc048f5fcd27d1b5747d885b13b0c1e22baa33594dfcd0df8f8f659`。
- 训练receipt SHA：`04659564185491863ecb5e3fd773e0541521575b43b951d905d6b8896e9349bf`。
- 评测receipt SHA：`4b21e10d28a1bd9c730081b5879ca8501f234c58e66395f34d8c14e9b7520d68`。

**本轮真正训练的任务已完成。保留next-scale联合训练的增量证据，原数字自适应仍为主基线；不自动改loss、扩大网络或续训。**
