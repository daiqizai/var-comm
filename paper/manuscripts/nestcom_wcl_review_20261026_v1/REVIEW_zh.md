# WCL 稿件与 Fig.1 技术核对

核对日期：2026-10-26。基准为当前远端 main `ff1273b2e71e0db73a8a306ef871f9c501f89ad9`，即用户原稿和最终 Fig.1 的归档提交。原始 PDF 保留不改，本目录另建可编辑、匿名的全文 LaTeX 版本。没有运行实验、神经指标或 bootstrap。

## 判断

实验数字与图表之间没有发现实质矛盾。稿件的论证顺序成立，但“基本可以投”不能仅由五页和数字正确来判断：实现描述、证据适用范围及投稿文件仍需核对。本次修订保留原稿技术主线、三张原图、所有均值和区间，只补实现依据和必要引用。

## 对五条建议的逐项处理

| 建议 | 判断及本次处理 |
|---|---|
| II.C 增加失败规则 | 合理且应补。实现不仅检查 CRC，还检查 profile 是否属于合法目录、长度是否合法、已知 padding 及算术源码解析。新稿写全这些**可检测的拒收条件**，随后输出固定灰图；没有把无法检测的错误或工程异常都宣称为 CRC 可识别。 |
| 说明 Latent JSCC 和 SwinJSCC | 合理。Latent JSCC 是本研究的卷积连续 latent 基线，不能暗示它是某篇已发表算法的原封复现。其损失为图像 MSE + 0.1 LPIPS + 0.01 归一化 latent 一致性，原建议只写 MSE+LPIPS 不完整。新稿列三类损失。Swin 使用最终 adapted 200k，并交代 256×256、固定 N1024 及付费 mask/power 信令。 |
| 补 LDPC、算术编码和经典渐进传输引用 | 合理。实际后端是 Sionna 5G LDPC，引用 TS 38.212 的 LDPC 与 rate matching；不据此宣称自定义 68 符号头部和整套帧均为 3GPP 协议。补 Witten–Neal–Cleary，以及 Chande–Farvardin 的目标码率下渐进图像传输背景，将本稿问题明确限定到 next-scale 合法端点粒度与硬预算装入。 |
| 写清 D_c 的适配 | 应补，但原建议的训练输入描述需要修正。当前 WCL 使用旧 Stage A、update 38000 的冻结 decoder，并非 10 月 22 日后续研究的新 decoder。其逻辑 batch16 固定为 8 个连续 F、4 个全量化 Fq、4 个 Fb_TX + α(F−Fb_TX)，α 均匀取 [0,1]；MSE+0.1LPIPS。新稿准确写成 continuous / fully quantized / prefix-completion-interpolated mixture，训练集为 20,000 张 ImageNet training images。 |
| 加作者信息 | 按用户回复保留匿名审稿版，正文和 PDF metadata 均使用 Anonymous Authors，未填姓名、单位或邮箱。 |

## 数字、版本和证据范围

- **Fig.2：**60 个均值和 120 个 CI 端点与 `eight_metric_summary.csv` 逐项精确相等；CLIP-I 读取原始余弦，19 dB 使用现有 T1-C 修正后 PARTIAL。没有重画或重统计。
- **Fig.3：**Pastries 95 与 Dog 21，N1024/10 dB，固定 `s10_i0095_r3` / `s10_i0021_r3`。十个已有 PNG 的 SHA、方法、LPIPS/DreamSim 标注逐一核验。示例属于已选 development 图，不称 common500 随机样本。
- **PDF 图稿身份：**原稿的 Fig.1、Fig.2、Fig.3 所有解码后的图形/图像内容流均与独立原图一致。这比“看起来是同一张”更强，但不等于重新执行科学验证；详细哈希在 `pdf_stream_identity.json`。
- **Table I：**利用率分别为 73.81/93.08、77.70/95.21、85.24/94.73%；三项 ΔPSNR 为 +0.616、+0.701、+0.319 dB；ΔDreamSim 均负，直接配对区间均不跨零。定义确为终止熵流 bit 数除以 `k_a−29`，与 Fig.1(b)、公式(5)相同。
- **同 MCS 的含义：**WHOLE/PARTIAL 在 4/10/19 dB 分别采用 QPSK 2/3、16-QAM 2/3、64-QAM 5/6，确实相同。不过目标端点和有序回退由两个端点集合分别校准。因此严谨结论是“端点粒度与校准后的实际码长装入共同带来的收益”，不应写成冻结所有策略后单独改变一个端点，更不能把五方法间的完整差距全归因于端点集合。新稿 III.C 保留这一边界。
- **NeST−ARPC 的 common500 PSNR：**4/10/19 dB 精确为 3.264229542154064、3.543238201321927、3.9650944168584097 dB，文中 3.26–3.97 正确。
- **10 dB 关键数值：**DreamSim 0.108/0.171/0.306、CLIP-I 0.873/0.789/0.713（NeST/Latent/ARPC）正确。LPIPS 的 NeST−Latent 直接区间为 [−0.003857436299199858, +0.00025122549831867175]，确实跨零；引用已存原 10 dB 配对统计，不通过边际 CI 推断，也未重算 bootstrap。
- **ARPC 敏感性：**相对原 τ=0.5 的最大绝对 PSNR 均值变化为 0.15038120621824597 dB；两替代设置在 4 dB 的 LPIPS/DreamSim 均变差。它支持“所测试的两种解码变体不能解释数 dB 差距”，不能扩写成所有解码均无影响或原采样普遍最优。19 dB 两替代设置也有改善，完整敏感性文件继续保留。
- **Table II：**三个条件及配对区间一致。19 dB 完整表示条件下 ΔDISTS=+0.005415155291557312，[+0.002400548428297043,+0.0085936841070652]；ARPC 的 DISTS 更好这一事实保留。4/10 dB adaptive 与 19 dB full 不合成同一固定配置曲线。
- **Table III：**五方法 TX/RX/Total 与冻结的 10 dB 表相等。Total 从未舍入值求和，不能要求显示到两位小数后的 TX+RX 必须恰好等于显示 Total。正文明确三个 warmup 是每 method/SNR 设置，48 次测量来自 16 源×3；这是 software processing latency，排除初始化、噪声生成及空口时间，不与旧 H800 数据拼接。

## Fig.1 与文字

Fig.1 当前的三面板关系与方法一致：系统流程、实际码长装入、预测边界尺度后覆盖收到的位置并推进。68+956=1024、13+16=29、77.70/95.21 也与正文及 Table I 一致。新稿继续嵌入用户原图，未替换成旧重排版。

Fig.1(b) 的利用率条是 500 源的均值，不是上方翠鸟单张图的实际码长；原图注已说明 measured means，新稿保留该说明。公式中的 `C_s` 在图注定义为 codebook lookup、resizing 与 residual transform；没有把算术解码已传 token 和生成未传 token 混为一谈。

最终 7.16 英寸宽的实际页面检查中，模块文字、箭头和图例可辨；辅助下标比模块标题小，应保留矢量 PDF，避免截图重采样。没有发现必须改变结构或重跑实验才能解决的问题。

## 原稿到 LaTeX 的其他有限修订

1. “four existing transmission approaches” 改为 “four other transmission approaches”，避免把自建 Latent JSCC 暗示成外部原封方法。
2. 增加 source-paired 的配对单位说明；方法各自保留历史噪声 namespace，不声称相同噪声 realization。
3. 图注说明 Fig.3 是 selected development examples，保留两个距离指标方向及单幅分数。
4. 计时一律写 software processing latency，保留共享机器负载记录，不声称独占无负载或对 RX 模块耗时做了未执行的分解。
5. 缩写期刊名、将长作者列表使用 et al.，保持正文 10 pt 和标准 IEEEtran 版式，未缩正文来压页数。

## 引用及提交格式依据

- [3GPP TS 38.212 v16.5.0，官方 ETSI 文本](https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/16.05.00_60/ts_138212v160500p.pdf)：第 5.3.2、5.4.2 节对应 LDPC/rate matching。
- [Witten–Neal–Cleary, 1987](https://doi.org/10.1145/214762.214771)：算术编码背景；[论文原文](https://www.cs.umb.edu/~rvetro/vetroBioComp/compression/WittenACM87ArithmCoding.pdf)。不是声明本仓库整数 CDF 实现逐行复制该论文。
- [Chande–Farvardin, 2000](https://doi.org/10.1109/49.848239)：渐进图像传输与目标传输率下联合源信道保护背景。仅据论文作者摘要和书目核对本次简短定位句，未声称已逐页复核其算法或做性能对比；IEEE 页面正文访问受验证限制。
- [WCL 官方投稿指南](https://www.comsoc.org/publications/journals/ieee-wcl/policies-guidelines)：双栏 10 pt，最多五页（含参考文献等），并另需双倍行距单栏稿。本轮交付的匿名双栏稿满足五页版式；单栏投稿文件、正式作者信息和投稿系统文件应在正式提交时补齐。本轮未提交期刊。

原始训练证据与代码路径列在 `implementation_evidence.json`；输入 CSV 路径、字节数和 SHA 在 `input_manifest.json`；数值核验在 `audit.json`。上述结论不把近期渐进 decoder 研究混入现有 WCL 主证据。
