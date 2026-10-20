# Results paragraph tasks

## III.A. Experimental Setup

**段落 1：主比较与版本。** Fig.2 使用 N1024、4/10/19 dB、原 common500。五方法为 entropy-coded NeST-Com、Latent JSCC 最终100k、SwinJSCC adapted 最终200k、现有 adaptive BPG + LDPC、adapted ARPC + LDPC。NeST 19 dB 用当前 T1-C PARTIAL；不混入旧 raw 或旧检查点。

**段落 2：评价与统计。** PSNR、LPIPS-Alex、DreamSim、CLIP-I；CLIP-I 是 OpenAI CLIP ViT-L/14 的图像—图像余弦，224 输入、冻结浮点预处理、FP32，不做 softmax 或百分比变换。三次原噪声先按每源平均，500 源等权；点位 CI 沿用既有 source-bootstrap。差异读直接来源配对 CI；历史方法噪声命名空间不同。

**段落 3：人口边界。** common500 主图、matched100 的 N2048／敏感性、fixed development 示例与计时分别报告。灰图、回退和失败输出按原协议保留。DISTS、DINO-L、agreement、MS-SSIM 完整结果仍归档；不借删指标制造一致优势。

## III.B. Comparison With Existing Methods

**段落 1：主性能。** 引用 Fig.2，分别描述像素、感知距离与语义表征一致性。保留 NeST 并非所有工作点 PSNR 最高、4 dB LPIPS 不如 Latent JSCC、LPIPS 与 DreamSim 排序存在差异。不能用曲线 CI 重叠判显著性，也不能把 CLIP 差异归因于未经消融的 caption 条件。

**段落 2：可视化。** 引用 Fig.3 的 Pastries95 与 Shopping cart75，描述实际可见的糕点轮廓、表面纹理、购物车网格与把手等；使用统一 ROI 比较局部细节，同时保留完整视野。明确两个案例是既有选定 development 示例，不外推总体优劣。

**段落 3：ARPC 解码敏感性（本轮已完成）。** matched100、N1024、同一900个接收事件。原采样 τ=.5 复用；显式 argmax 与预定低温 τ=.25 只改变未传尺度生成。19 dB 两种设置四项均改善，贪心 ΔPSNR=+0.150 [0.087,0.207] dB、ΔDreamSim=−0.0143 [−0.0206,−0.0084]；4 dB 两者 LPIPS 和 DreamSim 均变差。10 dB 低温的 PSNR/LPIPS 改善，但贪心四项 CI 跨零。全设置都报告，不用子集替换 common500 主曲线。

## III.C. Effect of Within-Scale Endpoints

**段落 1：装载效率。** 引用 Table I，三个 SNR 的 PARTIAL 利用率均提高；10 dB 从77.70%提高到95.21%。利用率只报告真实均值，不把样本分位数当均值置信区间。

**段落 2：同先验机制检验。** Δ=PARTIAL−WHOLE。三点 ΔPSNR 约+0.616、+0.701、+0.319 dB，ΔDreamSim 为−0.0207、−0.0240、−0.0030；直接配对区间分别全正／全负。说明合法尺度内端点改善预算匹配，不额外宣称唯一因果机制或新训练收益。

## III.D. Performance With a Larger Symbol Budget

**段落 1：有限预算扩展。** 引用 Table II。matched100、N2048；4/10 dB 为冻结 adaptive，19 dB 为 full representations。每源三噪声，每方法每条件300/300正确送达。表内报告 ΔPSNR 和 ΔDreamSim，均为 NeST−ARPC，不把三行画成同一配置曲线。

**段落 2：必须同时保留的边界。** 19 dB 的 NeST PSNR 更高、DreamSim 更低，但 DISTS 差为+0.0054，95% CI [+0.0024,+0.0086]，ARPC 的 DISTS 更好。保留这一指标取舍，不将 N1024 CLIP 填入 N2048 表。N2048 CLIP 本轮未启动。

## III.E. Computational Cost

**段落 1：测量口径。** 引用 Table III，仅取现有10 dB、16 development、batch1、每源3次预热+3次测量。相同 RTX PRO 6000 Blackwell GPU1 和6个CPU核心；保持原方法环境／精度和共享负载记录，不能声称独占或所有框架完全一致。

**段落 2：质量与处理时间。** TX/RX/Total 为 software processing latency，单位 ms/image；Total=TX+RX。NeST 均值约220.89/398.82/619.71，连续基线约42–45 ms总计；ARPC约1520.30、BPG约2293.01 ms总计。说明生成式恢复与源码处理的实际成本，不称空口时延或端到端网络时延，不与历史 H800 六方法计时合并。
