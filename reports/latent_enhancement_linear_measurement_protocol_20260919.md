# 新机制一：真实低维测量＋接收端线性纠偏

第一轮使用N4084/E8168，保留m8/N3060数字基础，增强层占1024复符号（2048实坐标）。A为固定seed生成的行正交矩阵`2048×8192`；这是可复现的线性原型，不把随机投影冒称为PCA最优方向。

两臂：

- source measurement：发送`A f`，RX用实际`Fb_RX`和测量观测做`Fb_RX + AᵀW(c_hat−A Fb_RX)`；
- projection residual：发送`A(F−Fb_TX)`，RX用`Fb_RX + AᵀW r_hat`。

测量使用训练集固定平均范数，禁止逐图免费增益；增强波形仍归一到精确N/E，范数丢失作为机制限制保留。W按SNR只在calibration拟合，development冻结。D0、VAR、量化器、Dc、m8数字协议不变；全部失败计入，不访问holdout。

此原型的目标是判断测量方向是否有独立价值，不与神经增强同称同等模型容量。结果需同时报告N/E、失败状态、PSNR/LPIPS/DINO和完整TX/RX成本。
