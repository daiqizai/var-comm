# 新机制二：共享预测器＋预测创新残差

固定N4084/E8168和原m8数字基础。冻结已选receiver-only refiner作为连续latent预测器P；TX用名义SNR和真实m8基础`Fb_TX`生成`P_TX`，RX用自己的实际`Fb_RX`和可观测状态生成`P_RX`。不把P接收输出称为已传输信息。

配对两臂：原`F−Fb_TX`残差控制，以及`F−P_TX`预测创新候选。两臂共享通信骨干宽度、初始化、数据顺序、SNR、实际数字候选和连续噪声；RX均从对应`Fb_RX`或`P_RX`开始。Decoder、VAR和预测器P冻结，loss为原MSE＋0.1 LPIPS＋0.01 normalized latent loss。

本轮登记5000次更新、每2500次完整校准、每500次保存；不访问holdout，不叠加功率分配或新Decoder。最终需要同时与原1024、无新增观测精化、数字同N对照比较；若只降低latent残差却不改善图像，不判定机制成立。
