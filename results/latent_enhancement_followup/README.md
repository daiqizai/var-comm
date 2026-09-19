# latent增强后续对照

本目录发布数字校准动作、N4084折中分配、修正在线计时和1024 Decoder适配的摘要结果。没有checkpoint、数据像素、latent缓存或holdout。

## 结果边界

- 数字校准：N3572/N4084、raw/arithmetic、m7/m8/m9、D0/Dc，动作只由原calibration冻结；当前header合法mode域不支持m10；
- 折中分配：N4084固定m8，3504数字+512连续，与4016+0和2992+1024对照；
- Decoder适配：10000步通信续训控制与通信＋Dc适配两臂，随后在100张development上重渲染；
- corrected timing：650调用，重构前后数字RGB最大差0，噪声移出RX计时且不重复D0/VAR调用。

完整解释见`reports/latent_enhancement_followup_convergence_20260919.md`。所有结果仍是development/calibration，不是新的holdout。
