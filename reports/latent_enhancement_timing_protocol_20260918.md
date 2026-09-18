# 连续latent增强：完整在线TX/RX计时协议

本轮在development的固定10张等距源图（0、11、22、33、44、55、66、77、88、99）上计时；每图五个SNR、一个预登记噪声seed2001，不访问holdout。模型加载、信道传播、指标计算、磁盘写入和排队不计入端点时间。

TX从源RGB输入开始，包含官方VQ encoder/quantization前向、真实m8基础VAR完成、raw或arithmetic源编码、以及512/1024增强Encoder（适用时）。RX从已生成的接收波形开始，包含原PHY译码、实际前缀VAR完成、增强Receiver和D0/Dc最终RGB输出。每个端点在GPU同步后测量，batch=1、FP32、TF32关闭。

计时方法包括原m8/D0、m8/Dc、receiver-only、512/1024增强，以及N3060/N3572/N4084上的代表性raw/arithmetic m8和m9。它们用于完整在线计算取舍，不将单次PHY或单步网络耗时外推为系统端到端无线时延。
