# R3已经实际训练：完整起点资格、首1000审计与续训

本机UTC+8日志：2026-09-15。R3不再只是登记；模型、训练/恢复、GPU profile和校准审计已实现并执行。首1000训练于00:31完成更新，00:35:11完整校准/审计PASS；00:38:24从实际1000模型/Adam继续到10000，00:46读取实际2000保存点确认继续生效。

## 唯一机制干预

保持相同三次16×16读取、两次接收端E回算、2,933,775训练参数、残差能量/SNR/阶段门控与Joint E/R训练；只把送入融合的向量从`y−E(自身连续状态)`改为`E(自身连续状态)`。新控制仍使用标量残差信息，不是完全移除残差。

向量方向和幅度统计都可能随构造改变，因此本轮只解释整体向量选择，不能进一步单独归因纯方向或精确似然。Joint学习后各发送器可能不同y；只配对源图、标准噪声、实际N/E和训练机会。详见`experiments/wetok-reencoding-vector-control-r3/docs/interpretation_limits.md`。

## 起点与执行资格

- 新模型匹配原残差臂的共同7000 single父点与零融合初始化，不从R2最优残差权重切换短训。
- 实际所有初始参数逐值匹配，CPU五SNR输出相同；真实GPU初始波形/RGB差为0。初始函数一致才复用原完整零步校准，不重跑或额外增加选择机会。
- 已核对原残差全部10000批次的源顺序/增强/SNR/噪声；fresh Adam0、lr1e−4、betas(0.9,0.99)、weight decay1e−4、clip1与有效batch4/micro1不变。
- 零更新GPU检查中，图像loss经过冻结Decoder回传通信E/R与波形边界；注册loss0.011260935与原残差第一步一致，参数未被profile更新。
- 四项CPU测试PASS，含非零融合时的接收信息边界，以及真实driver中断恢复的模型/Adam逐值等同连续执行、保留较早选择和零步校准。
- 物理账本仍是Fq32×16×16、8192rawbits、3060复信道使用、0header、总能量6120、实噪声方差1/γ。视觉模型冻结，无类/caption/真历史/反馈。

## 首1000完整校准，仅是过程结果

两者均限于1000更新机会、相同1000校准源×五SNR，均选择1000；没有拿新1000与旧残差10000比较。

| 向量 | PSNR ↑ | LPIPS ↓ |
|---|---:|---:|
| R3 prediction向量 | 21.20901 | 0.198349 |
| 原residual向量，同1000机会 | 21.19071 | 0.202011 |

这是完整校准均值，**不是最终development结论**。早期相对走势可能在后续训练改变，不据此宣布通用特征已胜出、物理残差无效或更改10000预算/选模规则。

## 当前运行与证据

当前trainer **1099666** / reviewer **1099667** / passive observer **1099668**，目标10000；原首1000进程1082178/1082179/1082180已正常退出，不重启。

- `outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-PREPARATION/initialization.json`：真实初始化/完整历史资格。
- `outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-PROFILE/profile.json`：真实GPU初始/梯度/零更新证据。
- `outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-ANALYSIS/calibration_0001000/completion.json`：模型、Adam、数据、功率与同机会选择审计PASS。
- `outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-PREPARATION/actual_continuation.json`：实际2000模型/Adam已更新，global9000；前1000完整历史、已有校准及绑定源码不变。
- `outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-TRAINING/status.json`与`outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-step0010000.pipeline.json`：实时阶段与自动终点审计。

最终23方法/48300质量主行和16模型/2560独立计时的协议、配置已登记，**评测驱动尚待补齐和验证，尚未排队最终质量/计时**。本次只已运行训练与终点校准审计队列。R2完整结果只读，不访问新正式holdout，不把全网格迭代叫next-scale，研究总目标仍未完成。
