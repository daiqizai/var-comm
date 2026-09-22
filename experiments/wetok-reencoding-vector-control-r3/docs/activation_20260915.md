# R3实际启动依据（本机UTC+8，2026-09-15）

本轮不是从R2最优残差权重切换后的短续训。新控制从原共同7000父点和相同零融合参数开始，fresh Adam0；参数数目2,933,775、两次RX内部E与完整10000配对数据历史均与原残差臂核验一致。

- 实际CPU资格：原始checkpoint/全部参数逐值相同，五SNR初始函数相同，两次E调用相同；已重放全部10000批次的源顺序/增强/SNR/标准噪声指纹。没有GPU更新或development推理。
- 实际零更新GPU profile：新控制与原残差的初始波形和RGB差均为0；原第一批次注册loss也完全一致。只用MSE＋0.01LPIPS时，E/R/波形边界梯度范数分别0.023990/0.026965/0.000928，冻结视觉/指标参数和通信参数没有被profile更新。
- 注册loss零更新批次约0.405秒，峰值PyTorch分配5,033,737,728 bytes（约4.69 GiB）；不是独立RX时延或整进程硬显存上限。
- 4项CPU测试通过：同参数/初始函数/向量内容与残差能量门控、接收信息边界、图像梯度、真实训练driver中断恢复后模型和Adam逐值等同连续执行；起点校准不重复，较早最佳点不丢失。
- 最初发现本地`profile.py`遮蔽Python标准库profile，已在任何资格/训练绑定前将新入口改为`profile_model.py`；未修改任何旧项目或冻结依赖。

先在空闲的授权GPU0运行1000更新及完整1k校准/审计，随后按协议从实际模型/Adam继续10000。GPU profile和CPU资格回执分别在`outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-PROFILE/`与`outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-PREPARATION/`。启动后的实际PID/步数以状态和/proc为准，不把本文件当成完成证明。

只有融合向量q替代y−q；标量残差能量门控保持原定义。不称完全无残差、不称精确MAP或next-scale。相同Joint训练机会并不意味着最终各算法的y相同。原100图仍是development，新holdout不访问，研究总目标未完成。
