# 固定分配：数字基础＋实际源残差通信

唯一问题：N3060/E6120、一次发送、无反馈，分出连续资源实际传送数字基础没有表达的源信息，能否提高PSNR而满足LPIPS非劣？不是两个完整系统各用3060再平均，也不是从不变token中继续搜索。

协议：`../../reports/hybrid_source_correction_protocol_20260915.md`。
配置：`../../configs/hybrid_source_correction.json`。
产物：`../../outputs/HYBRID-SOURCE-CORRECTION-20260915/`。

## 固定内容

- 原官方冻结VAR/VQ，raw m7，68 header＋1882数字数据＋1110连续残差。
- 一个164万参数的连续E/D；add、snr_gain、reliability_gain三臂。两个gain臂同1642752参数，add为1642637。
- 同20k训练、1k校准、真实数字错误、同数据与噪声；每臂10000更新、有效batch16。
- 主1/4/7 dB；相对两条完整数字自适应分别约束LPIPS差≤0.005，再选择PSNR最高的完整校准checkpoint。无可行点时明确不合格。
- 原固定m7/m8/m9是“把连续额度给更多保护或更多真实token”的直接数字控制，完整感知Deep保留。缩短数字基础仅是消融。

## 有界队列

先构建确定性TX基础缓存；随后CPU回归、真实GPU梯度/测速、2-source质量及同端点计时工程回归，再正式训练、完整开发比较、只补新系统计时与源图级统计。任何失败停队列，保留日志，不自动换架构、分配或延长训练。

实时状态以`pipeline_001/status.json`及其中child PID、各阶段真实日志为准，不根据文档中的旧PID启动重复任务。未产出完成回执不能称训练/评测完成；工程smoke不进入科学排名。

运行环境沿用`experiments/backbone-eval-20260912/.venv/bin/python`；不安装全局依赖、不租卡、不启动旧WeTok队列。脚本都在项目`scripts/*hybrid_correction*.py`。

训练缓存不出现在在线计时里，TX必须实际运行自己的VAR补全。真实类别在TX已知、名义SNR理想共享的假设保留。1000旧holdout已用于新假设，不再是新独立测试；本队列不访问新holdout。
