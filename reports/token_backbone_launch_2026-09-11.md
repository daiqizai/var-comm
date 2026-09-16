# 通信骨干2×2：实现、GPU检查与启动

## 固定边界

按用户新方案，只升级通信Encoder/Decoder。旧B selected step6000及其模型、结果、完整receipt均只读；VQ、VAR、官方RGB Decoder、m8、68+2992=3060 uses、平均复符号能量2和真实header规则不变。
现有小网络已经是Transformer，不以换名称制造新架构。增强使用192维、6头、4层Pre-LN发送+4层跨尺度共享接收；全局注意力、尺度编号及尺度内二维坐标；不硬拼二维图，不加巨型flatten全连接、Rate ModNet或功率模块。

| 家族 | Encoder参数 | Reader参数 | 总训练参数 |
|---|---:|---:|---:|
| SNR匹配小骨干（128维/2层） | 503013 | 1056041 | 1559054 |
| 增强骨干（192维/4层） | 1871141 | 3268521 | 5139662 |

每个家族包含parallel/next-scale两臂，参数与初始化SHA完全匹配。四臂都使用同款SNR特征调制；小骨干因此比原151万版本增加约5万参数，不冒充旧B逐参数不变。
名义SNR在发送前为TX/RX所知是本轮明示的理想条件，不是本次噪声或RX反馈，也不表示实际CSI获取无成本。

## 完整训练历史匹配

四臂都从头初始化模型/Adam，没有向增强臂硬加载不兼容optimizer。共同配方warmup 10000步，之后B的2000 prefix+8000 joint，总20000步/80000图像曝光/臂；数据、翻转、SNR和噪声逐batch配对。
每臂独立端到端训练发送和接收模块，配对的是源图及标准噪声，不保证训练后的发送波形相同。因此本轮是接收策略下联合通信映射的架构比较，不是固定同一接收y的纯接收器消融。
从第一步起都用自身历史，不复刻旧预训练的teacher阶段；原B只作为工程参考，严格结构结论来自新的四臂同历史对照。
B阶段权重/学习率不变。1000图完整选模、100图监控、LPIPS主指标和0.2 dB PSNR约束保留；共同PSNR参考预先指定small_next_scale的warmup端点。DINO不训练、不选模，不访问新holdout。
完整协议位于代码仓库`reports/token_backbone_protocol_2026-09-11.md`，不按development结果调整方案。

## 检查与测速

CPU：参数/同家族初始化、全部参数梯度路径、5984实坐标与功率、真实二维坐标、可微累计state、阶段边界、采样resume及16800行合成统计检查通过。合成记录不是新实验性能。
GPU：四臂hard图像均与原接收算法像素差0；image/state梯度到E/D非零。prefix阶段parallel不调用VAR，next-scale调用8尺度VAR，两者都不构建后缀/RGB；所有冻结权重未变。profile没有任何optimizer更新。
实测训练主体8.17 GPU小时，计划校准1.85小时，合计10.02小时；另计optimizer、I/O及最终development评测。不能按3倍参数推断3倍工期。

## 启动状态与故障保留

首次尝试于14:44（UTC+8）在创建空历史分支的selected.json时发现目录未预建，新增更新0次。已修复入口目录创建并补CPU回归检查；原失败目录`outputs/VAR-TOKEN-BACKBONE-20260911-TRAINING/`保留，不改其旧snapshot或失败记录。
为保留现场，重启仅改训练输出路径到`outputs/VAR-TOKEN-BACKBONE-20260911-TRAINING-R2/`；科学配置与已测速版本一致，profile原配置未改写。
当前本机配置`configs/token_backbone_20260911_r2.local.yaml`；监督器PID 1408947，训练PID 1408979。2026-09-11 14:46启动，14:46:15四臂各完成并保存第1个真实optimizer更新；不是仍在排队/profile。
14:53已核验四臂各250个共同更新：Adam step250、batch指纹完全配对、loss/gradient有限、teacher=0、最大功率误差5.96e-7，旧B及新运行源码snapshot未变。核验收据位于PIPELINE目录的`first_update_audit.json`；后续进度以实时status和共同断点为准。
先训练，再自动运行16800行development评测及2×2配对/交互分析；任何阶段失败立即停止，不自动调整阈值、模型或继续扩大实验。旧B文件校验不因新增代码而失效。

## 查看与恢复

以下相对`/workspace/projects/var-next-scale-comm`：
- 实时阶段：`outputs/VAR-TOKEN-BACKBONE-20260911-PIPELINE/status.json`。
- 阶段日志：同目录`train.log`、`evaluate.log`、`analyze.log`，首次失败和新启动按时间保留。
- 四臂共同断点：`outputs/VAR-TOKEN-BACKBONE-20260911-TRAINING-R2/resume.pt`；首步和每250步持久化，恢复时不能单独推进某臂。
- 恢复命令使用相同本机config/assets，`scripts/run_token_backbone.py pipeline --resume --execute --background`；须先确认旧监督器确实结束，不改冻结源码和配置。
- 完成报告预期在`outputs/VAR-TOKEN-BACKBONE-20260911-ANALYSIS/report.md`，完成receipt生成前不宣称新骨干收益。

当前粗估全部完成在2026-09-12凌晨，实际取决于完整训练吞吐、校准及I/O；不是时刻保证。
