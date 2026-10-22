# 渐进 token 率失真研究：本轮完成交付

本轮登记的缓存扩展、三组匹配 decoder 试训与最终校准均已完成。八个扩展子进程均由父进程实际 wait 得到退出码 0，无超时/终止；原调度进程已退出。全部已登记调用保持原预算，没有追加训练或信道实验。

## 数据覆盖

- 新扩展 18,968 个训练源 × 8 个端点 = 151,744 条；与已闭合种子库合并为 19,968 个训练源和 100 个校准源，各有 8 个端点，共 160,544 条。
- 另有 benchmark 的 32 个训练源 × 2 个端点 = 64 条，单独保留。没有补跑其余 6 个端点；不能写成完整 20,000 × 8 库。
- 包括 benchmark，独特 TX 源总数 20,100；复用原 RGB/token，不重做 tokenizer。每源一次概率遍历生成独立终止源码流。
- 扩展的源解码、原生 predict–overwrite–advance、LPIPS 和 MS-SSIM 各 151,744 次。模型/指标加载另列于 closure ledger，不计为图像恢复。
- 保存实际码流、接收 token、FP32 完成 latent、完整精度码长及质量。除已登记预览，不保存全量高分辨率重建图。无无线 PHY；这些是无损源码接收条件下的率失真标签。

## 训练源的描述性率失真表

以下为冻结当前 Dc 的 19,968 训练源均值；不是新训练模型的测试集性能。码长包含实际算术终止，未计无线保护和帧开销。校准100的独立均值保留在同一 CSV，不混合人口。

| (m,K) | 终止 bit | bit/pixel | PSNR (dB) | LPIPS | MS-SSIM |
|---|---:|---:|---:|---:|---:|
| (7,0) | 1556.001 | 0.023743 | 18.5667 | 0.213150 | 0.644326 |
| (7,25) | 1786.818 | 0.027265 | 18.9188 | 0.200552 | 0.663074 |
| (8,0) | 2500.955 | 0.038162 | 20.2951 | 0.156853 | 0.739196 |
| (8,84) | 3242.709 | 0.049480 | 21.0485 | 0.137546 | 0.775017 |
| (8,126) | 3617.188 | 0.055194 | 21.4978 | 0.126728 | 0.796433 |
| (9,0) | 4003.905 | 0.061095 | 21.9173 | 0.117720 | 0.811079 |
| (9,128) | 4980.219 | 0.075992 | 22.6706 | 0.102087 | 0.841841 |
| (10,0) | 5968.061 | 0.091065 | 23.5416 | 0.084237 | 0.872069 |

## 匹配试训结论与限制

三组各 5,000 更新、batch4、20,000 样本暴露，相同初始化与数据顺序；仅 decoder 更新。多前缀无条件相对等更新旧配方，在 (7,0)/(7,25) 的 PSNR 约提高 0.337/0.319 dB，在完整 (10,0) 下降约 0.152 dB。较长端点 LPIPS 多数变差。端点条件带来的 PSNR 增量约 0.002–0.028 dB，LPIPS 多数配对区间跨零。
相对冻结当前 Dc，多前缀无条件在 (7,0) 提高约 0.307 dB，在完整端点下降约 0.270 dB。没有全端点一致获益，不能直接替换当前 WCL 模型。所有端点、五个对比和不利结果均保留在 decoder_delivery_v1。
这只是一个训练随机种子的 calibration100 结果；一次已登记来源配对 bootstrap，逐点区间未校正多重比较，不代表训练种子不确定性、盲测或无线收益。缓存扩展没有重做该统计。

## 实际吞吐与存储

扩展八分片从最早启动到最后实际退出约 11803.842 秒，聚合 1.6069 源/秒（含前三卡等待训练造成的启动错开）。全体在原绝对截止时间前退出；截止时间没有重置。
逐分片数据见 throughput.csv。I/O 秒数是已埋点文件阶段，不能当作全机器磁盘等待，也不能据此断言无 PFS/CPU 瓶颈；实测源处理含 CPU 算术码及模型工作。设备为驱动标识 NVIDIA RPBZZZ6，FP32/TF32 off，原6 math/2 interop，独立单卡推理，没有 DDP 或跨卡显存池。

| 数据类型（种子＋扩展） | 文件数 | 实际文件字节 |
|---|---:|---:|
| latents | 20068 | 5298774788 |
| wire | 20068 | 594575083 |
| received | 20068 | 158617472 |

以上按已校验的实际描述符汇总，不含元数据、日志、checkpoint、原 RGB/token 依赖和 benchmark。FP32 latent 的真实体积不同于最初 FP16 容量估算。大文件继续放在共享 PFS，未下载或新做完整备份。

## 后续恢复清单

- recovery_manifest.json 给出缓存索引、实际注册、精确恢复注册的源码与三个最终 checkpoint 路径/字节/SHA。cache_expansion_index_v1.json 可按 source_id 找到扩展 completion、wire、received 和 latent；其 seed_index 引用同样定位原种子库。
- 全量源身份及预处理依赖按 expansion_registration 的 selected_sources 对齐；保留 source_index，不重新编号。加载前核验 SHA、source_id、端点顺序和 FP32 latent 身份。
- 三个 step_005000.pt 含 model/optimizer/scheduler/scaler=None、global_step、processed_samples、Python/NumPy/Torch/CUDA RNG、mixture RNG、data_order、config。checkpoint 在最终校准前保存，内嵌 calibration=None；必须同时保留 completion.json 绑定的最终 calibration.csv。
- 尚未执行单卡恢复后的更新，也没有已验证的 resume CLI；原训练入口为一次性入口，不能直接重跑冒充续训。后续应先实现受审的状态恢复入口，不能只加载模型权重后丢弃 optimizer/RNG/order。迁移设备或改变 batch/world size 不承诺逐位重现。
- 依赖当前冻结 VAE、VAR、Dc、原 RGB/token 缓存和原环境。本清单不是独立恢复包；用户允许推迟备份。

## 已保留失败与未开展范围

原种子 v1 的8次导入失败发生在模型加载主体之前，零成功科学调用；原现场和失败 reservation 全部保留。修复只处理导入命名冲突。早期绘图裁切及布局守卫误检也保留，未因此重算科学结果。
本轮未开展 encoder/VAR/codebook 更新、码率代理训练、内容选择器、分层 PHY、heldout/common500、新超参或随机种子扫描。临时八卡任务到此停止；进一步研究需新的有限方案，不自动消耗剩余窗口。

## 文件入口

- `../cache_expansion_delivery_v1/endpoint_m*_k*.csv`：8份逐源端点表，每份20,068行，源文件完整精度；`endpoint_means.csv` 分列训练与校准。
- `../decoder_delivery_v1/REPORT_zh.md`、`summary.csv`、`paired_differences.csv`：全部匹配试训结果。
- `../decoder_delivery_v1/figures_final_v1/calibration_rd.*` 与 `calibration_paired.*`：已实际检查的矢量曲线与配对差图，不重绘。
- `completion_evidence.json`、`recovery_manifest.json`、`throughput.csv`：本轮闭合、计算范围与恢复依据。
