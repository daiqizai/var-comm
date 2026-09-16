# 复现等级与外部资产

## 1. 当前公开仓库可以独立完成

- 检查发布清单与文件SHA；确认没有大型checkpoint、数据像素或凭据。
- 从`results/hybrid_weight_closure/development.csv`及`references.csv`复算均值。
- 使用校准冻结的概率复算单系统帧间选择的期望指标。
- 先在每源图内平均三个噪声和相应SNR，再做10000次source-image paired bootstrap，复算1856项原配对区间。
- 从`full_calibration_all_metrics.csv`核对六点的校准选模，区分校准、development和旧holdout。

需要Python 3.10+和`requirements-results.txt`。结果复算不需要PyTorch/CUDA，也不会下载模型、读取外部图像或启动训练。

## 2. 当前没有作为一键包提供

重新训练、重新生成图像、重新计算LPIPS/DINO像素指标及在线计时，仍需要：

| 外部资产 | 本仓库提供 | 没有提供 |
|---|---|---|
| 官方VAR / VQ视觉骨干 | 接口、配置、所用checkpoint哈希 | 上游vendor、权重 |
| LPIPS / DINOv2 | 评测代码、配置及报告 | 预训练权重、第三方实现副本 |
| ImageNet训练/校准/development | 源标识、协议和指标，不含图像 | 数据像素、缓存shard、未用测试候选清单 |
| 历史JSCC与WeTok参考 | `experiments/`下自有通信/审计代码 | 外部视觉vendor、原历史项目的完整依赖树 |
| 暖启动与optimizer | 选择步骤、SHA、完整历史审计回执 | 大型模型/Adam文件 |

所有大型原始产物仍在研究者原工作区原位保留，没有为了Git整理而移动或删除。

## 3. 路径与哈希不能混淆

发布副本将机器home/workspace前缀改为`/workspace/...`。`release_manifest.json`分别记录原始SHA和发布SHA；这不是修改历史实验结果，也不能冒称发布副本与原文件字节完全相同。

GPU脚本、协议中的`outputs/`引用和历史哈希属于当时完整工作区。公开精选结果位于`results/`，不是整个`outputs/`镜像。缺失历史输入时脚本应报错，不能将检查关闭或伪造SHA来跑通。

新服务器迁移需要单独配置资产路径，核对权重/数据/初态/Adam后登记新的执行清单。当前未做这一新环境的端到端训练资格验证，README不提供会隐式启动旧队列的“全部运行”命令。

## 4. 已观察到的研究环境

原会话Python 3.10，CUDA GPU；实际安装版本记录如下。这是环境记录，不是声称任意硬件上自动可用的锁文件：

| 包 | 原环境版本 |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| matplotlib | 3.10.9 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| timm | 1.0.27 |
| lpips | 0.1.4 |
| einops | 0.8.1 |

完整图像处理时间不包括空口、排队、模型加载、类别获取或CSI估计。理想共享名义SNR与发送端已知真实类别仍是原系统假设，不因迁移而消失。
## External baseline release addendum

`python tools/reproduce_external_results.py` uses CSV and NumPy only to recompute 119 summaries and 1,624 paired metric intervals from 25,500 published rows, and verifies frozen mode selection from 18,000 calibration rows. It does not read pixels, models or GPUs.

The selected comparison/reconstruction PNGs are user-requested research illustrations; panels contain source references, but standalone source files, datasets and raw arrays are not distributed. HiFi has only a dated running snapshot here. Original and published hashes, plus CSV partition selectors, are recorded separately.

The author models use an isolated torch1.12.1+cu116 environment. The torch2.11 table below describes the original digital/metric path, not the author Swin SA/RA runtime. Full GPU replay still needs licensed data, public weights, pinned upstream code and historical dependencies.
