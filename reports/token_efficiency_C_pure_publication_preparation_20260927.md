# P4084 C 首轮接续的独立 CPU 发布入口（2026-09-27）

本次新增工具 `tools/publish_c_pure_milestone.py`，仅为已登记的 pure_seed2026092304 从历史 10k 接续到总 20k 的校准里程碑提供核验与封存。它不启动训练或 GPU、不更改活动源码、协议、权重、数据或硬件。当前只是发布准备，不是 20k 完成、development 质量验收或合并实验交付。

## 已实际核验

在 `CUDA_VISIBLE_DEVICES=''` 的 CPU 环境中，历史 parent checkpoint SHA 为 `b6bd1f0bb01bbd734eeb6a63afac1b8f7c57998085c616011f37554035434e1a`。新注册 SHA 为 `2536a256d42bd2e9beb1214dd8a00bb1723b3859f01cabd9f7cff677fd0ed431`。真实父 checkpoint 与接续的 step10000 checkpoint 逐张量比较，模型参数和 buffer、已填充优化器、数据顺序及训练随机生成器状态均完全一致。该检查不声称新校准与历史 development 使用相同 population 或噪声，也不证明整个后续训练轨迹与旧引擎逐位一致。

首次 `--audit-only` 完整重验已封存的 10000、12500 两轮，每轮 1000 原 calibration 源 × 五 SNR × 三噪声 seed，共 30000 条记录：CSV SHA、完整唯一键、源顺序、loss 公式及 utility 均值通过。实际 selected checkpoint SHA、原真实 GPU 训练资格及父/当前绑定通过。10 份原 F calibration tensor 的 SHA 和源顺序已重验；200 份 train shard 元数据保留，训练大 tensor 未由本发布器重新哈希，原训练 loader 的完整检查范围不得夸大。详细预检 receipt 只保存在 outputs 同实验 monitoring 下。

## 发布约束

只有 `delivery_chain_v1/stages/C_initial_pure.json` 返回成功且不可变 completion snapshot 的 SHA、总更新数 20000、完整校准 20000、唯一 P4084 臂与 parent_updates=10000 均通过，默认入口才允许创建不可覆盖的结果目录。完整首轮应该包含 10000、12500、15000、17500、20000 五轮共 75000 条记录；本阶段新增更新数为 10000，不能写成从零完成 20000 新更新。

工具保存完整原 CSV、源图/SNR 均值、校准曲线、selected/父谱系、原 F 缓存清单、资源账本及带 SHA 的文件索引。它使用成功阶段 snapshot，避免后续延长更新活动 completion 或 selected 文件时篡改历史选择。

pure 的 digital body CRC 始终为 `not_applicable`，header_ok=1 是无数字头的序列化约定；不能把这些字段转换成数字传输成功率。N=4084 全部用于连续波形，数字 header/FEC/code slots 为零；序列化源 bit 数不适用，绝不以零 bit 代替 N。E=8168 为登记约束，校准 CSV 不含逐帧实际 E 或在线 TX/RX 计时，后续真实评测仍待执行。噪声使用原 `VAR-CONTINUOUS-4084|image_id` 命名空间。

`--audit-only` 仅验证现有完整校准前缀并将准备证据存入 outputs，不生成阶段 publication；准备 PASS 不覆盖 20k 尚未完成的事实。正式校准延长决策仍由原 scheduler 作出，发布器不选择或注册新实验。历史 continuous_grid_v1 的旧 P4084 10k 结果保持其原 lineage。

## 工程检查与待办

五项新增 CPU 回归覆盖无 CRC/未掩码 loss、完整 source/noise 键、总 20k 与继承 10k 约束，以及权重/优化器/数据顺序/RNG 的真实差异拒绝。首次父模型检查因 dict 与 OrderedDict 类型不同误拒绝；逐张量实查无差异后修正为按映射内容核验，并覆盖回归。这是独立发布器工程检查问题，不是训练失败。

仍需原 pure 阶段实际完成后运行正式发布、审阅图表、提交数据与报告并独立远端重验。之后仍有 C 延长、N3060、多 seed、selected 正式评测与计时、诊断、四个历史 GPU worker 和最终全方法配对交付。新 holdout 和内容选择器继续暂缓。
