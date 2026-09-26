# C 原始 N4084 10k 延长发布准备

2026-09-27 新增独立 CPU 工具 `tools/publish_c_extension.py`，不修改活动训练、模型、信道、缓存或调度器。适用范围固定为原 N4084、seed 2026092304 的 m6、m7、m8、pure，目标为 30k 及后续 10k 完整边界；不用于 N3060 或另外两个 seed 的不同引擎。

本次仅做准备审计，没有发布 30k 结果，没有新增 GPU 质量或在线计时。真实 m6 延长由既有队列执行，其他组按既有顺序继续。

## 发布门槛

默认入口要求原 `C_N4084_<group>_seed2026092304_until<step>` 成功阶段回执和不可变 completion snapshot、原训练命令、准确的单臂/双臂更新数、完整四轮新增校准及正式 scheduler 决策。读取不可变 snapshot，不把后续合法更新的活动 completion 当作历史损坏。

工具重新核验全部初始发布索引及后续已发布扩展索引、原 registration/模型/Decoder/调度源码绑定、十个真实 calibration tensor、CPU checkpoint 内部 registration/state/完整已填充 optimizer/恢复状态。核验上个边界 checkpoint 的已发布 SHA；不声称比较了两个不同更新边界的模型参数相等。

每轮完整 source/SNR/noise 键、utility 公式、失败字段及均值再次核算，全历史最小校准 utility 决定 selected，允许选中较早 checkpoint。正式延长规则保持“至少一臂末两个区间均改善不低于 0.2%，登记臂一起再训练 10k”，校准退步与停止原样保留。

只新增四轮原 CSV、源均值、SNR/分项曲线、决策、阶段/selected/边界谱系和资源账本；历史原 CSV 通过已发布索引 SHA 引用且逐项核对。m8 只有 H8-V；pure 从继承 10k 开始记录，source bit 与数字 CRC 不适用。校准 E 仍是登记约束，不能冒充逐帧新测量 E 或全在线时间。

## 已执行验证

七项 CPU 工程回归通过：配对决策、负改善、缺失轮次与错误规则、真实臂数量、活动 completion 变化与不可变 snapshot、保留较早最优 selected/pure 继承数、文件损坏与越界路径拒绝。

CUDA 完全屏蔽下，`--audit-only --until 30000` 已实际检查四组本地真实资产：
- m6：9 轮、270000 条历史校准记录；
- m7：9 轮、270000 条；
- m8：9 轮、135000 条；
- pure：5 轮、75000 条，继承 10k。

合计 750000 条是已有真实校准记录的 CPU 重新核验，不是新增独立 PHY 传输。四组各十个校准 tensor SHA、原 20k checkpoint 内部恢复状态与已发布 SHA 均通过。训练大 tensor 没有重新哈希；保留原实际 loader 验证范围。

证据仅写入 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/monitoring/C_extension_preparation_*。准备审计不创建 C_extensions 发布目录。整个默认完成阶段发布分支尚待真实 30k 阶段到达后执行；CPU 工程测试不代替这项真实完成门槛。

## 使用与后续

准备核验：
```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -m tools.publish_c_extension --group m6 --until 30000 --audit-only
```

原阶段实际成功、正式 decision 存在后，去掉 `--audit-only` 才可发布。目标目录为 results/token_channel_efficiency_20260923/C_extensions/<group>_seed2026092304_until<step>，已存在时拒绝覆盖。后续边界要求上一扩展已经封存发布。每个真实阶段仍须视觉检查、单独结果报告、显式 Git 提交/推送和独立远端完整数据复验。

原 C 延长、N3060、多 seed、selected development/计时/诊断、四个历史 GPU worker、最终全方法配对与合并报告仍待执行；新 holdout 和内容选择器继续暂缓。
