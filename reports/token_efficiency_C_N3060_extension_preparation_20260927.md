# N3060 C 延长阶段独立发布准备（2026-09-27）

本次只准备未绑定的 CPU 发布工具，未修改训练、调度、PHY、模型、loss、硬件或活动 worker。N3060/m6 首轮 20k 发布提交为 `78b0a3bb5635808e732e34d656873a6927d642ee`；其独立远端验收已经完成，221 项 CPU 检查、269 个文件、270000 完整校准记录和 90000 条逐源均值通过，原阶段回执缺口及空退出码仍保留。

## 入口与限制

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -m tools.publish_c_n3060_extension --until 30000 --prepare-only
```

新工具只接受原 scoped N3060/m6、seed2026092304、初始 20k 后的 10k 边界，不适用于 N4084 或另外两个 seed。实际完整边界及原校准 decision 出现后，去掉 `--prepare-only` 才能正式发布；`--audit-only` 核验已完成边界但不创建结果目录。正式成功 stage 的不可变 snapshot 优先，不能用活动 completion 的变化误判历史结果。

若原热暂停导致 stage 缺口，只能显式使用 `--allow-thermal-receipt-gap`。该分支要求准确的边界 completion、原命令、已退出进程身份、内外热停止证据和严格时间顺序、完整无失败校准日志、原 runner 重启及正式 decision。它封存可验证终端资产，不推定原退出码或伪造 stage；是否延长仍取原 decision。没有修改原 N4084 仅接受停止边界的恢复发布器。

正式发布会核验全部历史完整校准键、公式、均值、两臂失败一致性和最优 selected，实际 terminal 与上一边界 CPU 完整模型/填充 optimizer/恢复状态，以及注册、缓存与原资格身份。仅新增四轮 120000 条校准记录及逐源均值；既往 CSV 与已发布副本逐字节 SHA 对照后引用其不可变 index。40k 及后续必须先完成前一延长的发布。输出 `C_extensions/N3060_m6_seed2026092304_until<step>` 不可覆盖。

## 已执行的 CPU 准备

`N3060_extension_preparation_1790518698927723293.json` 实际重审 0 至 22500 共十轮 300000 校准记录的完整 source/SNR/noise 键、公式、均值和两臂失败一致性；核验 52 项训练绑定、两 role 缓存 completion/registration、10 个 calibration tensor SHA、已发布 20k index 和原 20k checkpoint 的填充 optimizer/恢复状态。训练大 tensor 未重新全部哈希，仍限定为原实际训练 loader 的验证范围。

七项新增 CPU 回归覆盖错误 scope/非 10k 边界、未完成双臂、immutable snapshot 与活动 completion 分离、失败 stage 不回退为热缺口、缺口默认拒绝、旧 CSV 被修改拒绝、仅发布新增四轮和缺少前一发布拒绝。初次测试错误地预期 AssertionError，而共用核验函数实际抛 ValueError；已修正测试预期并通过全部 228 项 CPU 检查。这不是 GPU 训练失败。

新工具 SHA256：`6ec34cf8b6b7e6879ac9fd3ea8ac946c1b13871d8718436275800fc5aeb10920`。
测试文件 SHA256：`6c25355171fa18cf22653d6fc217670d9a73a186f8be6e600a99634cc36721af`。

## 仍待实际执行

正式 30k 发布分支尚未执行，没有生成 30k 结果目录。CPU 准备不是 30k 完成、GPU 质量验收或整份实验交付。校准缓存复用评分不是新增独立数字 PHY 传输，E6120 是登记约束而非新逐帧能量测量。N3060 校准延长、另外两个 seed、selected development/诊断/在线计时、四个历史 GPU worker、最终全方法配对统计和合并报告继续由原队列推进。
