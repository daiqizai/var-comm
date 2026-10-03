# 统一指标 SNR 字符串兼容恢复

本目录登记独立的严格整数 SNR 字符串子类兼容层及其工程证明。原实验、行标识、数值、模型和策略保持不变；兼容资格与真实首源回放证明分别保存，最终指标仍以统一指标完整结果为准。

## 顺序发布

`numeric_delivery.py` 等待 `outputs/M2-TELEMETRY-20261002/publication.json` 的正常推送证明、`publication_status.json` 的完成状态和发布进程退出。随后只读加载已经发布且 SHA 验证通过的 JSON 恢复发布框架，继续核验原 M2、统一指标与最终缓存监督者均已完成、推送并退出。等待期间不进行 Git 操作。

```text
python numeric_delivery.py --root /home/liulu/projects/VAR_COMM
python test_numeric_delivery.py
```

可选参数：`--runtime`、`--numeric-manifest`、`--cpu-qualification`、`--real-first-source-qualification`、`--poll-seconds`（1–60 秒）、`--once`。资格参数必须匹配固定 manifest 白名单；`--once` 等待时退出码为 75。

## 固定输入

默认 manifest 为 `outputs/METRIC-NUMERIC-RECOVERY-20261003/numeric_publication_manifest.json`，包含：

- `status="NUMERIC_RECOVERY_READY_FOR_PUBLICATION"`，`scope="STRICT_INTEGER_SNR_STRING_SUBCLASS_ONLY"`。
- `source_bindings`：整个 runtime 中全部 `.py`/`.md` 的绝对路径和 SHA256。
- `proof_bindings` 和 `artifacts`：审核过的小型证明文件；发布副本保持字节，文件须在项目内，每个不超过 8 MB。
- `qualification_receipts={"cpu": "<绝对路径>", "real_first_source": "<绝对路径>"}`，两份均须列于上述证明与发布白名单。
- `training_updates=0`、`policy_selection_updates=0`；`inference_functions_changed`、`original_values_changed`、`row_hashes_changed`、`model_weights_changed` 均为 `false`。

CPU 兼容资格状态为 `REAL_NUMERIC_SNR_COMPATIBILITY_PASS`。首源状态为 `REAL_FIRST_SOURCE_PARITY_PASS`，要求 `parity_passed=true`、`synthetic=false`、整数 `source_index=0`、`original_source_row_ids_preserved=true`，以及 `checkpoints_bound`、`inputs` 的文件 SHA 映射。首源 `inputs` 应绑定恢复 manifest、原生 CPU 资格、原始表与缓存，不绑定后续会继续变化的操作回执。

## 发布范围

源副本进入 `experiments/metric-numeric-recovery-20261003`，证明与 `NUMERIC_RECOVERY_REPORT.md` 进入 `results/metric_numeric_recovery_20261003`。发布只允许这些显式文件及 `release_manifest.json`，运行仓库核验、CPU 检查和本目录测试，再正常提交、推送并验证远端提交。

输出回执为 `outputs/METRIC-NUMERIC-RECOVERY-20261003/publication.json`。保留已有失败、缓存和结果，检查过的 `COMMITTED` 状态支持同字节推送重试。此入口仅发布工程恢复证明，不启动 GPU，不评分，不新增科学结论。
