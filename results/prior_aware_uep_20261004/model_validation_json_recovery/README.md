# 模型验证结果的 JSON 写入恢复

2026-10-05，UEP N1024 完成实际链路、9000 行图像指标、1200 行 P1024 对照以及独占接收计时后，模型验证在保存结果时停止。原统计函数返回的 41 个布尔叶子为 `numpy.bool_`，严格 JSON 编码不接受此类型。

恢复直接使用冻结策略、原信道表和已有指标，重新执行原统计函数；没有重跑图像或信道，没有重新选择策略。实际结果复现了原写入错误。只在输出边界将这 41 个 NumPy 布尔值转成原生布尔值，记录每个字段路径并逐值核对；不支持的其他类型及 NaN 仍被拒绝。所有原始判断、阈值、数值及输入文件保持不变。

`qualification_summary.json` 记录实际验证与字段清单；`execution_registration.json`、`completion.json`、`failure_archive_receipt.json` 和 `owner_resume_launch.json` 记录结果写入及恢复过程。原 owner 和 worker 的失败证据继续保留。`.py.txt` 文件为实际恢复代码的逐字节快照，SHA256 见 `source_snapshot_manifest.json`。

模型验证完成与验证通过是不同含义。此次四个 SNR 的结果均为 `UNCERTAIN`；恢复保留这一判定，扩展决定由原登记规则执行，不增加样本或修改门槛。
