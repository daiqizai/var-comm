# 无可行端点：独立实现核验

**已核验，未改动协议。** N512 的 `choose` 在无可行源码端点时选择付费 NULL profile、发送两位零载荷；实际接收映射把成功收到的 NULL 计为灰图。N1024 原端点实现与后续 m10 实现均抛出 ValueError，未把无可行源码伪造为信道失败。已评估 N512 的 TX_NO_FIT 为零，已有 N1024 报告亦未出现此情形。

直接从冻结文件抽取原始纯函数，完成 7 项 CPU 合成检查，包括 no-fit、NULL→gray 和正常可行输入。它们是软件行为检查，不是新增无线资格或图像实验。函数、文件哈希及结果见 [implementation_status.json](validation/implementation_status.json)。

本工单要求核验，且仅在统一失败行为时要求修补。为保持现有两套冻结协议及历史结果，这次没有统一实现；正文 II.B/II.C 如实区分两种行为。所有已评测正常输出保持不变。以后若决定统一，需要独立版本与资格，不能只改文字。

复核命令：`python scripts/verify_no_fit_behavior.py --repository <repo-root> --output <new-receipt.json>`。脚本先核源文件哈希，再运行纯函数；不载入模型，不运行 PHY、指标或训练。
