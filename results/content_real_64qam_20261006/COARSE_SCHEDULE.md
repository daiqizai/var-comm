# CPU 测表接续登记

初始 GPU 批次保持原登记。新增独立 `coarse_execution_v1`，仅执行两个 CPU worker 的正文误块率测表及合并，共 8 个实际布局 × 3 个 SNR × 2048 次 = 49152 次译码，使用预留的 coarse 额度。

等待器已在服务器启动，核验状态为 `WAITING_INITIAL_H_BATCH_EXIT`。它只在初始批次完整完成、原 owner 与所有 worker 退出、完成凭证及输出 SHA 核验通过、共享预算账本没有未完成调用后启动测表。失败或 STOP 会保留证据并停止，不自动重试。等待阶段不占视觉 GPU，不执行译码。

- 新执行登记 SHA：`6e13b9cd20713a7e69f87bbd6c1ad9e88e8316c0ea67282e15f4159f5fa2664a`
- 新 owner 配置 SHA：`7ce7a7cde64c6d49eb41f8fa3fe4405ca3d53e5a46641d334fc741de29452906`
- 等待配置 SHA：`94815534501b6270d0d621352b4e955a670d5fe42df0232e551ecd0ddbe2d8b7`
- 部署后的测表、等待器及原 owner 共 31 项 CPU 测试通过。

本登记中的 `report` 阶段仅指 CPU 表格合并，不是 H 最终报告。有限真实载荷校准、完整主系统校准、development、在线计时和 C-REAL 仍须按冻结科学方案接续实现、登记及核验。
