# VAR 通信报告

## 2026-09-12最新选型结果

- `backbone_selection_review_2026-09-12.md`：有界骨干选型的中文复核与迁移判断。
- `backbone_selection_result_2026-09-12.md`：自动生成完整数值结果。
- 结论：暂不迁移到当前XQ配对，WeTok保留完整codec参照；不自动追加训练。

本目录既承接新报告，也原样归集了七份历史VAR相关报告，供当前VAR通信主线直接查阅。
原文件仍留在旧目录，复制件与原文件逐字节一致，来源和SHA见 `../historical_sources.json`。

## 当前阅读入口

- 已授权两阶段协议：`prefix_refinement_protocol_2026-09-08.md`（同模型/optimizer与更新预算；2000+8000阶段，LPIPS选模附PSNR guard；状态见项目PROGRESS）
- 现有校准与固定m8复核：`prefix_calibration_fixed_m8_review.md`（只有两个校准端点；next第二轮LPIPS退化，固定m8数字链仍有明显优势；无续训/重选）
- 最新训练型结果：`learned_prefix_result_2026-09-07.md`（两版本真实训练完成，next相对parallel有增量，未胜强系统）
- 本轮训练协议：`learned_prefix_training_protocol_2026-09-07.md`（m8全局发送/逐尺度读取，冻结大模型但保留输入梯度）
- 最新整帧末尺度结果：`whole_frame_prior_result_2026-09-07.md`（全量完成，未胜列表65，停止当前4前缀候选）
- 整帧预注册：`whole_frame_prior_preregistration_2026-09-07.md`；API补测边界：`whole_frame_latency_protocol_2026-09-07.md`
- 实施前建议核查：`whole_frame_prior_review_2026-09-07.md`（历史建议记录，后续已完成）
- 最新7 dB失败输出回放：`crc_failure_replay_result_2026-09-07.md`（追回64.74%差距，但二候选oracle仍输整帧，停止该选择路线）
- 本次回放预注册：`crc_failure_replay_preregistration_2026-09-07.md`
- 最新完整实验链：`next_scale_decoding_result_2026-09-07.md`（同观测机制有收益，强系统对照未胜出）
- 单尺度译码预注册：`single_scale_channel_preregistration_2026-09-07.md`
- 完整链路/强对照预注册：`progressive_channel_preregistration_2026-09-07.md`
- 新候选第一关结果：`next_scale_prior_result_2026-09-07.md`（先验log-loss通过，尚无FEC增益）
- 新候选第一关预注册：`next_scale_prior_preregistration_2026-09-07.md`
- 基线：`var_fixed3060_adaptive_scale_result_2026-09-04.md`
- VAR补全的实际作用：`var_semantic_rate_contribution_result_2026-09-04.md`
- 跨数据集与实例保留：`var_receiver_prior_cross_dataset_result_2026-09-04.md`
- 接收端候选负结果：`var_prefix_consistency_result_2026-09-06.md`
- 该候选的原始预注册：`var_prefix_consistency_preregistration_2026-09-06.md`
- 无线误码诊断：`var_m8_wireless_diagnostics_result_2026-09-04.md`
- Decoder微调权衡：`var_decoder_only_m89_result_2026-09-03.md`

## 历史路径说明

为保持七份历史报告完整性，没有改写其中的旧路径、日期、主线描述或运行命令。
这些历史报告中的 `outputs/...`、`configs/...`、`scripts/...` 等路径相对于原工作区：

```text
/workspace/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc/
```

这些命令不是新项目的运行入口；直接执行可能写回旧目录。
当前记录位置以本项目 `README.md` / `AGENTS.md` 为准，不以归档报告中的旧说明为准。
新候选三关的预注册与结果报告相对于VAR_COMM，不适用上述旧工作区路径。
后续新报告直接写入本目录，并使用新的明确产物路径。
