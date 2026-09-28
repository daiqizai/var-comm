# 额外 seed pure 发布准备：原 F 缓存登记格式

2026-09-28。第二 seed 2026092404 的 fresh pure 已由原队列接续，使用 scoped_N4084 / token_efficiency.C_train，parent=null、parent_updates=0。真实训练资格验收已通过，probe 更新丢弃。CPU 准备首次在创建发布目录前因原 F cache registration 没有 bindings 字段而拒绝，错误日志保留于 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/monitoring/C_repeat_pure_seed2026092404_preparation.log。这是未绑定发布工具的 schema 假设错误，不是 GPU 训练失败。

仅修改 tools/publish_c_repeat_initial.py 和回归测试。先按原流程核验完整第一 seed 发布 index、原 F registration/receipt 与已发布逐文件 SHA 一致，再识别 float32、old_fp16_fhat_used=false、非空 historical source_snapshot、config/image manifest SHA。历史源码路径来自迁移前，不能将其重标为当前执行绑定，也不能要求旧原始 cache 登记具备后来混合 cache 的 bindings 字段。当前 repeat training/runtime 的实际源码绑定照常逐 SHA 核验；m6/m8 混合 cache 仍强制非空 bindings，不允许借 pure 分支跳过它。旧 cache/receipt/训练源码未改写。

新增两项回归覆盖错误 precision、布尔类型、缺失/空/坏 SHA 历史 snapshot、pure 错误 bindings，以及 m6/m8 缺失执行绑定拒绝。写入前确认发布器不在主链、四历史 worker 和当前 pure 注册绑定中；修改前等待 m8 30k 独立远端验证实际成功退出，避免影响在途代码一致性复验。首次编辑脚本因函数签名空格检查在写入前拒绝，纠正定位后才写入；不涉及实验变更。

修正后的真实 CPU prepare-only 已通过：15000 条 step0 完整 source/SNR/noise 校准键、公式、均值、纯连续 CRC not_applicable、全历史 selected 与实际 step0 空 optimizer/RNG 状态；55 合并绑定、210 shard 元数据、10 calibration tensor SHA、候选及第一 seed 已发布身份全部通过。大训练 tensor 未全量重哈希，仍限定原 loader 验算范围。selected step0 utility 0.1749371148382624，checkpoint SHA 934e41249ab131ab6db60ae7f654527428459578f2a9f80cfbd53986e1cf77f9。这不是20k阶段完成或正式 selected development 质量。

工具 SHA256：184d4a66c77eb4718202cebb0cb357bd41a30eadf14ed20043d6a99865d9cedf
测试 SHA256：782d3188ffd16984ec9ba2aa1876e152c7a3db5dddfc0c2f684e4a3913f2ebc1

CUDA 完全屏蔽，CPU 线程限制为2。使用现有 --group pure --seed 2026092404 --prepare-only；真实20k及原正式决策完成后才执行正式不可覆盖发布。所有后续 GPU 训练、selected 评测、计时及四历史 worker 保持原队列。独立远端复验实际结果见 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_pure_cache_schema_verification/receipt.json。CPU 工具验证不等同新增 GPU 质量或全实验交付。
