# C 额外训练 seed：初始 20k 发布准备（2026-09-28）

本次新增独立 CPU 发布入口，固定 N4084、原登记 seed 2026092404/2026092504，以及已冻结候选 m6、直接控制 m8/pure。活动训练、调度和历史 worker 源码没有改动。此结果是发布工具及真实资产准备，尚不是额外 seed 的 20k 完成、正式 selected development 质量或完整在线计时。

m6/m8 使用原 short_prefix.train，分别为 H6-V/H6-P 和唯一 H8-V；pure 使用 scoped_N4084 的 token_efficiency.C_train，从零训练、parent=null/parent_updates=0。不得沿用第一 seed pure 的历史 10k parent。三组均需从 step 0 起九轮全校准，m6 为 270000 条，m8/pure 各 135000 条。工具只支持初始 20k，后续 10k 延长仍需独立发布支持。

## 已实际核验

第二个 seed m6 的 step 0 完整 30000 条原校准记录通过 source/SNR/noise 完整键、公式、均值与两臂失败一致性检查。原真实 GPU qualification 身份已核对，但本 CPU 工具未运行新的 GPU 验收。54 项合并源码/资产绑定、两 role 缓存 completion 与 210 份 shard 元数据、10 个校准 tensor SHA、第一 seed 已发布索引及冻结候选 SHA 均一致。训练大 tensor 未全部重哈希，保留原 loader 验算范围。

两臂当前 selected=0，utility 分别为 .06516972769579539 和 .09553400774300098。真实 checkpoint 在 CUDA 屏蔽下完整加载，确认 step 0 optimizer 尚为空，以及模型、order、channel RNG、torch/CUDA RNG 恢复字段；非零更新必须有已填充 optimizer。此前热暂停后的正常恢复继续由原 guard 负责。

N4084 付费资源按原 m6/m8/pure 账本核验；E8168 是登记约束，不是此校准中新测量的逐帧能量。多轮缓存复用评分不等于独立数字 PHY 传输。图像 bootstrap 和训练 seed 变异保持分开。新 holdout 和内容选择器继续暂缓。

## 使用与边界

准备入口：
```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -m tools.publish_c_repeat_initial --group m6 --seed 2026092404 --prepare-only
```

只有原 20k 全校准完成且正式 at_20000 决策存在后，去掉 --prepare-only 才能发布；--audit-only 对完整边界仅核验。优先要求原成功 stage 的不可变 snapshot、准确引擎/seed/命令、returncode=0。若缺回执，仅显式 --allow-thermal-receipt-gap 才允许在原内外 launch/已退出身份、双方热证据、严格时间顺序、完整无失败日志、原重启和正式决策均通过后封存真实终端资产。缺失原 stage 和 process_returncode=null 必须如实保留，不伪造回执或 finalization，不重复训练补回执。

发布目录 results/token_channel_efficiency_20260923/C_initial_milestones/N4084_<group>_seed<seed>_20k 不可覆盖；将保留全部原 CSV、源均值、SNR/分项曲线、两幅校准图、原账本、selected/lineage、边界证据和逐文件 SHA 索引。正式发布分支本次未执行。m8、pure 及第三个 seed 的真实准备核验等待原 registration/资格/校准生成，未使用合成替身充当资产验收。

十项回归覆盖六个合法 scope、原 seed/N3060/m7 拒绝、fresh pure parent 与引擎、m8 单臂、RNG、完整 20k 边界、step 0 与非零 optimizer、全历史 selected、错误 stage 命令、缺回执显式门控和 pure CRC not_applicable。根目录完整 238 项 CPU 测试、3440 项仓库文件检查、release、fsck 与暂存 diff 检查全部通过，日志位于 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/monitoring/C_repeat_publisher_root_checks。独立远端验收仅在实际 receipt 成功后认定。

准备审计：outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/monitoring/C_repeat_m6_seed2026092404_1790528600904056845.json。

- tools/publish_c_repeat_initial.py SHA256: 6bab3cdafd132f9337a6ac3a7a0cc4b73354571a7d65189830cc1968f7011a47
- tests/test_c_repeat_initial_publication.py SHA256: 3d26b1713e808953605c629a57ba83d9cedcc0a3a76ef59af02e88ed9af39f6e

第二个 seed 训练仍在推进。其余 seed、正式 selected 评测/诊断/计时、四个历史 GPU worker 和最终全方法配对报告仍待完成。
