# C 额外 seed 10k 延长发布准备（2026-09-28）

新增独立 CPU 工具 tools/publish_c_repeat_extension.py，固定 N4084、原额外训练 seed 2026092404/2026092504、冻结候选 m6 及直接控制 m8/pure。活动训练、调度和历史 worker 源码未修改。此项是发布准备，不是 30k 完成或新增 GPU 质量。

m6/m8 仍使用 short_prefix.train，fresh pure 使用 scoped_N4084 的 token_efficiency.C_train；parent=null/parent_updates=0，不套第一 seed pure 的 10k 继承。只接受 30000 起每 10000 更新的正式边界。m8/pure 单臂每次新增四轮 60000 条，m6 双臂新增 120000 条，完整历史重新验算，旧 CSV/元数据逐项校验发布 SHA 后引用。

正式发布要求此前每个边界均已发布并通过全部索引 SHA、同 scope/registration/真实 terminal checkpoint 核验，且原校准决策确实授权下一次 10k。当前边界必须有原成功 stage 的准确命令和不可变 snapshot；活动 completion 之后更新不会被误判历史损坏。若缺原 stage，仅显式 --allow-thermal-receipt-gap 且原内外 launch/实际退出/双方热证据/严格时间顺序/完整无失败日志/原重启/正式决策均核验后封存真实资产；明确 stage 缺失与 process_returncode=null，不构造调度回执，不重复训练补回执。

完整边界将重审所有 source/SNR/noise 键、公式、均值、失败、全历史最优 selected、前次 parent/terminal/selected 的真实 CPU 模型与填充 optimizer、order 和 RNG 恢复状态。真实资源账本按 m6/m8/pure 分开；E8168 为登记约束，不是缓存评分新增的逐帧实测。校准缓存复用不等于独立数字 PHY，图像 bootstrap 不包含训练 seed 变异。

## 本次真实准备证据

只读准备已重审第二 seed m6 的 step 0/2500 两轮 60000 条完整原校准记录；两臂 selected=2500，实际 checkpoint CPU 加载与填充 optimizer 恢复字段通过。54 项合并绑定、两 role 缓存 completion、210 份 shard 元数据、10 个 calibration tensor SHA、第一 seed 历史索引和冻结 candidate 身份通过。训练大 tensor 没有全部重新哈希，仍明确限定为原训练 loader 的验证范围。

准备检查发现当前额外 seed 初始 20k 发布尚不存在，显式 missing_previous_publications=1、previous_boundary_verified=false、parent_checkpoint=null、boundary_published=false。这是尚未到达的训练边界，不是实验失败。正式 collect 会拒绝缺失的前序发布。m8、pure 和第三 seed 的真实准备必须等待原 registration/实际资格和校准生成，未运行替身评测。

## 入口与验收边界

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m tools.publish_c_repeat_extension --group m6 --seed 2026092404 --until 30000 --prepare-only
```

先使用已发布的 tools.publish_c_repeat_initial 完成实际初始 20k 交付；只有原 30k 全校准及决策实际完成，才去掉 --prepare-only 进入正式发布，--audit-only 只核验完整边界。后续 40k 必须先发布 30k。目标 C_extensions/N4084_<group>_seed<seed>_until<step> 不可覆盖，正式发布分支本次未执行。

八项新回归覆盖六种登记 scope/引擎与 checkpoint 名称、第一 seed/N3060/m7 及错误边界拒绝、缺前序发布、错误 seed/继承 parent/停止决策拒绝、单臂完整性、不可变 snapshot 优先、缺回执显式门控、历史记录归属与 SHA、20k 前准备的诚实缺项。首轮错误命令测试因测试自身共享同一 list 引用而未触发拒绝，改为复制测试输入后八项全部通过；生产校验源码无需修改，不是 GPU 实验失败。

根目录实际 246 项 CPU 测试、3443 项仓库文件检查、release、fsck 与暂存 diff 检查全部通过，日志位于 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/monitoring/C_repeat_extension_root_checks。独立远端完整验收以实际成功 receipt 为准，不提前称远端通过。后续 selected 正式 development、诊断、完整在线计时、四个历史 GPU worker 和最终全方法合并报告仍待完成。新 holdout 和内容选择器继续暂缓。

准备审计：outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/monitoring/C_repeat_extension_preparation_m6_2026092404_1790530596025929319.json。

- H6-V selected 2500 utility 0.02499727077248196；checkpoint SHA 3d0c7f404cfd88cfde93051602afd9bf3d1e7132dc58dd3f9456da34a91ff37e。
- H6-P selected 2500 utility 0.024319408000012237；checkpoint SHA 3d0c7f404cfd88cfde93051602afd9bf3d1e7132dc58dd3f9456da34a91ff37e。

- tools/publish_c_repeat_extension.py SHA256: e73346ce543df8ed4021aa658824738c6a8ab4829e7d7cfc8d01818e4da98fe0
- tests/test_c_repeat_extension_publication.py SHA256: 12ceeca77e421688ffac4d567f0cb614de096fe5c3afaace497bbf0b1bd96654
