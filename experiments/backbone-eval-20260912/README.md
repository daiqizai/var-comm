# 配套视觉骨干的一次有界、无训练选型

任务：`ei-liulu-xqvar-eval-20260912-v1`。日期按本机 Asia/Shanghai（UTC+8）记录。
用户最后授权：**在本机既有 GPU 任务自然结束后运行**；不再连接需认证的开发机。

- 不启动新训练，不继续扩大原版 VAR 模块/训练；不打断上一轮已运行的2×2整条流水线。
- 只读使用历史100张 development、模型、统一评测器；不改变旧项目/源码/checkpoint/结果。
- 新源码、私有依赖、候选下载全部隔离在本实验目录；运行输出在 `VAR_COMM/outputs/ei-liulu-xqvar-eval-20260912-v1/`。
- 本目录是独立评估分支，不是新顶层项目或对旧 Git 的切分/提交。

## 入口

- 冻结范围与判据：`docs/protocol.md`
- 实际配置：`configs/eval.local.json`
- 自动等待配置：`configs/queue.local.json`
- 无CUDA监视器：`scripts/wait_and_run.py`
- 单次顺序执行：`scripts/run_evaluation.py`
- 四种模型共用评测：`scripts/evaluate_backbones.py`
- 配套适配：`scripts/xq_adapter.py`、`scripts/wetok_adapter.py`
- 分析：`scripts/analyze_backbones.py`
- 资产准备完成条件：`docs/assets-ready.json`，只能由校验脚本生成，不能手工触碰空文件绕过。

监视器只在上一轮 supervisor 已退出、其 pipeline 达到终态、资产/代码校验通过、同一GPU连续3次空闲（60秒间隔）后启动。
不使用 SIGSTOP 占住GPU；不杀他人任务；不改驱动、功率、持久化设置或系统依赖。运行中检测外来GPU进程则只退出自己的评估进程组。
没有共享调度器时轮询不能原子预留GPU，不能保证绝对零瞬时竞争；不把监控写成公司级资源隔离保证。

**已于本机2026-09-12 01:49完成，09:45独立CPU产物/统计复核通过。** 本轮不正式迁移到XQ；WeTok只保留完整codec参照。决策解释见 `../../reports/backbone_selection_review_2026-09-12.md`，正式结果及复核保存在本任务输出根目录；不追加训练。
