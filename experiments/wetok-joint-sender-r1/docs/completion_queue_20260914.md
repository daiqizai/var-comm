# Joint5000自动收尾工具

2026-09-14新增`../scripts/finish_registered_trial.py`，不修改当前训练、选模、评测或统计实现。它只接续已登记的5000更新实验，不开启新的训练候选。

## 激活边界

先完成2500完整校准/审计，人工依据记录作继续决定；若三臂共同继续5000，再向收尾工具提供实际5000 trainer和review watcher的PID。工具核对当前用户、脚本、5000里程碑、对应trainer及进程启动标识，不接受2500队列作为最终评测前提。

5000训练和review均退出后，调用既有`matched_milestone()`核验相同5000机会、实际模型/Adam、校准选模和冻结引用。未通过即停止收尾，保留失败现场，不篡改校验结果。

GPU存在其它计算任务时等待，不停进程、不抢占；空闲后运行原`evaluate.py --step 5000 --execute`，再执行原`analyze.py --step 5000 --execute`。比较保留三Joint＋全部六R-only＋七个历史系统，共33600主行、600支持点行和900无噪诊断行。训练数据、1000校准、原100 development与物理账本不改变。

## 重入与记录

- 独占收尾锁避免本工具的重复实例；实际评测仍使用原GPU占用检查。
- 已有完整阶段先校验回执、先决条件、源码和全部产物SHA，不重新运行。
- 部分评测只走原有`--resume`；不自动删除、覆盖完整结果或修复统计目录。
- 失败写入明确状态；最终状态注明只是本试验评测完成，不是研究总目标完成。
- 日志/状态前缀为`outputs/WETOK-JOINT-SENDER-R1-step0005000-finish`。

6项新增CPU测试和原16项测试均通过，共22项；覆盖PID复用/僵尸、错误里程碑或trainer、回执与产物篡改、断点评测、已完成阶段不重跑、dry-run不启动进程。CPU测试没有占用GPU。绑定的训练metadata、profile和reference资格源码均保持原SHA。

工程回执：`outputs/WETOK-JOINT-SENDER-R1-COMPLETION-TOOLING-20260914/receipt.json`。该回执只证明工具准备完成，实际是否激活及PID以`CONTINUE.md`和进程为准。
