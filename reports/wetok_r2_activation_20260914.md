# R2实际续训启动：四模型与Adam不重置

2026-09-14已完成四个真实5000模型/Adam逐值恢复检查，并启动同配方四臂续训。没有改变旧模型、旧结果、损失、学习率或物理预算，也没有重新跑起点完整校准。

## 已执行

- single、4/8/16 state来自原Joint5000末端；普通full-grid state与full-grid innovation来自Grid5000末端。四个模型和Adam moments/step均与封存端点一致。
- 初始Adam为5000步，第一条新批次来自global index12000，实际更新后global step12001；不是从7000重新采样，也不是新建optimizer从step1训练。
- CPU真实driver测试包含非零Adam恢复、跨边界数据续接、不重跑已有校准、保留旧最优点，以及中断恢复和连续运行的模型/optimizer逐值一致。两项测试通过；最初fixture缺少历史耗时字段，已在测试数据中补齐，未修改旧训练数据。
- 20:09启动trainer769675，reviewer785157、被动观察器785158。20:10已实际保存5001步；之后读取真实5500保存点，核实四臂Adam均为5500、global12500、前5000历史不变，所有模型确实更新。不是只启动了空进程。

## 当前范围

先到总7500步并完整校准审计，再依据结果共同推进10000。新每500步监控不用于选模，完整校准只新增7500/10000；原5000及更早合法全校准记录冻结继承。

原MSE＋0.01LPIPS＋0.01bit BCE＋0.01state、1e−4 AdamW、batch4/micro1、20k训练源、1k校准、五SNR、连续接收接口、N3060/E6120和视觉冻结均保持。普通迭代不称为next-scale，更多训练不当作新机制。旧no-history/R-only/强系统结果保留为历史参考，后续不得混称同10000更新机会。

当前仍没有R2的正式全校准或development结果；不访问新正式测试集。所有进度以实际进程和`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`为准。

证据：`outputs/WETOK-JOINT-SUFFICIENCY-R2-PREPARATION/resume_initialization.json`、`actual_activation.json`与新训练目录中的`initialization.json`/`resume.pt`。原5000文件保留，不迁移或覆盖。
