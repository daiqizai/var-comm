# Joint网格结构控制

遵守VAR_COMM根规则。本目录两臂5000训练、全部质量/统计/CPU复核与独立计时均已完成并冻结，不重启。结果见`docs/CONTINUE.md`；当前新接续为`../wetok-joint-sufficiency-r2/docs/CONTINUE.md`，仅处于四臂同配方续训准备状态。

不修改旧Joint/R-only/geometry的源码、配置、模型、optimizer或结果，不重训已完成三基础臂。普通16/16/16迭代不得改名为next-scale。首个新臂只把读取网格从4/8/16改为16/16/16，保持相同通信参数与自身历史；第二个新臂是既有full-grid innovation的Joint E/R控制，不引入新损失或新视觉模型。

数据、源输入、N/E、无反馈、名义SNR、连续接口、原父点、配对采样、新增更新机会和选模规则须与当前Joint基线一致。任何GPU启动仍先核实授权及其它工作；不得抢占、停他人进程或改设备设置。本准备不等于新试验已完成或研究总目标达成。
