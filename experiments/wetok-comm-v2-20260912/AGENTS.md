# WeTok通信A0/A1：v2研究任务

最新按本机UTC+8日志2026-09-13进入global geometry比较：204×30新三臂训练、153×40合格历史控制只读复用，均continuous_mean。入口docs/geometry_execution.md。原接口5000已完成并冻结，不再默认启动旧九臂。所有新比较仍遵守N/E/信息/训练资格与当前冻结源码边界。

2026-09-12 16:06当前进入v2 B接收接口修订：三结构×hard_identity/hard_bounded/continuous_mean，见docs/interface_protocol.md。continuous仅限这一明确注册、训练/评测一致的分支，不能当native码字或免费上界；下方保持native符号的要求继续适用于两个hard分支。原bit权重六臂已探索性提前停止，不自动恢复。

遵循项目入口AGENTS和`../../../next_scale_communication_research_brief_v2.md`所指的工作区任务书（工作区根路径优先）。
只在已授权本机GPU运行；不修改既有视觉选型vendor/checkpoint或旧VAR结果，不更改全局依赖。使用既有视觉选型私有venv，新增输出位于VAR_COMM/outputs/。
每轮先冻结协议/配置。A0与A1及无历史控制必须有相同源输入、N/E、训练机会、可用观测与明确参数/计算对照。不要用多次完整y读取冒充逐包渐进传输。
保持native±1最终量化特征，所有GT特征只用于TX和损失；接收API不得带源图、原始Fq、类别或oracle误码。中间4/8状态是连续通信估计，不是原生WeTok token。
视觉参数冻结但Decoder输入梯度必须保留；检查不用官方inference_mode包装训练前向。现有WeTok版本没有随机生成Decoder，不添加无依据采样。
保存真实状态与失败记录，不把缓存/测速/工程通过当训练收益。训练里程碑不是研发上限，结合完整固定校准决定延长或改变假设；不因为旧20k或两epoch自动停研究。
实现先运行小范围CPU及真实梯度/identity/功率检查，再正常训练；不把它扩成无训练性能gate。修改文件使用apply_patch；不提交、push或搬动大型资产。
