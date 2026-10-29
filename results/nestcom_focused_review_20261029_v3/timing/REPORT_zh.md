# Quarter / Eighth TX 计时：真实失败与部分诊断

状态：blocked，完整 benchmark 未完成。不能将本目录称作 100 源×3 重复的完整实测，也不能用它宣布速度优劣或最佳粒度。

原上限 610 次 TX：4 资格、6 warmup、600 测量（100 源×2 网格×3 重复），两小时绝对截止含加载；无重置。共享 PRO6000，固定设备、线程、CPU affinity，交错网格顺序。总 TX 计时涵盖驻留 RGB 到完整波形的预处理/H2D/tokenizer/prior/CDF/算术编码/独立终止/装入/LDPC/QAM/持久 CPU IPC，不含文件读取、模型加载与持久审计写入。

第一执行因资源门限在科学调用前退出；独立资源准入修订后，第二执行因旧 checkpoint 路径在模型前向前退出。原文件/哈希/失败与真实 wait=1 保留。第三执行只修订为已迁移且字节 hash 一致的模型路径，没有改变模型数学。

第三执行实际完成 4 次资格和 6 次 warmup。前 88 个源的两网格首重复共 176 个测量通过 token/CDF/码流/端点/波形身份检查。随后第 89 个源（原 index 114，`n02123394/ILSVRC2012_val_00013785_n02123394`，`measured/088/4/0`）新 tokenizer 输出不满足既有 token 身份，立即停止。未将缓存 token 替代新结果，未跳图，未自动重跑。根因尚未确立，不能将其写成已证实的数值精度原因。

总计 187 次 tokenizer、1,870 次 prior-scale、187 次完整 TX 波形；186 次完整身份通过、1 次身份失败。测量尝试 177、有效 176；剩余 423 次 TX 未尝试。没有 RX、质量评分、噪声、训练或 bootstrap。GPU worker/owner 真实 wait=1 且已退出，CPU TX worker wait=0 且已退出；原 ledger 的一项未完成调用原样保留。

部分 88 源×1 次结果：Quarter TX 均值 289.465 ms、中位数 280.975 ms、p95 346.681 ms；Eighth 287.541/283.283/331.781 ms。Eighth−Quarter 源配对均值 −1.924 ms，中位数 +1.683 ms。它们仅是共享设备上失败前数据，未完成规定总体，不作性能排名。

两网格每图均 10 次 prior-scale、680 个 token 单次编码，分别 25/49 次独立终止。`partial_*` 不做质量增益与耗时的正式关联；已有固定质量控制的 LPIPS/PSNR 增益和 DreamSim 跨零结论保持原样。恢复前必须先解决 source-token 身份差异并另行确定不重复已成功调用的方案；本交付没有启动后续。

`actual_compute_and_reuse.json` 与 `receipts/` 是计数、模型、数值路由、资源采样、失败、退出状态和 SHA-256 的权威记录。大日志和数组留 PFS。
