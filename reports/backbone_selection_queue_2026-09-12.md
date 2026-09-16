# 配套视觉骨干选型：本机排队执行记录

本机日期：2026-09-12（UTC+8）。任务名：`ei-liulu-xqvar-eval-20260912-v1`。

## 当前授权与实际状态

用户最后明确改为本机，等现有GPU任务结束后再运行。未对原GPU任务发送任何停止信号；公司开发机连接在认证前超时，未修改远端。

- 当前monitor PID：1856483；每60秒只读查询原完整pipeline及指定GPU。
- CPU准备supervisor：1869580；在当前XQ单流下载完成后依次下载WeTok、验证SHA/源码/config/100图资产。
- 本轮新增模型GPU评测还未开始；尚无新PSNR/LPIPS/DINO结果与迁移决定。
- 78项CPU-only检查已通过，包含缩小随机模型的XQ官方API/多分支/补全回转；不把它当真实预训练性能。
- 独立XQ tokenizer权重已下载并通过LFS SHA、CPU参数/配置元数据读取；生成器与WeTok权重仍按串行任务准备。

## 明确的自动执行条件

源码/依赖/资产校验receipt完成，旧2×2的整个train/calibrate/evaluate/analyze流水线到终态且supervisor退出，同一GPU连续三次空闲；然后才启动本轮2图smoke和100图评估。
不在旧train/eval之间的空隙启动；不使用多GPU、并行推理、训练或修改旧decoder/通信参数。
单GPU、batch1、2 CPU线程、nice10、CUDA allocator上限70%；探针异常或外来GPU计算进程出现，只退出自己的评估进程组。缺少共享调度器时不能原子预留GPU，因此不能将轮询保护表述为零瞬时竞争保证。
任何资产/strict加载/工程检查失败都会停止，不静默换checkpoint、放宽断言或开始训练。

## 范围、账本与输出

完整四臂：原官方、已有fidelity、XQ实际配套tokenizer、WeTok指定8192-bit完整codec。
XQ只用真实自然prefix8/9：静态2424/3960 rawbits，full6864；全含两个PQ分支。与旧3060/5088/8160是**源表示**比较，非3060complex uses等资源通信实验。
同100图、旧预处理、同PSNR/SSIM/LPIPS/DINO；记录浮点重建、逐图差值、耗时、显存、困难样例与离散率—质量图。
主补全为类条件argmax，无CFG混合；作者采样参数仅预定三种子的单独敏感性，不选择最有利种子。类标签、XQ的DINO训练暴露均披露。

- 协议：`../experiments/backbone-eval-20260912/docs/protocol.md`
- 队列实时状态：`../outputs/ei-liulu-xqvar-eval-20260912-v1/queue-status.json`
- 资产实时状态：`../experiments/backbone-eval-20260912/docs/prepare-status.json`
- 最终分析（只有成功运行后才存在）：`../outputs/ei-liulu-xqvar-eval-20260912-v1/analysis/report.md`
- 成功后自动发布小型报告：`backbone_selection_result_2026-09-12.md`

原有性能/通信基线不变。不自动迁移主线；是否采用XQ须看实际prefix+completion的PSNR/LPIPS，而非只看生成FID或DINO。后续如采用新视觉骨干，同骨干数字/学习通信的独立对照仍必须重做。
