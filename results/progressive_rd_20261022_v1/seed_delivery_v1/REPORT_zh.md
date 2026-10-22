# 渐进 token 率失真研究：种子库里程碑

独立后续研究，沿用冻结WCL模型与实际接收器；论文正式图表与结果未改动。

## 已实际完成

- 421个原缓存文件逐文件SHA一致，包含20,000训练源与1,000校准源。
- benchmark：32训练源×2端点；50次decoder更新。实际exit0，完整检查点已保存。稳态约21.19图/秒，保存约1.08秒；不是质量提升结论。
- 种子库：1,000训练源＋100校准源，各8个合法端点；1,100次TX概率遍历、8,800次独立源解码和原predict-overwrite-advance恢复。
- 8,800行MSE/PSNR、LPIPS-Alex、MS-SSIM，FP32完成latent、实际终止码流和接收token全部保存并SHA核验。tokenizer、PHY、训练调用在此阶段均为0。
- 八个分片实际退出0、父子进程均已退出、ledger无未决调用。端点包含完整m10；既有七个端点和m10编码均通过原实现逐位一致的合成检查。
- 第一批seed尝试因工具模块同名导入失败，发生于模型加载函数体前；8个model_load失败reservation均保留，科学调用0。独立v2隔离导入后执行；没有重跑成功科学调用。

## 当前已启动，尚非完成结果

三组匹配decoder：多前缀无条件、端点条件、等新增训练量旧配方。每组同一Dc初始化、1,000训练图、5,000更新×batch4；同源顺序、同参数量，MSE+0.1LPIPS。条件仅来自公共端点profile。旧Fb_TX只作为历史旧配方训练输入，不作为当前RX。只在固定calibration100×8做最终评价，不选中间最优检查点。

缓存扩展为另外18,968个未处理训练源×8。GPU3–7先执行，GPU0–2等待各自训练真实完成后执行。最终可形成19,968训练源×8，加保留的32图×2 benchmark；不会为凑整数重复原benchmark的成功TX遍历。共享绝对资源deadline不重置，未启动任何encoder/VAR/codebook训练或无线实验。

## 数据与恢复

`source_rd.csv`保留逐源原精度；`endpoint_means.csv`仅分split汇总，不是无线质量曲线。缓存模型、预处理、源ID、实际码流、接收token及latent身份由registration和closure审计绑定。

大数组、checkpoint和日志位于共享PFS progressive_rd_20261022_v1，资源收回后仍可访问；未做本机备份。expansion原完整登记因13.7MB不进入Git，去重元数据expansion_registration_compact.json可由restore_expansion_registration.py逐字节恢复，SHA不变。不是删去登记内容或更改过滤规则。
