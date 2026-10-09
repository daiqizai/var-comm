# A0 复用矩阵与真实缺项

本地只读核对快照：2026-10-09T10:40:12+08:00（Asia/Shanghai）。矩阵共 **150 行**，状态分布：{'REUSE': 121, 'MISSING': 15, 'COMPLETE': 3, 'UNSUPPORTED': 2, 'NOT_RUN': 9}。本次没有 SSH、GPU、模型推断、信道仿真、统计重采样或科学结果修改。

`reuse_matrix.csv` 按方法、数据集、N、SNR、接收规则、指标/交付物、缓存与凭证列出。`evidence_sha256` 是本次实际读取的本地文件哈希；相对路径以仓库根为基准。远端缓存路径只是现有凭证中的位置，本任务没有重新核实远端进程。新 A1/A2 由主代理调度；本快照随后收到并核验A2实际完成凭证，已更新。表里的 MISSING 表示本地尚无完整结果凭证，不能据此重复启动，也不等同于远端没有运行。

## 可以直接复用

1. 原发布 R6 的 36 个方法/SNR 点：6 种输出×6档、500来源×3噪声，summary 756行、paired 882行。summary/paired/points/pairs/fairness 五文件哈希与既有作图验证完全一致；所有单方法点含21字段、四个核心指标完整。两个核心消融和原45个图格式文件保持原版本。
2. **原分辨率 native256 BPG 已实际完成**：500来源、9000计划帧、24个四指标汇总；现有只读QA记录505/505本地输出绑定、500个来源检查点、1283次唯一质量计算及7717次相同图像复用。原14-MCS/100校准源PSNR策略限制仍保留，不能扩大成自适应降采样或全MCS最优结论。这里并非“BPG尚未跑”。新文档A1a/A1b的多分辨率路径是另一项。
3. 三方法 **8源** endpoint单输出计时已完成：raw partial、连续JSCC、Swin，六档SNR，每源测量1次、每档另2次预热；所有48个原接收迁移检查/方法已完成。这不是文档要求的五方法、旧校准16源、7/13/19dB、每case测3次的完整A3。
4. A4冻结配置24行、BPG配置6行可以直接整理：实际k/n、源码/控制位、header/body/padding、allocation_modes已有。配置表属于发送前计划，不能替代实际接收错误/能量分布。
5. A5已完成13dB五方法development可视化两组（8张源），同工作点16源共有缓存清单；四列机制图目前是13dB。common500的13/19dB三方法图也已导出，其中19dB分类预测变化案例是明确标注的事后诊断。历史HiFi16与common500不重叠，development图不能冒充holdout。
6. 100源六条件参照已有 **126行=6×21指标**：identity、same_class、unrelated、gaussian_blur_sigma1、gaussian_noise_sigma2_255、jpeg_quality90。直接复用均值和区间，不补跑评价器。
7. A6作者原文/源码核对已交付；native真实复现仍NOT_RUN，仅ARPC作为一个候选；作者源码表支持256，默认demo却为1024且码长分母写死，付费无线接口未实现。

## 文档要求仍缺的交付

- **A1**：新增32校准源×多分辨率源码摸底的正常结果、源码规则与MCS冻结、真正有变化的新码流校准/holdout。原BPG paired差值表在本次检查的交付目录中未找到；源均值已存在，若需要新增配对统计应只做这部分。
- **A2已更新完成**：`results/paper_supplement_20261008/a2_swin/native_v1`的20源native同观测对齐PASS，最大RGB差5.364418029785156e-07、uint8差≤1，20次native AWGN forward、0付费PHY。本机核验四个交付文件与completion哈希一致。C7未训练、C13正文超N1024有实际support_scope凭证；无新增受支持策略，旧C6复用。19dB超训练/校准范围及头256未证最优的限制照旧。
- **A3**：完整5方法16源指定重复协议、单输出软件端到端、P95/峰值allocated与reserved、参数和权重存储。原fixed16 raw成本包含VAR和direct双输出，明确写着single_arm_end_to_end=NOT_MEASURED；不可借该总窗口填单臂时延。
- **A4**：预算堆叠/源信息效率与质量图、真实帧能量分布、token错误率分位、头拒绝/正文CRC/错接受/灰图与各状态质量的论文级汇总。先从原登记/接收记录导出；本地核对没有把全部500源接收记录重新下载，也不据此要求新链路。
- **A5**：10/19dB四列机制图、同development源BPG列、参考指标表与主体/数量/属性/布局的简单错误类型说明、完整逐图指标展示。现有图像provenance足够定位缓存，但图像来源表不是逐图指标表；典型示例不推导总体成功率。

`MISSING` 只说明要求尚无完整可复用交付，`UNSUPPORTED` 说明当前检查点/实现不覆盖，`NOT_RUN` 用于未启动或条件触发的范围，`REUSE` 是已完成旧资产可用，`COMPLETE` 是本轮独立交付已实际完成。所有准备脚本、request和execution材料均不算实验完成。B/C类按用户文档的条件触发；本矩阵不构成新训练、新模型或重复测量授权。

## 核对凭证

- 原R6镜像：`.research/main_raw64_20261007/take_over_v1/final_publication_r6/actual_staging_r6/results/main_raw64_20261007/final_common500_r6`；summary SHA256 `5899fe10ccf2e7fe93ef36c59b9e14c6622f542d58ee1c0bbf0c8332a5577859`，paired SHA256 `dd8221b1fe8aa9d86f9cbf78ea281fd2da860e2ffdcd6fb83daa487eebc0c235`。
- 原作图验证：`paper/figures/mainraw64_holdout500/validation.json`。
- BPG实际完成/本地验证：`.research/main_raw64_20261008_paper_supplement/bpg_delivery_actual/ACTUAL_BPG_DELIVERY_QA_V1.json`。
- 新计时原始case与正常完成：`.research/main_raw64_20261008_paper_supplement/timing_audit/actual_results_v1/{P1024,SwinJSCC80k}` 和 `actual_results_r3/RAW64_PARTIAL`。
- 配置来源：`paper/tables/mainraw64_supplement_20261008/sources.json`。
- 原参考表：`.research/main_raw64_20261007/take_over_v1/final_publication_r6/actual_staging_r6/results/main_raw64_20261007/final_common500_r6/development/reference_summary.json`。

这是一份有时间边界的复用/缺项快照；主代理收到后续A1/A3等完成凭证后应更新对应行，不能把本文件当长期监控器。原输出文件均保持只读。
