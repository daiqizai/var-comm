# SP0 共享存储与 GPU 核验（2026-11-19）

通过 A800 开发机读取共享 PFS 后，已核对 Pilot50 全部 50 份原生输入、680-token 数组和历史分段归档，覆盖 3,400 条双段码长记录。1,200 份历史干净重建的原生 float32 数组均与登记哈希一致。原始输入中曾缺失的 `0008.npz` 已找回。详见 [资产核对](audit.json)。

已为这 1,200 份重建及 50 个固定灰图计算 DISTS 和 DreamSim，共 1,250 个源图与输出配对。评分文件的 SHA256 已与阶段回执核对。[逐对结果](cached_scores_cpu.csv) · [计算回执](cached_scores_completion.json)

以预先固定的第一个源验证原生 VAR 数值路径。A800 上核对过权重身份，实际计算了十个尺度的整数 CDF。十张 CDF 和 96 条可对照的实际码流均与 H800 旧缓存有差异。CPU 的同源对照也得到十张不同的 CDF。这个检查只确定这些结果无法逐位复用，单凭它不能定位差异来自硬件还是软件设置。[A800 对照](native_a800_qualification.json) · [CPU 对照](native_cpu_qualification.json)

SP0 尚缺相容数值路径上通至 `(10,0)` 的 200 条条件增强码长及 50 个完整端点的质量记录。SP1 和 SP2 均未启动。单码字对照已按原表固定为 12 个 MCS 与 25 个 NeST 上限的 300 个组合，4、13、19 dB 使用同一清单。[选择记录](single_catalog_decision.json) · [候选表](single_codeword_300.csv)
