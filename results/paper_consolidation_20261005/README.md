# 论文收口：S0 与 C 模型筛选里程碑

**先读 [C 审查结论](C/review/REVIEW.md)**：门槛点估计、尾部敏感性及独立指标必须一起解释。原报告的 point_gate 不是最终扩展决定；决定见 [decision.json](C/review/decision.json)。

S0 基线身份、结果复用与计算量审查见 [S0_NOTES](S0/S0_NOTES.md)，主系统草案见 [system_draft](S0/system_draft.md)。

C 的表模型结果见 [报告](C/report/REPORT.md)。这里只发布已完成的 S0 和 C 模型筛选；C 尚无新实际衰落译码或视觉推断，结果不是独立 holdout。

S0 证据来自本地历史下载资产，来源别名和原始 SHA 保留，并不表示这些文件已存在于本仓库。详见 LOCAL_EVIDENCE_SCOPE.json。

公开配置、完成记录和源码副本均已移除机器私密路径：REPO 表示仓库根目录，EXTERNAL_ASSET 表示匿名外部资产；原始文件与反向映射留在实验端。PUBLICATION_MANIFEST 同时记录 raw_sha256 与 export_sha256，公开副本不能冒充原始执行源码或原始凭证字节。

大型科学数组留在实验服务器，原 completion 的完整 SHA 引用以别名保留。没有发布张量、原始载荷、模型权重或运行日志。S1 的登记/执行属于独立里程碑，本目录不声明其完成。
