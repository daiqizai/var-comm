本轮五路线实际结果入口

先读 REPORT_zh.txt 与 stage_completion.json。A/B/C/E未通过，D通过FIT/CONFIRM并完成common500回归；NEW200受阻，不能宣称独立新留出完成。
route_summary.csv保留全部候选；per_source_metrics.csv/paired_deltas.csv显式标注FIT、CONFIRM、common500，禁止跨样本混算。原逐帧数据在各fit_*_delivery、confirm_D_delivery、final_common500_delivery/frame_metrics.csv。
configuration_index.json连接冻结配置/身份；actual_compute_and_reuse.json与资源CSV记录真实执行。figures为可编辑SVG、PDF、PNG；原生数组位于PFS，图源manifest含完整hash。
本轮没有自动commit/push；原V6未改写。验证结果见validation，完整待办/阻塞见stage_completion.json。

最终核对补充：8张PNG及8页PDF均实际打开检查，见 validation/artifact_review.json。
容量表的 Ada10bit 类别开销已按实际记录统一口径；N1024 Ada 缺失的容量元数据已补齐。原统计文件不改，见 Ada_capacity_identity_supplement.csv 和 validation/capacity_reporting_annotation.json。
7dB拒收数6→13/1500与该点感知区间跨零已明确写入报告。旧V6论文/主图保持原文件；本包为新的研究交付，common500不等同NEW200。
超过单文件限制的CSV/JSON以无损分片交付，large_file_index.json给出重组规则和原始hash。PFS原件不删除。
