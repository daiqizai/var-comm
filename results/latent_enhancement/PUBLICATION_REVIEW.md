# latent增强development结果：发布复核补注

日期：2026-09-18。此次复核不改训练checkpoint、质量CSV或原始计时记录，只修正公开展示和结论边界。

## 逐SNR图

原绘图脚本按method/SNR取到了第一条源图记录，原`quality_per_snr_selected.png`实际是source0视图，不是100源总体曲线。原图保留为`quality_per_snr_original_source0.png`；公开`quality_per_snr_selected.png`改为从逐源逐SNR表重新聚合的100源均值。质量数值本身没有改动。

## 在线计时

原`online_timing_v1`计时保留为诊断记录，但代码复核发现：纯数字TX统一计算了Fb_TX，Dc渲染路径还计算了D0图，算术路径存在重复VAR/D0工作，噪声生成位于RX计时区间，且端点同步并未完全统一。因此原约116/127/251ms不能用于速度排名或“算术慢两倍”的系统结论。

本轮不重跑训练，也不篡改原计时CSV；后续若需要在线计算结论，应去除冗余调用、固定TX/RX边界、统一同步和数据传输口径后重新测量。

## 强数字对照边界

本轮质量矩阵保留固定raw/arithmetic的m7/m8/m9。新增预算的校准自适应动作和可行全尺度数字方案没有在本轮完成，不能把固定m8对照推广为完整数字VAR上限。原报告中的固定模式质量观察保留，但最终系统定位仍需补完这项对照。

本次没有访问新holdout、没有重新训练或重选checkpoint。
