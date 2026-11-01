# V8 大纲与证据职责

- Abstract：99 词；固定帧源参考感知重建、同模型利用率和感知改善、紧预算结果及已实测成本。
- I Introduction：目标；粗到细完整粗表示/残差组织；整尺度与实际码长不匹配；系统与内部/外部证据。
- II.A Next-Scale Representation and Digital Frame：表示、功率、AWGN、68-symbol header、开销与训练/冻结身份。
- II.B Token-Wise Endpoints and Actual-Length Fitting：独立终止实际码长、最多 token 可行端点、离线策略、N512/N1024 实现差异。
- II.C Prefix-Constrained Reconstruction：覆盖收到的位置，同时更新残差和后续条件；未收边界位置不进行同尺度重预测。
- III.A Experimental Setup：独立校准、来源配对、完整灰图分母、五指标、七系统适配、真实计时边界。
- III.B Effect of Token-Wise Frame Fitting：Table I 的共享模型、匹配粒度、同前缀顺序干预。
- III.C Reconstruction Quality and Processing Cost：Fig.2 四基线优势和 Ada 交叉；E2/90bit 示例；Fig.3 结构保留；Table II 的辅助指标与真实成本。
- III.D Transmission With a Reduced Symbol Budget：Table III；N512 内部一致收益、Ada 高预算优势及独立校准。
- III.E Operation Near the Channel Threshold：固定 10 dB 策略在 8–10 dB 的直接观察，完整分解在补充 S5。
- IV Conclusion：保留和使用源信息、内部像素/感知共同改善、紧预算及处理成本、已选链路可靠工作区间。

Supplement S1 设置、S2 完整主结果、S3 机制控制、S4 E2 等信息预算、S5 E6 门限、S6 同源图与选择协议、S7 E5 统一计时。
