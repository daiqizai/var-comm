# SwinJSCC 与 HiFi-DiffCom 外部对比：已启动

本轮训练一个全新 SwinJSCC SA+RA MSE 模型，覆盖总 N1024/N2048、整数 SNR1–13 dB。使用原 ImageNet20k、校准1k、256×256 图像，只在校准集选模。旧 P512/P1024 的预算截断说明保留。

[执行协议](../experiments/external-comparison-20261004/PROTOCOL.md) · [资产盘点](external_inventory_20261004.md) · [历史统计](../results/external_comparison_20261004/step0_statistics/report.md) · [六组指标参照](../results/external_comparison_20261004/reference_metrics_cpu_full/report.md) · [分类置信度](../results/external_comparison_20261004/history_confidence/report.md)

## 已完成

- 6,600 帧历史结果的源配对统计与分类置信度完成，分类器预测逐项与旧测量一致。
- 100 源图、六组条件共 600 帧的无信道指标参照完成；具体数值及区间见上述报告。
- 固定样例已事前登记；历史资源图保留付费类别、Dc/D0 和缺测点标注。
- CPU 工程检查、真实 GPU Swin 训练/恢复资格、随机 Swin 与真实 ADM 接口预检均通过。修复了严格确定性下通道选择反向的算子兼容问题，首轮失败日志保留；测试通过不代表新模型质量已经完成测量。

## 后台执行

Swin 与 ADM 的工程资格已在 GPU 空闲时完成。后台会先确认旧补评完成并推送，再正式训练 Swin；随后进行选定模型 HiFi 资格、两接收器同波形完整采样和统一指标。

总资源包括掩码排名、功率、CRC、tail 和冗余。N1024 正文 768/头 256；N2048 正文 1664/头 384；E=2N。保留原头码率不高于 1/4 的规则，不宣称该分配已经优化。CRC 失败时两接收器均输出同一灰图，并计入指标。

20k 为首次里程碑，每 2500 步完整校准；按登记平台规则两次降低学习率，最早 80k 停止，上限 240k。未满足最终平台规则则标注 budget_truncated。吞吐、显存与 ETA 由实际 GPU 资格及稳定训练测量更新。

## 待完成

新训练、正式外部质量结果，以及同预算本项目 P/D_U/方法一的整合仍待完成。N2048 无类别数字与方法一使用独立登记的校准；未测点不填补，不用旧付费类别结果替代。两外部接收器完成不等于整个对照计划完成。全程不访问 holdout。

提交时的后台状态快照：`WAITING_FOR_HISTORICAL_PUBLICATION_OR_GPU`。实时状态见服务器 controller/status.json。
