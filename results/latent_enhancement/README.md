# m8数字基础＋连续latent增强：公开结果

本目录发布2026-09-18完成的development阶段汇总，不包含checkpoint、数据集像素、latent缓存或完整15MB逐帧原表。

**先读[发布复核补注](PUBLICATION_REVIEW.md)：原计时只能作诊断，新增预算的自适应/全尺度数字对照未完成。公开逐SNR图已修正为100源均值，原错误图保留。**

## 结果

- 原100张development图、5个SNR、3个噪声、41个方法，共61500行内部逐帧结果；
- `method_summary.csv`：方法级PSNR、LPIPS、DINO均值；
- `per_source_snr.csv`：按源图和SNR先平均噪声后的质量表；
- `failure_summary.csv`：按方法/SNR的header、body CRC和未完成候选计数；
- `selected_resource_table.csv`：主要方法的N/E与质量对照；
- `paired_comparisons.json`：源图级配对bootstrap区间；
- `online_timing_summary.csv`、`online_timing_per_call.csv`：原计时记录，含冗余调用，只作诊断。
- `per_snr_summary.csv`：对100源聚合后的逐SNR均值。

## 图表

- `report_assets/quality_resource_selected.png`：质量—信道资源取舍；
- `report_assets/quality_per_snr_selected.png`：修正后的100源逐SNR质量；
- `report_assets/quality_per_snr_original_source0.png`：原误标为总体曲线的源0图，仅供追溯；
- `report_assets/failure_rate_selected.png`：失败率；
- `report_assets/online_timing_selected.png`：原计时诊断图，不能用于系统速度排名。

## 边界

连续增强的512/1024点使用N3572/N4084，不能与N3060方法混成单一排名。DINO只报告不选模，但项目早期数字模式开发曾暴露DINO。1 dB仍受m8基础body CRC失败限制；development结果不是新的holdout，也不是完整训练资产包。

对应协议和结论见仓库`reports/latent_enhancement_development_result_20260918.md`。

## 无GPU复算

```bash
python tools/reproduce_latent_results.py
python tools/reproduce_latent_results.py --figures
```

第二条命令额外需要Matplotlib。复算输出只写`reproduced/latent_enhancement/`，不修改发布数据、不读取像素、不下载权重、不启动训练。
