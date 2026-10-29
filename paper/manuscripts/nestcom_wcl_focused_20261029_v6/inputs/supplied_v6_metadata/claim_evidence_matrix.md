# 主张与证据

## V6 的定位和图像决定

| 主张或决定 | 依据 | 状态 |
|---|---|---|
| ARPC 是渐进压缩相关工作；本文实现其固定帧数字链路适配 | references.tex 中 ARPC 原文；inputs/six_system_configuration.csv 与 configuration_evidence.json | 引言与实验设置明确区分 |
| NeST 的贡献是逐图硬符号预算下可解码前缀、实际长度装入与重建的完整设计 | sections/method.tex；Table I 的共享模型、粒度与接收端证据 | 方法全文与V5逐字节一致，贡献叙述已改 |
| N1024、4 dB 的 NeST/Ada 主要感知距离差异可正面呈现 | figures/figure2_data.csv | 本文新增叙述0.263/0.184与0.395/0.243来自原冻结均值 |
| 4 dB 适合准备为主定性图候选 | 原4 dB统计与低源负载问题定位 | 推荐，不声称已看过当前方法4 dB图组 |
| 现有Fig.3必须保持10 dB身份 | figure3_metrics.csv；inputs/figure3_4db_audit/figure3_4db_availability_audit.json | 已交付图片未改 |
| 两源4 dB五方法输出已有实际执行身份，可按原接收输入恢复 | inputs/figure3_4db_audit/timing_4db_fixed_two_expected.csv | 仅读取现存记录，本次未补推理 |

## 沿用并在正文呈现的既有证据

| 主张 | 证据 | 范围 |
|---|---|---|
| Swin is trained for 200k updates and selected by minimum calibration MSE | inputs/training/swin_selected_checkpoint.json and swin_calibration_history.json | 25 original checkpoint evaluations; 1000 calibration sources, two budgets and six SNRs; no test-set selection |
| Swin late-stage loss is largely stable | inputs/training/original_training_plot_data.csv and swin_training_audit.json | Complete 500-update blocks; last-20k mean calibration MSE improvement 0.2000%; original strict stopping records unchanged |
| Ada sampling is stronger than argmax in the three tested perceptual controls | inputs/e1_paired_deltas.csv | Fixed100 sources; N512 at 4/10 dB and N1024 at 10 dB; six LPIPS/DreamSim intervals above zero for argmax minus sampling |
| Frozen 10-dB policies have an observed digital-link failure transition | inputs/review_audit/snr_mismatch_rejections.csv and snr_mismatch_summary.csv | 100 sources x3 receptions per method/SNR; all fail at8dB, all succeed at10/12dB; no fine threshold localization |
| Representative NeST processing time is reported | inputs/timing_summary.csv | N1024/10dB, TX220.89ms, RX398.82ms; RTX PRO6000, six CPU cores, batch1; 16 development sources x3 repeats |
| ARPC captions and visual payload have measured fixed-frame shares | inputs/review_audit/arpc_common100_actual_payload_summary.csv and corresponding receipt | Existing100-source audit; caption with mode/length averages268.64bit; ratios are equal-source means; not a500-source caption statistic |

## 既有核心证据

| 主张 | 证据 | 样本与范围 |
|---|---|---|
| LPIPS/DreamSim improve at six common500 budget/SNR points | `inputs/internal_common500_paired.csv`, `inputs/paired_deltas.csv` | 500 sources per budget, three outcomes/source; shared models; 12 original paired upper bounds below zero; no 500-source CLIP-I inference for N1024 |
| 13.7% LPIPS / 18.3% DreamSim relative mean reduction at N1024 10 dB | `inputs/internal_common500_paired.csv`, `inputs/common500_summary.csv` | common500; metric-mean percentages, not human preference improvement; original WHOLE means verified in prior evidence |
| Residual correction and later conditioning both reduce perceptual distances | `inputs/b_paired.csv` | fixed 100-source subset, N1024 10 dB; ordered B0-A0/C0-B0; A0 still has generation; final-scale propagation zero |
| Eighth adds LPIPS and PSNR gain over Quarter | `inputs/a_paired.csv`, `inputs/a_efficiency.csv` | fixed 100 sources, 10 dB, fixed maximum endpoint/MCS/model; DreamSim interval includes zero; not equivalence, saturation or optimality; timing blocked |
| Six-system perceptual and pixel-fidelity tradeoffs | `inputs/common500_summary.csv`, `figures/figure2_data.csv`, `tables/table_ii.csv` | N1024 common500; all 72 four-metric means retained; Ada/Latent advantages explicit; no overall SOTA or isolated representation-order causal claim |
| N512 joint-success directions retained | `inputs/reception/joint_success_paired_quality.csv` | 500 retained sources, 1491–1500 actual joint events depending on comparison/SNR; 24 descriptive source-equal conditional means; no new CI; failure-inclusive main estimates retained |
| N512 mechanism gain and complete-system comparison are shown together | `inputs/paired_deltas.csv`, `tables/table_iii.csv` | Table III(A): NeST minus WHOLE; III(B): NeST minus Ada; original means and paired intervals, 12 perceptual metric records |
| Kodak24 endpoint gains under unchanged ImageNet policies | `inputs/internal_kodak_paired.csv`, `tables/supplement_kodak_v4.csv` | Three original N1024 LPIPS rows moved from Table I to Supplement S2.5; 24 sources, all original values preserved |
| New TX timing is incomplete | `inputs/timing/actual_compute_and_reuse.json` | 88 complete source pairs at first repeat only; 176 validated measured calls; source 89 tokenizer identity mismatch; no full-benchmark or speed-order claim |
