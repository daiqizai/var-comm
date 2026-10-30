# 主张与证据

## V6 实际执行后的图像与范围

| 主张或决定 | 证据 | 范围 |
|---|---|---|
| Fig.3 为 N1024、4 dB 的固定 Pastries95/Dog21，Original/WHOLE/NeST/Ada/ARPC | figure3_metrics.csv 与 results/nestcom_fig3_4db_20261029_v1/delivery/source_receive_reconstruction_map.json | 两张既定 development 源，完整视野，不代表 common500 总体 |
| 十个历史输出已恢复且逐数组哈希完全相等 | delivery/actual_compute_and_reuse.json | 五方法×两源，接收机仅用实际已接收输入，无旧 PHY 重跑 |
| 缺失 WHOLE/Ada 图已补齐 | delivery/new_frame_events.json；登记与实际退出记录 | 新4帧、8包译码、4次接收重建；全部成功，无训练 |
| 两例 NeST 的 LPIPS/DreamSim 都低于 WHOLE | figure3_metrics.csv | 仅显示重复，不能据两图声称总体显著性 |
| Latent 在两例 LPIPS 以及 Pastries DreamSim 优于 NeST | figure3_metrics.csv；Supplement S8 | 不利结果保留，七方法图组完整展示 |
| 原 N1024/10dB 图与指标保持原身份 | supplemental/N1024_10dB_figure3.pdf 与对应 metrics.csv | 未改图、未改数、未重命名为4dB |
| 主实验均值、CI、Fig.1/2与三表保持原证据 | validation/data_identity_check.json | 本次两图工作不更新正式主曲线 |
| 新增7dB属于另行授权的有限校准与两图比较 | results/nestcom_7db_20261030_v1/protocol.json | 不扩common500，不把校准集当正式测试 |

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


## 7 dB正式补点

| 主张 | 证据 | 样本与范围 |
|---|---|---|
| Fig.2新增7 dB总体点 | `inputs/common500_7db_summary.csv`, `figures/figure2_data.csv` | 同一common500，六方法×500源×三次信道；12个新均值与逐点CI，旧36点保留 |
| 7 dB策略在测试前冻结 | `inputs/seven_db/protocol.json` | 100图有限校准，与原三个SNR校准规模不同；未用common500重选 |
| 拒收保留完整分母 | `inputs/seven_db/actual_compute_and_reuse.json`, 正式frame_metrics.csv | 固定灰图计入所有1500次/方法；不以成功独立重建数替代逻辑次数 |
| 直接逐源配对 | `inputs/common500_7db_paired_deltas.csv`, `inputs/seven_db/statistics_identity.json` | 源内平均三次后直接相减；原10000×500 bank，不相减边际CI |
| TableII仍限定原三个SNR | `tables/table_ii.csv` | 未生成7 dB CLIP-I；新PSNR完整结果在S9和CSV |
