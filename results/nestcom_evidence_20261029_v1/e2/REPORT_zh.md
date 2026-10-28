# E2 实测结果

固定 external_common100_v1 的100源，源内平均3次原信道重复。1200个逻辑结果；新增200个唯一接收端反事实重建，4个原路径资格回放。新增无线帧、训练均为0。

原10000×100来源重采样索引完整复用；当前比较定义见 paired_deltas.csv。非独立盲测，不混用500源均值；所有灰图和分母保留。

| N | SNR | 比较 | 指标 | 均值差 | 95%来源配对区间 |
|---:|---:|---|---|---:|---|
| 512 | 4 | R1-VAR-completed minus R0-coarse-only | psnr_db | +0.026089 | [-0.204079, +0.270891] |
| 512 | 4 | R1-VAR-completed minus R0-coarse-only | lpips_alex | -0.265211 | [-0.280597, -0.250025] |
| 512 | 4 | R1-VAR-completed minus R0-coarse-only | dreamsim | -0.420389 | [-0.448760, -0.392461] |
| 512 | 4 | R1-VAR-completed minus R0-coarse-only | clip_image_cosine | +0.202770 | [+0.183714, +0.221842] |
| 512 | 10 | R1-VAR-completed minus R0-coarse-only | psnr_db | +0.770622 | [+0.562589, +0.985582] |
| 512 | 10 | R1-VAR-completed minus R0-coarse-only | lpips_alex | -0.263846 | [-0.277448, -0.250438] |
| 512 | 10 | R1-VAR-completed minus R0-coarse-only | dreamsim | -0.451895 | [-0.479027, -0.424814] |
| 512 | 10 | R1-VAR-completed minus R0-coarse-only | clip_image_cosine | +0.240277 | [+0.220596, +0.259502] |

所有均值、单源值、逐重复值及原生文件身份分别见 summary.csv、source_means.csv、frame_metrics.csv。区间为逐点95%，未作多重比较校正。
E2限定于同一适配后冻结接收机、相同实际WHOLE前缀的补全作用；并非A0/B0/C0，也不排除decoder适配因素。
