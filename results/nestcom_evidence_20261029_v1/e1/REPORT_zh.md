# E1 实测结果

固定 external_common100_v1 的100源，源内平均3次原信道重复。2700个逻辑结果；新增285个唯一接收端反事实重建，4个原路径资格回放。新增无线帧、训练均为0。

原10000×100来源重采样索引完整复用；当前比较定义见 paired_deltas.csv。非独立盲测，不混用500源均值；所有灰图和分母保留。

| N | SNR | 比较 | 指标 | 均值差 | 95%来源配对区间 |
|---:|---:|---|---|---:|---|
| 512 | 4 | Ada-argmax minus Ada-sampling | psnr_db | -0.474938 | [-0.800863, -0.151990] |
| 512 | 4 | Ada-argmax minus Ada-sampling | lpips_alex | +0.042317 | [+0.029419, +0.055485] |
| 512 | 4 | Ada-argmax minus Ada-sampling | dreamsim | +0.045017 | [+0.028232, +0.062103] |
| 512 | 4 | Ada-argmax minus Ada-sampling | clip_image_cosine | -0.026093 | [-0.037773, -0.014273] |
| 512 | 4 | NeST minus Ada-argmax | psnr_db | +4.631989 | [+4.182979, +5.092706] |
| 512 | 4 | NeST minus Ada-argmax | lpips_alex | -0.236476 | [-0.256728, -0.216696] |
| 512 | 4 | NeST minus Ada-argmax | dreamsim | -0.146063 | [-0.170916, -0.121794] |
| 512 | 4 | NeST minus Ada-argmax | clip_image_cosine | -0.000252 | [-0.019443, +0.018500] |
| 512 | 4 | NeST minus Ada-sampling | psnr_db | +4.157051 | [+3.772097, +4.557673] |
| 512 | 4 | NeST minus Ada-sampling | lpips_alex | -0.194159 | [-0.209972, -0.178887] |
| 512 | 4 | NeST minus Ada-sampling | dreamsim | -0.101045 | [-0.124808, -0.076892] |
| 512 | 4 | NeST minus Ada-sampling | clip_image_cosine | -0.026345 | [-0.044892, -0.007848] |
| 512 | 10 | Ada-argmax minus Ada-sampling | psnr_db | -0.651108 | [-0.984212, -0.337883] |
| 512 | 10 | Ada-argmax minus Ada-sampling | lpips_alex | +0.029551 | [+0.020070, +0.039439] |
| 512 | 10 | Ada-argmax minus Ada-sampling | dreamsim | +0.034543 | [+0.023164, +0.046077] |
| 512 | 10 | Ada-argmax minus Ada-sampling | clip_image_cosine | -0.023150 | [-0.033683, -0.012772] |
| 512 | 10 | NeST minus Ada-argmax | psnr_db | +4.599936 | [+4.187953, +5.018345] |
| 512 | 10 | NeST minus Ada-argmax | lpips_alex | -0.159792 | [-0.175426, -0.143862] |
| 512 | 10 | NeST minus Ada-argmax | dreamsim | -0.090114 | [-0.112030, -0.068927] |
| 512 | 10 | NeST minus Ada-argmax | clip_image_cosine | +0.010247 | [-0.007150, +0.027979] |
| 512 | 10 | NeST minus Ada-sampling | psnr_db | +3.948828 | [+3.566358, +4.347697] |
| 512 | 10 | NeST minus Ada-sampling | lpips_alex | -0.130241 | [-0.142944, -0.117765] |
| 512 | 10 | NeST minus Ada-sampling | dreamsim | -0.055571 | [-0.075962, -0.036013] |
| 512 | 10 | NeST minus Ada-sampling | clip_image_cosine | -0.012902 | [-0.029997, +0.003860] |
| 1024 | 10 | Ada-argmax minus Ada-sampling | psnr_db | -0.155857 | [-0.369175, +0.057268] |
| 1024 | 10 | Ada-argmax minus Ada-sampling | lpips_alex | +0.003779 | [+0.000669, +0.007072] |
| 1024 | 10 | Ada-argmax minus Ada-sampling | dreamsim | +0.003581 | [+0.001005, +0.006014] |
| 1024 | 10 | Ada-argmax minus Ada-sampling | clip_image_cosine | -0.001306 | [-0.005232, +0.002529] |
| 1024 | 10 | NeST minus Ada-argmax | psnr_db | +0.567523 | [+0.245604, +0.902262] |
| 1024 | 10 | NeST minus Ada-argmax | lpips_alex | -0.000347 | [-0.007197, +0.006192] |
| 1024 | 10 | NeST minus Ada-argmax | dreamsim | +0.017803 | [+0.012858, +0.022892] |
| 1024 | 10 | NeST minus Ada-argmax | clip_image_cosine | -0.012422 | [-0.022166, -0.002478] |
| 1024 | 10 | NeST minus Ada-sampling | psnr_db | +0.411666 | [+0.156894, +0.676734] |
| 1024 | 10 | NeST minus Ada-sampling | lpips_alex | +0.003433 | [-0.002106, +0.008900] |
| 1024 | 10 | NeST minus Ada-sampling | dreamsim | +0.021384 | [+0.016658, +0.026202] |
| 1024 | 10 | NeST minus Ada-sampling | clip_image_cosine | -0.013728 | [-0.023490, -0.003838] |

所有均值、单源值、逐重复值及原生文件身份分别见 summary.csv、source_means.csv、frame_metrics.csv。区间为逐点95%，未作多重比较校正。
E1只改变未收尾部的选择；原sampling主基线不替换，不能排除tokenizer/prior/decoder差异。
