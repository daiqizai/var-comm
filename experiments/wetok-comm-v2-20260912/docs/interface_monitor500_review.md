# 接口训练：500追加步固定监控

2026-09-12 17:09（UTC+8）九臂监控已全部输出，实际trainer仍存活。此处是预定100张calibration子集×1/4/7/13/19dB的均值，不是development、全1000图校准或新holdout结果；没有据此选checkpoint。

## LPIPS概览

| 接口 | single_pass | 无历史多尺度 | 条件多尺度 |
| --- | ---: | ---: | ---: |
| native identity-ST | 0.330198 | 0.331525 | 0.334989 |
| native有界-ST | 0.303840 | 0.301364 | 0.305688 |
| 训练后的continuous_mean | 0.257093 | 0.257975 | 0.257031 |

同一批监控源的原native 2000起点LPIPS约0.287–0.288。两个hard接口虽然PSNR上升，LPIPS仍较起点变差；有界-ST减轻了退化，但尚未消除这一取舍。continuous初始未适配时约0.397–0.404，训练500步后降到约0.257，现已低于这批源的原native起点。

这支持**继续训练和检验连续接收接口**，不代表已胜强数字/Deep系统。不能将此calibration均值直接与另一批development表混排。

条件continuous对无历史continuous的LPIPS均值差仅约-0.000944，对single_pass约-0.000062；当前不据此宣布next-scale具有稳定独立增量。

## 条件接收器逐SNR

| SNR | hard identity LPIPS | hard bounded LPIPS | continuous LPIPS | continuous PSNR |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.425604 | 0.403149 | 0.353321 | 17.89784 |
| 4 | 0.371797 | 0.343050 | 0.286169 | 18.89865 |
| 7 | 0.330687 | 0.297804 | 0.245294 | 19.59552 |
| 13 | 0.282965 | 0.250059 | 0.206487 | 20.32468 |
| 19 | 0.263890 | 0.234378 | 0.193885 | 20.53493 |

continuous条件版的thresholded-logit BER约0.1581，高于其共同原生起点约0.1484；因此此时图像改善不能等同于“恢复了更多可靠bits”。continuous并未把输出作为数字码字交付，BER只作诊断。其平均特征绝对幅值约0.7458，实际Decoder输入也不是偷换后的hard符号。

## 执行决定

保持九臂、loss、学习率、数据/噪声配对及N/E不变，继续当前1000追加步检查点。之后根据完整1000图校准和多点走势决定共同追加到2500/5000；不因为一个监控点就改权重、挑最佳模型或开放新test。

原始汇总与九臂监控SHA：`docs/interface_monitor_0000500_observation.json`。逐源/逐SNR明细位于训练目录各臂`monitor_0000500.csv`。status.json可能在calibration结束后到下一次100步保存之间仍显示上一条CALIBRATING；应结合实际/proc、日志与新断点判断，不能据旧状态标签重启。
