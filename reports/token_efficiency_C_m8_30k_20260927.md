# C m8：30k 单臂延长校准结果（2026-09-27）

唯一登记臂 H8-V 完成30,000次更新，原调度器正式登记 `extend=false`。selected 保留27,500步，utility为0.02707198905016606；30k退步至0.027094192424882202，完整保留。25k→27.5k改善+0.832532%，27.5k→30k改善−0.082016%，不满足连续两段均 >=0.2% 的原延长规则。规则停止不证明理论收敛。m8没有H8-P。

结果在 `results/token_channel_efficiency_20260923/C_extensions/m8_seed2026092304_until30000`。本次新增22.5k/25k/27.5k/30k四轮60,000条完整校准记录、逐源均值、SNR分项、十三轮历史曲线和两张SVG。初始九轮135,000条经既有索引与CSV SHA引用；独立CPU发布器重审十三轮195,000条完整source/SNR/noise/method键、公式、均值、失败、全历史最优selected和正式决策。末轮15,000条记录含0个header失败、3,055个body CRC失败（1dB 3,000、4dB 55），没有剔除失败。各checkpoint复用数字cache，这些校准重评分不代表新增独立数字PHY传输。

本次原始阶段回执完整：`C_N4084_m8_seed2026092304_until30000` 的原命令、returncode=0和不可变completion snapshot经核验，发布状态为 `REAL_C_EXTENSION_CALIBRATION_VERIFIED`，`complete_stage=true`。snapshot SHA为1930b9df8866c99daa698bc11312c9f938e2b01971f15cc6e942572988f76276。没有使用终端回执缺口恢复分支；此前m6/m7已公开的缺口仍保留。

selected checkpoint SHA为5efb3159bc02904d832eb08fb764dbb5a9d044de60c997f76d2c255011e44e88；terminal30k SHA为ddf072ba375ef1661cc7d1e56990e6adb89368f624f60f3409b1d3e6b3753f1c。20k parent SHA为1cac15785f32521da2bb25dcc542ab322f7295754995b9ed727f62df8a51e3f4，registration SHA为79a2e13d35416c2d7a84960ee70020bfb421bc5e67b589b3c722afd38b666407。发布器核验63项绑定、10份校准tensor SHA及真实20k/30k CPU checkpoint的填充optimizer、模型和恢复状态。训练大tensor仅保留原loader核验范围，本次未重哈希全部训练tensor。

N4084=68付费控制+2992数字+1024连续，source3060bit、body母码6164、原puncture发送5984coded slots。E8168是登记约束，校准CSV没有新增逐帧实测E；训练/校准秒数不是完整在线TX/RX时间。

本结果是校准阶段交付。pure延长、N3060、另两个训练seed、selected真实development评测/实际E/完整在线计时、诊断、四个历史GPU worker及最终全方法配对统计仍待完成。源图bootstrap与训练seed变异分开，新holdout与内容选择器继续暂缓。
