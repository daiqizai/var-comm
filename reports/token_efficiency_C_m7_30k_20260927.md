# C m7：30k 配对延长校准结果（2026-09-27）

H7-V/H7-P 各完成30,000次更新，原调度器正式登记 `extend=false`，随后接续 m8。V 臂选中30k，P 臂保留27.5k；登记规则停止不证明理论收敛。

| 臂 | selected | selected utility | 25k→27.5k 改善 | 27.5k→30k 改善 |
|---|---:|---:|---:|---:|
| H7-V | 30000 | 0.022501603710340958 | +0.533510% | +0.024635% |
| H7-P | 27500 | 0.022927330860309302 | +0.196339% | −0.626604% |

原规则要求至少一臂连续两段均改善 >=0.2% 才共同延长，两臂均未满足。P 臂30k utility为0.023070994359239316，退步完整保留。每臂末轮15,000条记录含1个header失败、2,992个body CRC失败（1dB 2,979、4dB 13，可重叠），没有剔除失败。

结果在 `results/token_channel_efficiency_20260923/C_extensions/m7_seed2026092304_until30000`。本次新增22.5k/25k/27.5k/30k四轮120,000条完整校准记录、逐源均值、SNR分项、全部历史曲线和两张SVG。初始九轮270,000条经不可变索引与CSV SHA引用；独立CPU发布器重审十三轮390,000条完整source/SNR/noise/method键、公式、均值、失败、全历史最优selected和正式决策。两臂及各checkpoint复用数字cache，这些重评分不代表新增独立数字PHY传输。

registration SHA、两个selected checkpoint SHA与20k parent SHA完整列在 `audit.json`、`selected.json` 和 `training/registration.json`。发布器核验63项绑定、10份校准tensor SHA、真实20k/30k CPU checkpoint的填充optimizer、模型及恢复状态。训练大tensor仅保留原loader核验范围，本次未重哈希全部训练tensor。

N4084=68付费控制+1968数字+2048连续，source1860bit、body母码3764、发送3936coded slots。E8168是登记约束，校准CSV未新增逐帧实测E；训练/校准秒数不是完整在线TX/RX时间。

本次也存在原调度回执缺口：原m7最后训练attempt及外层C_followups收到原热保护停止请求，训练仍实际完成30k全校准并写出terminal checkpoint、selected、completion和MILESTONE_COMPLETE；外层runner在停止分支归档stage前退出。原重启lifecycle依据终端completion及完整校准登记停止决策并接续m8，因此缺少原 `C_N4084_m7_seed2026092304_until30000` stage receipt。全部原launch、内外热标志、时间顺序、终端无失败日志和重启证据封存在 `terminal_evidence`；没有重启m7补跑。

未改动的独立CPU发布器通过显式 `--recover-finalized-boundary` 分支核验正式停止边界并捕获终端资产。发布状态明确为 `REAL_C_EXTENSION_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING`，`complete_stage=false`、`terminal_artifacts_verified=true`、`process_returncode=null`。原退出码没有持久化，不能宣称exit0或原调度回执齐全；没有伪造、写回原stage或改动活动实验源码。

本结果是校准阶段交付。m8/pure延长、N3060、另两个训练seed、selected真实development评测/实际E/完整在线计时、诊断、四个历史GPU worker以及最终全方法配对统计仍待完成。源图bootstrap与训练seed变异分开，新holdout与内容选择器继续暂缓。
