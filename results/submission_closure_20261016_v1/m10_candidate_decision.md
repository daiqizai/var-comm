# m10：可以装入，须修正候选范围；第一批在此停止

原 calibration1000，完整第10尺度(10,0)=680token。新完成1000次TX、前8源独立RX，所有8次token精确往返；同一次TX输出的24,000个旧终点码流与当前H800已闭合旧流逐字节一致。没有额外TX、图像重建、信道或评分。

实际含finish码长：min **3823**，median **6071.0**，P95 **6568.05**，max **6855** bit。P95为线性分位数。0源超过8191，但这不保证别的数据集不会溢出。13bit长度与16bit CRC另占29bit。

| q | 标称码率 | 实际k/n | 算术容量 | 可装 / 1000 |
| --- | --- | --- | --- | --- |
| 2 | 1/2 | 956/1912 | 927 | 0 |
| 2 | 2/3 | 1274/1912 | 1245 | 0 |
| 2 | 3/4 | 1434/1912 | 1405 | 0 |
| 2 | 5/6 | 1593/1912 | 1564 | 0 |
| 4 | 1/2 | 1912/3824 | 1883 | 0 |
| 4 | 2/3 | 2549/3824 | 2520 | 0 |
| 4 | 3/4 | 2868/3824 | 2839 | 0 |
| 4 | 5/6 | 3186/3824 | 3157 | 0 |
| 6 | 1/2 | 2868/5736 | 2839 | 0 |
| 6 | 2/3 | 3824/5736 | 3795 | 0 |
| 6 | 3/4 | 4302/5736 | 4273 | 4 |
| 6 | 5/6 | 4780/5736 | 4751 | 18 |

因此最大4751bit容量可装18源，4273bit可装4源；其余十档未观察到可装。不得以比例低为由忽略，也不以raw8160bit排除熵码m10。只证明这1000源的源码可装性，不证明可靠性、质量提升或任意测试源可装性。

## 缓存与真实调用

恢复了历史m10的1000源码资产，但当前H800与该历史缓存的全部1000个m9流不一致；来源/680token一致不足以证明整数CDF和实际码流一致。本轮因此未复用旧m10科学结果，按授权做了一次当前环境TX遍历。历史缓存和失败/诊断记录保留，未用历史16/1000替代本轮18/1000。

## 仅报告的候选修正

所有旧候选保留。新增最大目标m10必须同时进入整尺度与允许部分尺度两家族：整尺度依次m10,m9…m4；部分尺度依次m10，再按累计token数递减检查全部原24终点。不能因K=0误走整尺度回退。RX仅知道付费的实际终点/profile，不能免费得知TX目标家族。详见m10/candidate_delta_manifest.json及逐源变化CSV。

在这批源码上，只有q6×{3/4,5/6}新增目标改变实际前缀，合计22个source–MCS组合，去重18个来源。两家族×三SNR×三噪声的m10改变子集为396逻辑方法帧；若相同付费profile真正共享，则是198物理帧、396包译码。**这只是改变子集，不是完整补测预算。**旧最大目标在其余来源的回退不变，但旧full1000仅测过筛后的finalists，不能把所有旧MCS当成已完成。

对这两个MCS、两家族、4/10/19dB、原1000×3噪声完整候选网格，保守上界为 **36,000逻辑方法帧 / 72,000包译码**（零复用、每帧头+正文）。在下一批登记前，必须完成原事件的逐帧静态匹配，明确实际profile/波形/接收目录/噪声/输出一致才扣减；当前不虚报精确净新增物理帧。相同m10两个家族应共享实际付费profile，不能为声称复用而借用旧y。新公共ID尚未分配；旧非法ID的接收语义影响仍须核查。

源码TX与tokenizer在cal1000上无新缺口。最多18个独特m10完整端点的干净质量/输出尚未核验；本轮未生成这些图。独立源码资格已完成前8源，但不把它们误当18个可装来源的质量证据。后续接收/恢复数量必须按实际事件登记。选择仍限原cal1000 DINO和既有有限筛选，保留旧赢家为锚；不得用common500/new100质量挑选。

条件任务T1-B、T1-C均NOT_RUN。规则冻结后common500/new100源码核查最多另600TX；若两家族三工作点全变，测试逻辑上界9000+1800分别统计，不与本轮混合。最终实现若改变，只讨论受影响端点计时，不自动重测六方法。

## English result statement

On the original 1,000 calibration sources, fully terminated entropy coding of all 680 tokens produced lengths of 3,823–6,855 bits (median 6,071; 95th percentile 6,568.05). Four sources fit the 4,273-bit capacity and 18 fit the maximum 4,751-bit capacity. All eight independently decoded source streams recovered the exact tokens, and all 24,000 legacy endpoint streams matched the current frozen implementation bit for bit. Full-scale m10 therefore cannot be excluded solely by the earlier interface bound. Symmetric candidate corrections for both whole- and partial-scale strategy families are required; no new wireless quality, reliability, or performance gain is claimed here.
