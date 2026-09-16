# WeTok bit监督权重续训：探索性提前停止

2026-09-12 15:52停止。原计划每臂追加5000，但本次未完成该计划，不能称最终成功/充分收敛比较。

六臂原joint/bit权重1都保留共同追加1000/global6000的完整模型与Adam；实际中断时共同计数1113，未提交更新不进入比较，实际占用约0.7593小时包含这些被排除更新。原Adam父点和物理/网络/原生接口均未变化。

## 同一固定100张calibration监控源

下表五个SNR都采用同一源图与校准噪声。父点/历史点仅截取完全相同的100张，不拿全1000图均值与100图直接作配对差。

| 结构 | 历史2000 LPIPS | 父点5000 LPIPS | 原joint再1000 | bit权重1再1000 |
| --- | ---: | ---: | ---: | ---: |
| single_pass | 0.288028 | 0.418492 | 0.685089 | 0.831983 |
| 无历史多尺度 | 0.286912 | 0.596219 | 0.728105 | 0.906102 |
| 条件多尺度 | 0.287803 | 0.438198 | 0.536525 | 0.760356 |

两配方均继续退化，未观察到增大bit权重遏制该过程。1000步bit权重1的BER约45.7%–48.0%；选择性继续跑一个较好的结构会破坏共同训练机会，因此六臂一同保留现场并停止。

这是观察结果后的探索性止损，不是预注册显著性停止规则。结论仅覆盖这一父点与配方，不能否定所有bit监督权重，也没有证明继续更久绝不恢复。没有为这次停止访问development或新test。

## 已定位与未定位

固定四张训练源的noiseless诊断中，原2000到5000的logits绝对最大值从约15增长到1229–1975；归一化前TX RMS约0.9→1.4–1.7，没有趋零，抽查的context LayerNorm方差也没有塌缩。不能由四张图排除全部病态，但不支持把当前问题直接归因于功率归一化零分母。

image梯度存在并不意味着identity-ST方向适合受损native bits。下一步按v2的接收接口/训练修订分支，固定网络、原loss、视觉和物理预算，对照native identity-ST、native有界均值-ST、训练和评测均连续的特征接口；目前仍是待验证候选。

## 原始产物

- 决策/安全停止回执：`outputs/WETOK-COMM-BIT-SUPPORT-20260912-TRAINING/exploratory_early_stop/termination.json`
- 共同checkpoint SHA256：`20e52bc3205e2d78e2c26fed2dff7b1a01b9a13be6ebde921c9f35ae62d94108`
- 同源监控表：`outputs/WETOK-COMM-BIT-SUPPORT-20260912-ANALYSIS/exploratory_early_stop/same_source_calibration_deterioration.csv`
- 训练窗口/梯度图：`outputs/WETOK-COMM-BIT-SUPPORT-20260912-ANALYSIS/exploratory_early_stop/retained_training_deterioration.png`
- CPU归一化诊断：`outputs/WETOK-MAPPING-CONDITIONING-20260912/summary.json`

停止这一权重续训候选，不停止VAR/多尺度通信研究，不覆盖原2000历史最佳或强数字/Deep对照。
