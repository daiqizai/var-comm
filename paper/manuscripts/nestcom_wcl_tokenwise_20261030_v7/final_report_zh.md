# V7 完整执行结果与交付

主配置为 `original_receiver_plus_D_token`：保留原冻结模型、适配后 D_c 与原 argmax 接收机，仅使用逐 token 实际码长装入。本轮没有新训练，也没有恢复此前停止的支线。

## 结果

- E1：66,000 逻辑位置、五指标，220 均值及 220 直接来源配对区间。七个预算/SNR 点的 NeST−WHOLE 五指标区间均有利于 NeST。N1024/10 dB 源码容量利用率为 77.70%→99.79%。完整系统比较仍保留 Ada 的感知优势和区间跨零结果，不宣称全面领先。
- E2：六个总信息预算、15,000 独立接收逻辑结果，90 均值和 90 配对区间。NeST−WHOLE 的 30 个区间均有利。NeST−Ada DISTS 在 620–1940 bits 有利于 NeST，在 2248/2577 bits 有利于 Ada；DreamSim 在 1940 bits 区间跨零，在高两点有利于 Ada。信息预算消除了无线/MCS因素，没有消除 tokenizer、prior、decoder 等系统差异。
- E3：100 源、共同模型/MCS/ceiling/噪声，Whole/Quarter/Eighth/Token 利用率为 78.13/95.19/97.44/99.85%。Token−Quarter PSNR +0.172 [0.125,0.221] dB，DISTS 与 DreamSim 配对区间也有利。实际端点数和检查数分别记录。
- E4：1,800 个实际前缀干预位置。10 dB B0−A0 与 C0−B0 的 PSNR 约 +0.287/+0.537 dB；最后尺度没有更细传播。不是将 Token−Quarter 增益拆成两项。
- E5：16 组、32 个 TX/RX 阶段全部实际退出0，5120 条计时记录（2048 预热、3072 测量），48 个汇总行。N1024/10 dB NeST TX/RX 248.4/411.6 ms，WHOLE 213.8/369.9 ms，Ada 7795.6/7626.1 ms。32 源、2 核、同 GPU、同原数值路由，含实际编解码和逐帧 ARPC IPC；BPG CPU-only。全部 median/p95、成功条件 RX 与资源记录在 S7/CSV 中。
- E6：30,000 实际帧，17,497 接收成功、12,503 拒收。8 dB 三法全部拒收，9 dB 误帧率为 WHOLE15.75%、NeST14.95%、Ada13.50%，10 dB 零观察拒收。不是 SNR 鲁棒性保证。

## 计算与复用

E2 新接收重建：NeST/WHOLE 195、Ada 8733；精确接收输入复用分别4466、225。新增评分8882原生图像对，6118逻辑位置复用已核身份的结果。E3为891唯一物理输入中的591新事件+300复用，100新RX+197复用。E4新增394 A0/B0重建，另2个资格回放核对原C0。E6不新增生成，只对88个灰图/源图对补两种距离，其余身份复用。

E5已有14组4480条成功计时原样保留；经用户明确续开最多两小时，仅补ARPC/BPG两组640条完整记录；另保留一次未完成的RX预热及其资源费用。所有旧失败、修订、资格、actual_wait 和原deadline留存。错误的`statistics`导入及BPG子进程核数适配分别修复；模型和源码数学不变。计时代码实际扫描完整候选列表，旧`length_checks`字段是所选端点降序位置；新计数表明确区分实际检查数量和该位置。没有重跑成功主实验。

本包受控作业登记GPU reservation合计 21.370266 h；包含独立PHY、BPG及历史失败保守余量的CPU合计 31.614985 h，均在原72 h上限内。BPG虽仅用CPU，原调度为独占保留了GPU预约，账本保留该保守费用并另计CPU。文件构建、仓库检查另有回执；不冒称模型推理资源。

## 稿件与图表

正文保留同模型收益、匹配粒度、接收端有序干预、低预算表现和完整系统感知取舍。Table II 补齐真实计时；S4/S5/S7给完整信息预算、门限和成本。Fig.1保留可编辑PPTX；Fig.2为N1024六方法4/7/10/19 dB的DISTS/DreamSim及48个逐点CI；Fig.3为7 dB两张按优势选择的全视野案例，明确不称随机或代表样本。旧图与同源全方法补图保留。

实际页数：双栏5页、双倍行距单栏14页、Word导出14页、补充材料16页。所有最终页面实际打开检查，哈希见`validation/artifact_review.json`。Word正文、三张表和OMML公式均可编辑；SVG字体可编辑性按来源记录说明，Fig.1文字编辑源为PPTX。

## 交付与 Git

入口：`main.tex`、`main_singlecolumn.tex`、`manuscript_doublecolumn.pdf`、`manuscript_singlecolumn.pdf`、`manuscript_review.docx`、`supplementary.pdf`。数据和代码在本稿`inputs/`、`scripts/`及`results/nestcom_perceptual_followup_20261030_v2/`、`experiments/nestcom-perceptual-followup-20261030/`。大数组、模型、日志仍只在PFS。主张、来源和验收对应`claim_evidence_matrix.md`、`figure_table_sources.json`、`validation/`。

全部科学结果与文档验收已完成；正常远端提交/推送及独立remote SHA以最终publication receipt为准，本文件不预填未来提交SHA。本机不commit，不强推。没有需要另开科学实验才能完成的默认项目；新训练、新200或参数扫描均not-run-optional。

正常远端提交与推送已完成并核验：`cdfdfab8bb78552d25205f1d31700b843627378d`，分支 `main`。仓库完整性、CPU 工程检查和 E5 七项聚合检查均通过。正式提交包含全部科学结果与已实际检查的最终文件；本条发布回执随后以仅元数据提交保存，不改变结果、源码或已渲染文件。
