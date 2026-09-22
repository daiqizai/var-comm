# latent增强后续：统计与记录修复

日期：2026-09-19。原始错误文件保留在`outputs/VAR-LATENT-ENHANCEMENT-20260917/followup/integrity_fix_v1/paired_vs_enhancement_original.json`，修订文件另存，不覆盖历史结果。

## A1 根因

原`digital_policy_development_v1/policy_development.py`在生成配对差时，用`bvals[(source,SNR)]`覆盖了同一源图/SNR的三个噪声记录，实际取到了最后一个noise seed；主表则先平均三个噪声。因此配对文件不满足：

```text
mean(A−B) == mean(A)−mean(B)
```

这不是checkpoint或图像推理混用，而是配对聚合实现错误。修订程序现在严格执行：三个噪声→源图/SNR，再对五个SNR取源图均值，最后做source-image paired bootstrap。所有A/B行都记录同一source、SNR、三噪声覆盖；最大均值恒等式误差为`3.78e-15`。

修订产物：

- `paired_vs_enhancement_original.json`：原错误版本，只读保留；
- `paired_vs_enhancement_repaired.json`：正确配对区间；
- `mean_identity_checks.csv`：逐family/budget/renderer/policy/metric的均值恒等式检查；
- `decoder_adaptation_start_verification.json`：实际Decoder续训起点核验。

## A2 实际Decoder续训起点

实际加载的是阶段B `enhancement1024`选定的**40000步checkpoint**，其SHA为`5efe5716de098ecfb8c1484dd07535868f1685a76557d3b2462f20a5e98793b0`，并恢复该checkpoint中的`enhancement1024` AdamW状态。阶段A Dc来自选定38000步checkpoint；Decoder适配第二臂使用其独立参数副本，学习率`1e-6`，通信臂学习率`2e-4`。

第一份实际适配checkpoint是step500，随后完成到step10000；因此“从原1024 40000步checkpoint继续、追加10000步”是实际执行事实，不只是预登记计划。

## A3 边界修订

- m10仍应表述为“尚未纳入当前数字协议”，不扩展为所有完整尺度方案不可传；
- 原41方法矩阵、阶段A/B历史checkpoint和质量数值不改；
- 后续三种机制的新实验必须使用修订后的配对统计和已核对checkpoint，不复用原错误配对区间；
- 新holdout未访问。
