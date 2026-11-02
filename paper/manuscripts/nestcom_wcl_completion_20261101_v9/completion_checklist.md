# 当前定位修订完成清单 — 2026-11-02

- **done** — 当前 main 基准、现有改动与原始输入身份核对。
- **done** — 摘要、引言、方法归属、结果证据角色与结论同步。
- **done** — 主图六方法与补充七方法；所有原始数值、CI 和不利结果保留。
- **done** — Fig.1、Fig.3 原字节不变；图注、Word 原生表与 TeX 同步。
- **done** — 双栏/单栏/Word/补充 5/15/15/27 页真实编译、渲染与 62 页目视审查。
- **done** — 本轮独立验收与来源映射；实际仓库检查由 repository_checks.json 记录。
- **not-run-optional** — 新实验、训练、参数扫描、PHY、指标、统计、计时及暂停支线。

本轮没有受阻的论文交付步骤。后续正文为旧轮次清单；以 validation/current_revision.json 指向的本轮记录为准。

---

## 当前 Fig.3 修订 — 2026-11-02

已采用 C10-016 / C10-017 的 N1024、10 dB 两行五列对照，加入 SwinJSCC。图下 DISTS / DreamSim 均为对应原第一重复的实际指标。双栏/单栏/Word/补充材料页数为 5/15/15/27，62 页均已逐页检查。数据、取舍与来源见 [Fig.3 修订记录](FIG3_SELECTION_20261102.md)。下面保留历史版本记录；当前验证文件以本轮为准。

## 2026-11-02 图版接续

本轮 Fig. 2 横排与全格式交付已完成实际编译和 34 页目视检查（5/15/14 页）。详细变更与来源见 [修订说明](FIG2_1X4_REVISION_20261102.md)。以下保留原 V9 科学交付记录；本轮未重跑实验。

# V9 completion checklist

- **P1: done** — Integer CDF/AC/PHY recipe, 1241 candidate rows, 24 LDPC layouts, model files/recipes/relocatable verification, 6 CPU cases and 2 actual TX/RX vectors. Evidence: `inputs/reproduction/; ../../../../experiments/nestcom-v9-completion-20261101/README.md`.
- **P2: done** — Current N1024 paid-NULL: 33 actual checks, 12 MCS paths, original records/waveforms preserved. Evidence: `../../../../results/nestcom_v9_completion_20261101_v1/execution_receipts/paid_null/`.
- **P3: done** — 216 Token logical frames, all 864 four-arm outputs, five metrics, 24-source paired intervals. Evidence: `inputs/kodak_v9/`.
- **P4: done** — Closed Ada sampling/argmax table; 165 descriptive rows recomputed; fixed median-rank/worst examples. Evidence: `inputs/v9_package/; figures/supp_examples_distribution.pdf`.
- **P5: done** — Matched control100 quality and timing32 TX costs; balanced complete-system evidence and ResARC positioning. Evidence: `tables/table_i.csv; sections/results.tex; references.tex`.
- **P6: done** — 5-page main, 15-page double-spaced single, 14-page native Word, 27-page supplement; 61 pages reviewed. Evidence: `validation/artifact_review.json`.
- **optional: not-run-optional** — Perceptual-Swin training, ResARC wireless baseline, extra data/SNR/parameter searches. Evidence: `Outside this work order`.

No scientific task is blocked. Both required repository checks passed and normal main publication was verified at artifact commit `c4e85c4d984e12b8c6e337f8cf97fd5cf8872b94`; see validation/publication_receipt.json. Private adapted weights remain in PFS with file identities and training recipes; they are not represented as public downloads.
