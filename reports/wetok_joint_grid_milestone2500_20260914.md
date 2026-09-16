# Joint网格控制2500：继续按匹配机会推进

2026-09-14 18:29两臂2500更新、全1000图×五SNR校准及CPU审计PASS。原E/R/Adam、配对数据/增强/SNR/标准噪声、能量与完整校准选择均通过检查。

| 同2500更新机会的结构/政策 | 选中step | 校准LPIPS ↓ | 校准PSNR ↑ |
|---|---:|---:|---:|
| Joint普通full-grid state | 2500 | 0.198209 | 21.24565 |
| Joint full-grid innovation | 2500 | 0.194304 | 21.37533 |
| 原Joint single | 2500 | 0.197908 | 21.22102 |
| 原Joint no-history | 0 | 0.200233 | 21.07275 |
| 原Joint multiscale state | 2500 | 0.198812 | 21.12145 |
| R-only full-grid innovation | 2500 | 0.196310 | 21.39516 |

当前校准中Joint innovation较其1000点明显改善，低于同机会R-only；普通full-grid则接近single。不能用此前1000的排序当终局，也不能由当前校准宣布独立验证或next-scale优势。

两臂从实际2500末端模型/Adam共同继续5000，global data 9500→12000，原参数/损失/学习率/输入接口与N/E不变。已有三Joint和六R-only均保留，不重新训练。

新Grid质量评测正在按预先的18方法矩阵实现：只新增两模型推理，复用全部16个冻结参考，物理与数据完全对应时才允许复用；当前时延字段与质量分开，后续单独计时。评测准备不占用当前训练GPU，也不修改已冻结训练源码。

截至此点观察的训练/校准墙钟约0.94909 GPU小时。原始校准与审计位于`outputs/WETOK-JOINT-GRID-CONTROLS-R1-ANALYSIS/calibration_0002500/`；仍无新Grid development结果或新holdout。
