# A1a：32 张旧校准图的 BPG 源码可行性

独立新增入口，原 native-256 BPG 的 500 来源、9000 帧及质量结果保持并复用。
本阶段不运行信道、模型或付费 PHY，不读取 holdout 图像或用 holdout 质量选参数。

固定候选为 256/128/64/32 方形输入，QP 粗扫 0/8/16/24/32/40/48/51。
对任意预定字节容量，在相邻粗扫点实际码长跨越容量时，细扫该区间内全部整数 QP；
不假设码长严格单调，不以最终胜出为停止条件。最多 6656 个分辨率/QP case。
使用真实 libbpg 0.9.8、x265、m8、420、ycbcr、8bit，完整 BPG 文件字节全部计入。
每个完整码流独立解码到对应分辨率，再用固定 Pillow uint8 RGB bicubic 恢复 256。
所有 PSNR 对同一原始 256 RGB，逐图 float64 MSE 后换算；本阶段只报告无噪声源码质量。

原 native-256 相同 QP 码流若有正常且 SHA 完全相符的编码凭证，直接只读复用；
缺少的 QP 与其它分辨率新编码。原计时与 fresh 计时单独标明，不混作统一在线成本。
已有同一新请求的完整 case 可复用；失败、无凭证文件或输入变化会停止，不能静默重试。

先在本机生成材料，deadline 必须由主代理填入已授权值：

```text
python experiments/paper_supplement_20261008/a1_bpg_adaptive/prepare_probe_materials.py --workspace C:/Users/11946/Documents/ChatGPT/comm --out results/paper_supplement_20261008/a1_bpg_adaptive/materials_v1 --deadline-unix AUTHORIZED_DEADLINE
```

`execution.json` 提供两个上传文件、原 Linux 精确 argv、有限 input pins 和最小取回清单。
主代理在远端先确认原解释器有 NumPy/Pillow，绑定源码/请求并监督一次实际进程退出，
CPU affinity 16/17、两线程、nice15、CUDA 空。不得把本机材料或 stdout 当实际完成。

取回 `cases.csv`（码长/质量/编码与解码耗时）、`feasibility.csv`（逐源逐分辨率容量）、
`summary.csv`（32 源的超预算率）与正常凭证后，才根据原 calibration 设计并冻结 A1b。
旧 14 MCS 的 64QAM 仅有 1/2、2/3、3/4、5/6；新增 1/3 的容量 235 字节已由旧
16QAM1/2 容量覆盖源码摸底，但其真实 LDPC/PHY 尚未资格，不能借用不匹配的 BLER。
此入口不冻结 MCS、adaptive-resolution 规则或 DINO 目标版本，也不宣称全局 R/D 最优。

实际 A1a 完成后，可运行下面的只读分析入口。它核验完成凭证、原始 CSV 哈希、
32 源身份、粗扫点、逐图 MSE/PSNR，生成逐源/容量的可行性与自适应源码选择表。
它只冻结源码端的分辨率/QP选择规则；MCS仍须经过真实旧cal链路评测才能冻结。
没有实际正常完成的 A1a 凭证时拒绝运行，不生成预测成绩。

```text
python experiments/paper_supplement_20261008/a1_bpg_adaptive/prepare_a1b_source_rule.py --probe ACTUAL_A1A_DIRECTORY --request EXACT_A1A_REQUEST_JSON --out results/paper_supplement_20261008/a1_bpg_adaptive/a1b_source_rule_v1
```

源码规则在实际编码、独立解码且完整文件符合容量的候选里，按对原256图的MSE选择。
同分时依次选更短完整文件、更高输入分辨率、更小QP。维度包含在付费BPG文件中，
接收端从实际收到的文件解析尺寸并恢复到256；不额外免费传输尺寸或源图真值。
原 native256 设置及其已完成结果仍单独保留。自适应码流发生改变时，旧PHY成绩不能
仅因MCS相同而复用。每SNR最多3个MCS进入新增真实校准，长度不符的BLER不可借用。

## A1b/A1c 有限实际链路入口（准备不表示已运行）

实际 A1a 正常结束且上一步源码规则生成后，用 `prepare_a1b_materials.py` 绑定真实文件。
此准备脚本不会启动任何编码或链路；生成 `execution.json` 给主代理逐阶段执行。
旧 native256 完成文件、原100旧cal来源和500holdout映射、原PHY实现全部固定。

```text
python experiments/paper_supplement_20261008/a1_bpg_adaptive/prepare_a1b_materials.py --workspace C:/Users/11946/Documents/ChatGPT/comm --probe ACTUAL_A1A_DIRECTORY --probe-request EXACT_A1A_REQUEST_JSON --source-rule ACTUAL_SOURCE_RULE_JSON --out results/paper_supplement_20261008/a1_bpg_adaptive/a1b_materials_v1 --deadline-unix EXECUTION_DEADLINE
```

`adaptive_stages.py` 的固定次序：

1. `qualification`：复用原14布局资格，仅新增64QAM1/3的2种满容量pattern无噪声往返；
   另对固定前20旧cal来源做完整自适应BPG容器实际往返和接收恢复比对（有源码不适配时如实记状态）。
2. `screen`：原100旧cal实际包满足相同k/n/q、数值PHY和至少32帧时，复用仅作预筛的投递统计。
   缺项按事前固定32轨迹补真实随机容器包。新增64QAM1/3不能借同容量16QAM的BLER。
   以32源原图/固定灰图与源码质量的预计逐图PSNR均值选每SNR前3个MCS，固定ID打破并列。
   原14统计在profile表增加前取得，且投递率受payload影响；它只作近似预筛，不当成最终图像成绩。
3. `codec_calibration`：前32源直接复用实际A1a；仅补原100源里其余68源的源码候选。
4. `calibration`：固定100×6×3MCS×3噪声=5400计划帧，以实际头/正文/完整BPG解码取得PSNR。
5. `freeze`：按100源各自3噪声均值，再跨源平均，选择每SNR真实cal最高PSNR的候选。
6. `codec_holdout`：冻结后才读取500源内容，以同一源码规则计算可传码流；不据holdout修改规则。
7. `holdout`：固定500×6×3=9000计划帧，保留全失败状态、实际接收字节、浮点重建与逐图PSNR。

这是最低有限增补设计，使用原先已固定的100张BPG calibration子集；不冒充主方法1000源全量校准。
若需要改变校准数量，必须在执行前另定请求与预算，不能在看新500结果后扩大直到获胜。

| 阶段 | 最坏新增实际packet decoder调用 |
|---|---:|
| 资格：2个新布局pattern＋20个完整容器 | 44 |
| 缺少匹配统计时的BLER预筛：15×6×32帧 | 5760 |
| 真实calibration：100×6×3×3帧 | 10800 |
| 冻结500holdout：500×6×3帧 | 18000 |
| 独立总上限 | 34604 |

头拒绝或源码不适配可使实际调用少于上限，不删除计划行。原root预算只读；失败/reserved调用
不能静默重试。单GPU完全不使用；源码/PHY最多8个worker各2个CPU，affinity16–31、nice15。
计时专用窗口不能同时运行此流水线。每阶段持有独占owner锁，正常结束凭真实子进程wait与日志。

接收器只接受实际接收字节；尺寸来自付费BPG格式头。CRC错接受照实际字节解码，源真值仅作为
事后差异诊断，绝不救回接收图像。源码不适配、头拒绝、正文CRC拒绝、parser拒绝、BPG格式/解码
失败分别保留并使用固定RGB0.5输出。每源保存浮点重建供原冻结四指标评价器稍后评分；此CPU流水线
结束并不表示LPIPS/DINO-L/ConvNeXt评分和配对统计已经完成。
