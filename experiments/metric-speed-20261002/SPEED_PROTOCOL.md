# M2 顺序接收器的有条件提速

## 执行边界

本轮仅减少 M2 逐 token 扫描中的设备到主机同步，不改变收发协议、预算、校准源、development 源、prior、lambda 搜索、Gaussian 工作目标、光栅 Gauss-Seidel 顺序、坐标轮数或 FP32 算子。原 M1/M2 源文件保持封存。新源码位于独立目录，在原 M1 完整发布并且原 GPU worker 退出后才进行真实权重工程基准。

顺序为：冻结所有新 Python/Markdown → 工程资格 → 资格与源码普通提交和推送 → 原 M2 各阶段通过 wrapper 顺序运行。基准前不要求新源码已发布；正式 wrapper 必须有匹配资格 SHA 和全部新源码绑定的 `source_publication.json`，状态 PUSHED、检查 PASS、commit 与 remote_commit 相同。原 M1 的完成和发布也必须是真实、零训练、已检查推送。

本轮不修改执行中的 M1，不写原 M2 的 stats/policy/config，不调用 development 来选择实现。controller 管理原 worker 退出和阶段 handoff；资格/适配器只检查条件，不主动停止其他进程。所有速度结论在实际资格前都只是预期。

## 算法保持与同步减少

缺失六尺度共650个位置。原实现每位置把旧 token、新 argmax 与目标增加值读为 Python 标量。候选把 old/new 保留为 device long，动态查表使用 `index_select`，避免0维 CUDA 索引隐藏的 `.item()`。每位置仍用同序 GEMV 算完整4096词表分数，再更新 error，随后才处理下一位置；没有同时更新或换成 GEMM。

`torch.where(new != old, error - delta, error)`在 token 未变时选择原 error，保留其比特，包括 signed zero。token 用 `copy_`更新。目标增加值和变化标记收集到每尺度末一次性传回 CPU，目标值按原来的 Python float 顺序累加，避免树形归约改变结果。STATIC 的同一先验只搬到设备一次/尺度，数值和作用位置不变。

新增 gather/where 内核也有开销。因此不能只根据减少约1950次标量同步声称固定倍数收益；资格会测包括验证和诊断传输的完整 `infer` 时间。

## 工程资格

入口 `python benchmark.py`，回执为 `outputs/METRIC-SPEED-20261002/m2_speed_qualification.json`。

- 读取原 `common.setup()`、`common.data('calibration')`，校准总体仍为1000，实际资格只用固定前四源。记录源ID、预处理ID、像素/F/T SHA、总体输入绑定和真实模型身份。没有 development 访问。
- PCA、结构均值/方差和 STATIC counts 只以这四源现场拟合作为工程夹具。它们只保存在 speed 自己的 `engineering_statistics.pt`，不用于正式策略或质量结论。
- 固定网格：4源×`g4_c8/g8_c32`×clean/7dB×VAR/STATIC/LIKELIHOOD/UNGUIDED，lambda1；另首源 `g4_c8`、7dB、VAR/STATIC 的lambda0.25及2，共68 case。noisy 使用原测量规则与seed4101，clean使用单次seed0。
- 共用相同原量化器、算子和统计，先完整热算子，并对每case热两种实现。每case两次交错 repeat：先原/候选，再候选/原。每次计时前后 CUDA synchronize，比较放在计时外。
- 每尺度 token 必须严格 `torch.equal`；fhat 严格 `torch.equal`；全部 diagnostics、inference 标签一致。每次推理后所有 VAR KV cache 必须关闭并清空。权重状态SHA前后保持原样，无梯度。
- 全部 case/repeat 的原时间之和除以候选时间之和，必须是有限数且至少1.10，才状态 `QUALIFIED`、选择 `accelerated`。这是预登记整个夹具的吞吐门槛，并不宣称每个case或整个M2流水线都提速10%。逐case时间全部保留，不删除慢case，也不按速度另选网格。
- 数值不一致写 `USE_ORIGINAL`及具体case/字段，选择原实现；吞吐不足也选择原实现。源码/数据变化、程序异常、加载失败等写 `FAILED`并非零退出，必须核查，不能伪装为安全回退。工程资格不是科学质量结果。
- 唯一自动重试例外是原 `common.assets.old.b.ResourceBusy` 类型。它表示原温度/资源边界暂不可用：工程资格保留未完成 `RUNNING`及 `resource_wait`，以exit75退出；wrapper也以exit75退出，由controller等待后重跑。其他类型即使名称或文本相同，也不能进入此重试路径。

回执包含所有 speed Python/Markdown SHA（包括 controller/publisher/docs）、原完整依赖绑定、M1发布绑定、模型身份、数值后端、四源夹具/统计 manifest 与 SHA、全部 case 延迟、严格等价标记和最终选择。资格时收集完整新目录，之后新增/修改源码都会使发布或 wrapper 检查失败。

## 正式 M2 适配

入口 `python wrapper.py --stage calibration|evaluation|actual|timing`。原阶段输出路径、执行顺序、1000校准/100development、策略搜索和实际链路 gate 保持原值。wrapper 仅在本进程中选择候选 `infer`，不会编辑旧模块；`USE_ORIGINAL`完全保留原 `infer`。

适配器扩大原 `common.source_bindings()`，加入完整新源码、资格回执和源码发布回执。原 registration 的 seal 增加 `runtime_profile`，并把资格/发布回执加入原 input_artifacts；既有科学字段由原 registration 原样生成。正式加载的模型身份与数值后端/精度标记必须匹配资格。恢复阶段必须匹配同一 runtime profile，禁止混用另一实现产生的结果。

未来统一指标 replay 继续使用原实现并严格比较封存值，不降低 parity 门槛。如果出现后续数据上的数值差异，应停止核查，不能靠放宽阈值掩盖变化。新提速不触发额外实验或训练。

## CPU 检查

`test_acceleration.py`检查顺序更新、FP32中间值、ties/signed zero和无真值接口；原生 Torch CPU 存在时额外直接比较原形式循环。`test_speed_wrapper.py`检查资格门槛（含NaN/inf）、完整源码绑定、原实现回退、注册/profile、原 M1 条件和 token/latent/诊断/KV 判定。CPU夹具不代替上述真实权重 GPU 资格。
