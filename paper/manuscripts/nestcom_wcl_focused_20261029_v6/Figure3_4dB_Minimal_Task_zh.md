> 历史工作单，保留供追溯。其4dB执行部分现已完成；当前结果和验收见README.md、final_report_zh.md与validation/。以下原文不代表当前未完成状态。

# Fig. 3 的 4 dB 替换判断与最小执行任务

## 本次结论

4 dB 值得作为主图的低负载展示点，前提是取出当前冻结方法的真实重建图。已经核验的总体结果表明：N1024、4 dB 下 NeST 的 LPIPS/DreamSim 为 0.263251/0.183874，Ada 为 0.395197/0.243103，ARPC 为 0.512403/0.500976。Latent JSCC 的 LPIPS 更低（0.218782），DreamSim 更高（0.231678），所以仍有取舍，不能写成全部感知指标领先。

现在不能直接把 V5 的 10 dB 图片换标题为 4 dB，也未找到足以直接排成当前方法对照的 4 dB 提交像素。本次只看到了 V5 的真实 10 dB 图片；没有看到当前方法真实 4 dB 重建，因此不声称 4 dB 的视觉效果已经优于现图。

## 已核对的素材

- V5 Fig.3：N1024/10dB；两例为 Pastries development index95 与 Dog development index21。对应完整 source IDs 在下文。
- 当前两例不属于 common500；不能仅按相同整数 index 连接 common500 结果。
- results/selected_examples_10db_20261020_v1/registration.json 明确 source_selection=Previously user-selected development examples; not random common500，选 first formal repeat r3，不按质量选噪声。
- results/wcl_adatoken_20261027_v1/development_scope.json 明确只执行这两例的 N1024/10dB，seed2001，不能据此声称已做 Ada 4dB。
- paper/figures/kodak24_fixed4_N1024/kodak_N1024_04dB_fixed01_04.png 确实存在，但属于旧 SWIN80K/P1024/VAR_UNCONDITIONAL/BPG。它缺少当前 EC_VAR_WHOLE/Ada/ARPC，不能用于当前 Fig.3。
- 已检查 entropy_main_common500、five_method_timing、wcl_adatoken、wcl_revision、kodak_current、nestcom_evidence、focused_review_v2、arpc_common500_extension 的提交目录，未发现所需当前 4dB 图片组。仅对这些路径作可用性结论，并未断言远端 PFS 缓存没有图片。

## 最小 Codex 任务

在当前真实服务器仓库执行一个新的有限可视化导出任务，保留所有原实验、注册、缓存、图表和统计不动。代码、检查点、冻结策略与 V5 当前结果一致，不训练、不重新校准、不扩大 common500 统计。

### 1. 固定源身份和工作点

- N=1024 complex channel uses，SNR=4 dB。
- Pastries：index95，n07860988/ILSVRC2012_val_00010848_n07860988。
- Dog：index21，n02105641/ILSVRC2012_val_00018762_n02105641。
- 沿用 V5 原始256×256像素及预处理身份；先固定这两例，不重新按 NeST 优势筛选。
- 首选主图列：Original / WHOLE / NeST-Com / Ada fixed-frame / ARPC + LDPC。
- Latent JSCC 仍保留在定量曲线；如果主图目的是展示纹理平滑取舍，可把 ARPC 列换为 Latent JSCC，但补充材料保留全部方法的完整图组。
- 如同时做 4+10 dB，仅以同一源做两行才能直观看 SNR 作用。两源×两SNR共四行更适合补充材料，不宜强塞五页主稿。

### 2. 首先导出已有缓存，缺像素才有限补 RX

先在 PFS 中检索已登记图像缓存，并核对完整 source ID、N、SNR、方法族、检查点、seed/namespace与实际接收记录。不要把 10dB 图、N512 图或 RAW64 WHOLE 图当作当前4dB。

原先五方法计时记录已经真实执行了这两例 4dB 的 NeST、Latent JSCC、Swin200000、BPG 和 ARPC；这些计时记录不含 Ada 或 WHOLE。

精确入口与记录：

- results/five_method_timing_20261020_v1/registration.json
- results/five_method_timing_20261020_v1/delivery/per_call.csv
- 记录键 s4_i0021_r3（counter63）与 s4_i0095_r3（counter285）。
- first formal repeat r3 为预先固定的展示选择。
- 使用原 method-specific noise：NEST/ARPC seed6201；Latent/Swin/BPG seed2001。相同 seed 标签不代表不同波形收到相同噪声实现。

PFS 首查目录为：

/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/five_method_timing_20261020_v1/

根据已验证 10dB 的布局，4dB 接收输入应在各方法的 rx_inputs/ 子目录中，以上述记录键命名。数字 NeST/ARPC/BPG 用 .json，连续 Latent/Swin 用 .npz。先在服务器核对实际存在性和原始记录绑定，不能仅按路径模板假定文件存在。

现有接收端像素恢复实现可参考：

- experiments/selected-examples-10db-20261020/execute.py
- experiments/five-method-timing-20261020/native.py
- experiments/five-method-timing-20261020/library_recovery_v1/adapter.py

必须创建新的 4dB 版本与新注册，不直接改旧脚本中的10dB断言，也不调用旧注册覆盖原结果。新注册先绑定真实接收输入哈希，再由原独立 RX 重建。原图和期望输出哈希不进入 RX；输出后再对照 per_call.csv 的原始 output_sha256。出现未知不一致时保存结果并停止，不反复更换环境直到“看起来相似”。

### 3. WHOLE 与 Ada 缺项如何处理

WHOLE：使用 V5 当前 entropy-coded WHOLE，4dB冻结目标 (7,0)、QPSK rate2/3；不能使用 RAW64_WHOLE_VAR_COMPLETION。优先复用同源完整token/源概率缓存与当前编码器，只为两例形成或导出实际4dB帧并走当前接收端。若确无现成接收记录，准确记为最多2个新的有限可视化链路事件。

Ada：先复用已经为 development95/21 生成的512个独立终止前缀和预测类别，避免重做分类器/tokenizer/源概率前向。现有源容器入口：

/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/wcl_adatoken_20261027_v1/ada_development2_source/sources.json

原始 SHA256：
086cb6378b087de7c6b4e98800eae53b31cd1f53bde216eef9063717d4ffb5a0

使用 results/adatoken_response_20261026_v1/c_predclass_cal1000_dino/frozen_policies.json 的4dB策略（原冻结文件SHA256 68d12e9b19006e8509b3ae5a62132e73044b88ad29ca041f96b436144923ae8c），保留付费预测类别、CFG4、T1、top-k2000、top-p1、FP32/TF32off。原 development_scope 只覆盖10dB；4dB新事件应创建新scope/namespace，预先固定seed2001与公开counter规则，不覆盖旧的10dB事件。若无现成接收记录，最多2个新的有限可视化链路事件和2次RX。

现有 Ada 适配入口：
experiments/wcl-adatoken-20261027/reuse_worker.py
experiments/wcl-adatoken-20261027/link_worker.py

这里只说明可复用的实现，不能直接把旧 stage 名称作为新任务执行；它们绑定旧注册与旧输出。

### 4. 输出与验收

- 每幅 native float32 重建和 original pixel hash。
- method/source/N/SNR/noise/actual endpoint/MCS/status 的完整行映射。
- 对这两例每方法计算 LPIPS/DreamSim；使用 native float32，不用PNG再打分。PSNR可保留在CSV。
- PNG仅统一clip/round到uint8，无锐化、修色、生成重绘。
- 失败图保留，不换seed，不把平均指标写在单张图下面。
- 交付4dB整组PNG/PDF/SVG、逐图指标CSV、source/receive/reconstruction provenance。
- 当前10dB图保留为补充。不要在替换4dB后删去正文关于10dB Ada占优的定量结论。
- 在本地新图真正生成并核验前，V6主稿保持可用的10dB Fig.3，不放空框或假4dB图。

本次审计本身：零新训练、零新推理、零新信道事件、零新打分。

