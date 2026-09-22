# VAR_COMM — VAR 用于图像通信

**latent增强后续收尾已完成（2026-09-19）：** 数字动作校准、N4084折中分配、修正计时和1024 Decoder配对适配均已完成。1024适配在development上相对同续训控制约+0.1205 dB PSNR、LPIPS−0.00416；折中分配改善低SNR但牺牲中高SNR。完整报告`reports/latent_enhancement_followup_convergence_20260919.md`。

**latent增强development测评已完成（2026-09-18）：** 100图×5 SNR×3噪声、41方法、61500逐帧结果已保存。512/1024相对同资源固定raw m8/Dc在4 dB以上有PSNR/LPIPS增量，但1dB基础失败仍限制感知质量；不访问holdout、不追加训练。原始650次计时仅作诊断，发布复核发现端点有冗余调用，不能用于速度排名。正式报告`reports/latent_enhancement_development_result_20260918.md`。

**阶段B训练已按用户决定收口（2026-09-18）：** 512/1024/receiver-only分别固定30000/40000/35000步选中点，最后安全checkpoint为44510步。近期边际提升较小，未继续到80000；全部checkpoint和结果保留。此为用户主动训练收口，不是充分收敛或系统成功结论。见`reports/latent_enhancement_stage_B_training_closure_20260918.md`。

**B已自动接上并实际训练（2026-09-17 22:46）：** A跑至40000安全上限、选中38000步Dc；B真实GPU资格及215000帧RX缓存完成，三臂每臂约1.2万步。10000步完整校准中512/1024新增观测已有质量改善，但同资源强数字/同Dc、development及完整在线成本仍待完成，不作系统胜出结论。实时入口`outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v1/auto_001/status.json`，详情见`PROGRESS.md`最新条目。

**阶段B自动接续已登记并挂起（2026-09-17 17:01）：** `reports/latent_enhancement_stage_B_handoff_20260917.md`。A继续按原协议运行；独立B控制器等待A结束、锁定合格Dc后，自动进行GPU自检、实际RX缓存和512/1024/无新增传输精化器的配对训练。13项CPU检查及21.5万次真实PHY回放通过；当前是等待接续，B尚未开始GPU训练，不是完整实验完成。

**当前新授权：m8数字基础＋额外连续latent增强（本机2026-09-17，UTC9月16日启动）。** 独立入口`experiments/var-latent-enhancement-20260917/`，协议`reports/latent_enhancement_protocol_20260917.md`。保留原N3060基础，追加512/1024复使用；不重开已停止的m7/RGB混合。真实GPU接口自检通过，FP32训练/校准缓存运行中，随后自动训练独立Decoder副本Dc；尚无新模型质量结果，阶段B尚未开始。HiFi按新优先级暂停，原计划/结果保留。见`reports/latent_enhancement_launch_20260917.md`。

**本次定位材料已推送（2026-09-16 22:12）：** `daiqizai/var-comm`的`main`现为`4be9faa`，远端核对一致。一页定位/差异表/收尾代码及三类取舍图、失败后果表位于公开`results/external_baselines/positioning_interim/`，首页已加入口。HiFi仍是运行中的阶段边界，不冒称最终结果。

**外部结果已同步GitHub（2026-09-16 19:53）：** `daiqizai/var-comm`的`main`已更新至`1eb526f`，远端核对一致。新增外部代码/报告/完整表与120张展示图，入口公开`results/external_baselines/README.md`。纯CPU复算及干净克隆检查通过；HiFi仍运行，只发布阶段快照。详见`reports/github_external_update_20260916.md`。

**外部对照阶段结果（2026-09-16 19:05）：** `reports/external_baseline_non_diffusion_result_20260916.md`。SwinJSCC/ADJSCC、N4204/4498数字等预算补充、完整收发计时及原图/表格已完成。相同N4498下数字VAR的LPIPS更低，但Swin的PSNR更高且更快；不是全面胜出。HiFi保留全部52帧后已恢复原队列，全量比较未完成，最终定位报告仍待它；不启动新架构或训练。

**GitHub已发布（2026-09-16）：** `publish/var-comm/`的500文件、约21.8 MB已推送到`daiqizai/var-comm`的`main`，本地与远端均为`16dd8d7`。使用原有非默认GitHub密钥，之前默认认证检查遗漏已纠正；未新建密钥或改全局配置。代码/报告/关键CSV/图与CPU复算工具已验证，原产物不动。详情`reports/github_publication_status_20260916.md`。发布副本不含大模型、数据像素、venv或凭据，不冒称全量训练资产已迁移。

**最新收敛结论（2026-09-16）：** `reports/hybrid_weight_closure_convergence_20260916.md`。三权重×两结构、完整校准/开发/独立复核全部完成。条件读取在相近感知或像素要求下有真实增量，但最低LPIPS混合点仍比完整raw/算术差0.0334/0.0346，原系统目标未达到。**结束当前固定m7混合主线，保留机制证据与全部模型/结果，不追加搜索或训练。** 固定比例帧间选择仅期望参考，不冒充部署；4dB局部取舍与训练尚有改善均如实披露。CSI新题仅有三篇原文核查和未执行草案，尚缺清楚技术差异，不自动开跑。

**本轮启动快照（2026-09-16 09:44，UTC+8，已结束）：** `reports/hybrid_weight_closure_protocol_20260916.md`。固定m7、N3060/E6120及原两结构，只比较MSE＋lambda LPIPS的0.01/0.03/0.1三档；0.01完整20000步历史通过初态/Adam/数据噪声审计后只读复用，另外四臂各从原共同暖启动独立训练20000步。原队列PID3440348、trainer3440380，不再据此重启。产物状态见`outputs/HYBRID-WEIGHT-CLOSURE-20260916/pipeline_001/status.json`。

**最新完成（2026-09-16）：** `reports/hybrid_base_conditioning_result_20260916.md`。实际基图条件两臂各新增20000更新后按上限收口；校准、开发、空间打乱、计时与独立复核全部完成。7/13/19相对同参数控制PSNR+0.36378dB，但LPIPS+0.005325；空间条件确实被使用，只有有限像素方向增量，没有三指标/系统联合胜出。原1/4/7对raw的LPIPS损失仍+0.056129。所有任务结束，不自动加训/改资源/加模块；下方执行记录是历史快照。

**最新执行：** `reports/hybrid_base_conditioning_protocol_20260915.md`。用户仅授权实际RX基图空间条件的两臂实验：相同小模块/1648125参数、零投影同起点、原loss及m7/N3060/E6120不变。23:05启动正式校准驱动队列，工程回放通过，见`reports/hybrid_base_conditioning_launch_20260915.md`。7/13/19机制与1/4/7系统目标分开，最多各新增20000更新，不访问新holdout。旧三通道gain负结果不改写。

**最新完成：** `reports/hybrid_source_correction_result_20260915.md`。固定m7、68+1882+1110的数字＋源残差三臂训练/校准/开发评测/完整计时及独立复算全部结束。主LPIPS0.236300，显著差于raw0.183763/算术0.182580；无合格校准checkpoint，未达到联合目标。真实分SNR图、固定原图对比与学习曲线在新实验`report_assets_001/`。不把本点失败泛化为混合JSCC无效，不自动追加新架构/分配/训练；下方执行状态为历史快照。

**最新执行：** 用户确认新通信问题后，按`reports/hybrid_source_correction_protocol_20260915.md`推进数字m7基础＋实际源残差JSCC的单一固定分配实验。68 header＋1882数字＋1110连续＝3060，E6120；不搜索m/功率/骨干，不恢复旧队列。新产物`outputs/HYBRID-SOURCE-CORRECTION-20260915/`，运行状态以实际`status.json`/PID/完成回执为准；缓存准备不等于训练或质量成功。上一轮1000 holdout已用于形成此假设，不再是新独立测试。

**最新完成：** `reports/communication_convergence_final_20260915.md`。R3收尾、实际整帧熵编码、校准模式策略及1000张未使用holdout全部完成并审计。最强熵编码quality与BLER策略等价；主LPIPS相对raw差+0.000134、CI跨0，仅局部区间有明确取舍。新调度方法尚不能定稿。全部任务已退出，不再自动开模型/搜索；原数据、checkpoint和失败记录均保留。

当前新收敛阶段：R3冻结质量/仅自身计时已完成（`reports/r3_completion_result_20260915.md`）；整帧VAR算术码＋FEC与三种校准模式规则正在1000校准源上执行，见`reports/whole_entropy_and_mode_policy_protocol_20260915.md`。不训练/新骨干/自动搜索，不用development或holdout逐图真值选择模式。新holdout仍未读取像素；后续先冻结方法与清单再验证。

最新完成（2026-09-15 13:10）：冻结系统同CPU端点计时与独立CPU复核完成，3360条波形/图像逐字节复现、权重不变。主1/4/7 dB数字adaptive TX/RX约10.09/62.28ms，R2为29.39/39.53ms，Deep为2.26/3.02ms；不含空口/排队。完整报告`reports/frozen_system_online_timing_result_20260915.md`，正式图和CSV在`outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/`。无训练/新架构/搜索/R3质量评测，GPU任务已结束，暂停不自动解除。

CPU补充核验：`reports/digital_online_timing_readiness_20260915.md`。640条旧数字记录可复用，但旧R2计时是设备驻留端点，不能直接与数字CPU输出端点横排。新完整计时未实现/运行，暂停仍有效；没有新训练、模型推理或GPU队列。

## 当前：先收敛，暂停新增训练分支/架构/搜索

最新报告`reports/convergence_report_20260915.md`：按本机UTC+8请求时点回看24h，4组完整系统比较、15个模型训练版本全部核对；表格/原始引用/PNG/PDF在`outputs/CONVERGENCE-AUDIT-20260915/`。数字VAR自适应重列候选主系统；保留全部失败记录、不用失格fallback做强基线、不泛化架构无效。

R3只完成了原10000上限的训练和校准（窗口后02:00），没有新的系统推理或收尾队列；当前VAR_COMM无活动GPU训练，不干预其他项目的GPU任务。仅保留一个学习待验证候选，旧R2等冻结。**报告完成不自动解除暂停**；以下训练PID/自动扩展说明均为历史。

## 当前：R3已实际继续到10000目标

本机UTC+8 2026-09-15：真实起点/GPU梯度/4项CPU与断点测试通过，首1000全校准/审计PASS；00:38实际续10000，trainer1099666/reviewer1099667/observer1099668，00:46核验2000。入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`；报告`reports/wetok_r3_activation_20260915.md`。最终评测驱动/收尾尚待实现，只有协议矩阵已登记；原R2已结束只读，下面准备状态均为历史。

## 当前：R2完整结束，准备匹配重编码向量控制

2026-09-14，R2四臂10000/质量/审计/独立计时全部结束，原队列已退出。4/8/16主LPIPS较普通full-grid差+0.002350；重编码残差版LPIPS0.220543，仍落后数字自适应/感知Deep，且需排除额外E和融合贡献。报告`reports/wetok_r2_result_20260914.md`，图表在`outputs/WETOK-JOINT-SUFFICIENCY-R2-ANALYSIS/quality_figures_0010000/`。新入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`，当前仅登记、未训练；旧结果只读，以下活动PID为历史。

2026-09-14 23:09，R2四臂10000训练/全校准/模型Adam与数据审计PASS；single/MS选7500，两个full-grid模型选10000。当前**quality983552、finisher880742、observer880743**，23:12提交23/100源图；原训练/审计进程已正常退出，不重启。入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`，报告`reports/wetok_r2_milestone10000_20260914.md`。质量/独立计时尚未完成，下面“训练中”均为历史。

## 当前：R2四臂从实际7500共同续10000

2026-09-14 21:35全校准/模型Adam/数据审计PASS，四臂校准LPIPS均改善。21:43启动**880740/880741**继续10000，**880742**自动排队质量/CPU统计/独立计时，**880743**被动观察；21:46已保存7600总更新。11项CPU及真实质量/计时driver断点测试通过，不重复训练或评测。入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`；阶段报告`reports/wetok_r2_milestone7500_20260914.md`。未有R2 development或新holdout结论；以下7500/旧队列描述为历史。

## 当前实际接续

2026-09-14 R2四臂已真实启动至7500（769675/785157），从旧5000模型与Adam完整恢复；已验证5500总步数＝新增500，global12500，20:32状态至5800。原配方、预算与旧结果不变，不重跑起点校准。最新入口**`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`**，证据`reports/wetok_r2_activation_20260914.md`。以下准备/旧队列描述为历史。

Grid5000质量、统计、CPU复核与独立计时已全部完成；当前4/8/16未证实独立增益，最优普通全网格学习链仍未解决强系统LPIPS差距。结果`reports/wetok_joint_grid_result_20260914.md`。当前入口改为**`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`**：四臂实际5000模型/Adam已核验，准备同配方充分性续训，尚未启动。以下旧活动PID/队列是历史，不重启。

2026-09-14原Joint5000完整结果已归档：有局部结构LPIPS增量，尚未赢强系统，`reports/wetok_joint_result_20260914.md`。现已实际训练匹配的普通全网格与Joint创新量控制（570837/579794），当前入口**`experiments/wetok-joint-grid-controls-r1/docs/CONTINUE.md`**；真实GPU profile和6项CPU/断点测试通过，已核实300步。以下原Joint等待/评测PID均为历史，不重启。

2026-09-14 16:12接续：Joint5000训练/全校准已完成，但原评测在提交49张后因其它GPU任务触发保护退出。恢复监督器473779及观察器473780正在等待资源，自动原协议`--resume`及统计已设置；不重启训练、不另外开评测。入口`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`与`docs/evaluation_recovery_20260914.md`（后者位于同实验目录）；旧PID状态均保留为历史。

2026-09-14 15:50 Joint5000训练/全校准/审计完成，自动development评测已启动，evaluator443167＋finisher348503；观察器357923记录并发。不要重启已结束的trainer或另开评测。全校准条件版有基础结构增量但仍弱于full-grid强参照，报告`reports/wetok_joint_milestone5000_20260914.md`，曲线`reports/wetok_joint_calibration_figures_20260914.md`。实时入口仍为`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`；以下旧状态为历史。

2026-09-14 Joint2500完整校准/审计PASS，14:37从实际末端继续5000，当前trainer/reviewer/finisher为348501/348502/348503。自动评测与统计已排队，不重复启动。single/state有小幅同机会校准改善，但state仍不及single，no-history仍选起点；报告`reports/wetok_joint_milestone2500_20260914.md`。入口`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`。下方旧PID/队列均为历史；原RX展示图入口`reports/wetok_receiver_figures_20260914.md`。

2026-09-14原RX-only5000已完整评测/分析并冻结；多尺度创新量未胜single/full-grid，不再追加该候选。现激活同起点Joint E/R三结构，trainer/review 188608/188609，先完整起点校准再1000更新。**当前入口：`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`**。结果边界见`reports/wetok_innovation_result_20260914.md`；下方旧队列信息为历史。

2026-09-14已恢复被中断的队列：真实保存点3000，经核验模型/Adam/采样后续到5000，当前新进程25991/25992/25993。先看`experiments/wetok-innovation-r1/docs/recovery_20260914.md`和实时状态；旧9月13日PID/WAIT记录不是仍在运行的证据。

新入口`experiments/wetok-innovation-r1/docs/CONTINUE.md`：2500更新/完整校准/审计完成，六臂正共同继续5000并已排队自动评测/统计。全网格迭代目前优于多尺度创新量，不能称next-scale成立。报告`reports/wetok_innovation_milestone2500_20260913.md`。同起点Joint/R-only后续已准备并通过真实梯度检查，但尚未训练，不抢当前队列。
geometry结果与限制见`reports/wetok_geometry_result_review.md`。以下geometry7000“训练中”条目为历史，不能据此重开旧进程。

## 最新执行入口

本机UTC+8日志2026-09-13：接口5000已完整评测，条件接收出现开发集增量，但未胜强系统；现开始同N/E的153×40对204×30 geometry比较。新三个结构训练中，旧控制合格历史只读复用，不重复训练。
先看`experiments/wetok-comm-v2-20260912/docs/geometry_execution.md`与`outputs/WETOK-GEOMETRY-20260913-TRAINING/status.json`。原接口结果`reports/wetok_interface_milestone5000_2026-09-13.md`；不把新准备/profile当成新方法成功。

## 当前：按v2任务书持续研究（2026-09-12）

最新结果：`reports/wetok_interface_milestone2500_2026-09-12.md`。接口九臂的2500开发评测已完成，连续接口改善但条件结构/强系统优势未成立；当前共同续训至总追加5000。实时状态仍在`outputs/WETOK-COMM-INTERFACE-20260912-TRAINING/status.json`，不要据下方历史准备记录重开旧任务。

最新接续：`experiments/wetok-comm-v2-20260912/docs/interface_execution.md`。原A0/A1首5000已完成但未胜强对照；随后bit权重续训已探索性停止。当前运行九臂接收接口对照，真实状态`outputs/WETOK-COMM-INTERFACE-20260912-TRAINING/status.json`。不将有界连续接口混称native token，也不把新流程启动当作已经获得图像收益。

执行依据为工作区`next_scale_communication_research_brief_v2.md`。旧金额、半天、固定epoch/候选数量限制已撤销；实际机器、收费账户、数据和通信N/E/CSI公平边界仍保留。
已启动`experiments/wetok-comm-v2-20260912/`：WeTok原生±1接口，single-pass／三阶段无历史／三阶段自身历史条件接收，约290万参数/臂。均为无无用类别header的3060 data uses、总能量6120。
20k双视图训练源、1k校准及100图development原生缓存完成，旧100图重建回放像素差0；真实Decoder输入梯度和E/D图像梯度检查通过。首个5000步为研究里程碑，不是总训练上限。
真实进度：`outputs/WETOK-COMM-A0A1-20260912-TRAINING/status.json`；日志与后续评测状态：`outputs/WETOK-COMM-V2-20260912-LOGS/`。根据校准/机制证据持续推进，不把换WeTok或完整codec更好当作通信创新。

## 最新完成：一次无训练配套视觉骨干选型

用户已改为本机执行，等现有2×2完整流水线自然结束再运行，不打断GPU任务。暂停新的原版VAR训练扩展/模块设计。
范围仅旧官方与已有fidelity、XQ-GAN配套MSVR10P2-4096+d17、WeTok8192-bit完整codec；复用100张development和统一四指标，XQ另外测真实prefix+completion。
独立入口 `experiments/backbone-eval-20260912/README.md`；协议 `experiments/backbone-eval-20260912/docs/protocol.md`。
本机2026-09-12 01:49（UTC+8）已全部完成，09:45独立CPU产物/统计复核通过。WeTok同100图full PSNR+0.9034dB、LPIPS改善17.91%，但仅作完整codec参照；XQ的prefix+completion未提供足够正式迁移证据，本轮不换主骨干。详见 `reports/backbone_selection_review_2026-09-12.md`。不追加训练。
下方“当前执行”是此前历史记录，最新授权以上方为准。

从2026-09-06起，这里是用户 **VAR用于通信** 的独立工作入口。
后续研究记录、新方案和实验报告不再挂在旧 diffusion-JSCC 目录下。

## 先看这些

| 文件 | 内容 |
|---|---|
| `PROGRESS.md` | 当前主线、已完成结果、已停止候选与待决定事项 |
| `EXPERIMENTS.md` | VAR通信实验索引，区分开发验证和历史正式测试 |
| `LITERATURE.md` | 仅归集VAR通信相关文献核查记录 |
| `reports/README.md` | 已归集报告的阅读入口与历史路径说明 |
| `historical_sources.json` | 报告原始来源、SHA及原始输出路径 |
| `AGENTS.md` | 后续协作、记录位置和研究边界 |

## 当前研究基础

### 当前执行：B配方下的通信骨干2×2

只升级通信E/D：192维、6头、4层发送+4层共享接收，实际5139662参数；补齐SNR条件匹配小骨干与增强骨干的parallel/next-scale。四臂从头10000 warmup+10000 B，旧B固定工程参考，不改视觉/m8/3060预算。
真实GPU工程检查通过，2026-09-11 14:46（UTC+8）启动正式配对训练，首个共同断点已保存。入口在`../var-next-scale-comm/scripts/run_token_backbone.py`；详细记录`reports/token_backbone_launch_2026-09-11.md`。当前没有新架构性能结论。

### 2026-09-11本机m8 A/B已全部完成

训练、development评测与统计于13:42（UTC+8）完成，GPU已释放。B对同预算A的主区间LPIPS改善27.60%，但未超过固定数字m8/数字自适应/感知Deep的LPIPS。
结果摘要`reports/m8_local_result_2026-09-11.md`；实际选模、配对区间与边界见`PROGRESS.md`。以下准备、授权及暂停条目保留为历史，不自动追加训练。

### 当前具体执行：固定m8 A/B训练（2026-09-11已获本机授权）

用户要求先厘清现有m8通信映射的额外损失，准备A原配方续训/B先prefix-state后图像联合微调，同起点和optimizer、同数据/SNR/噪声、各新增10000更新初始计划。
代码、配置、校准选模、评测和资产转移准备在`../var-next-scale-comm/`；入口文档`docs/REMOTE_M8_HANDOFF.md`。
2026-09-11用户明确授权本机启动：无更新GPU测速/工程检查后执行配对训练和既定development评测；全尺度输入/功率分配不同时加入。实际状态见`PROGRESS.md`；以下历史暂停及总体方向保留，不恢复旧队列。

### 新方向：固定资源下的多尺度条件通信（尚未实现/训练）

用户已将主问题更新为：在明确的带宽、总能量与信道信息约束下，联合训练多尺度通信编码、尺度保护和next-scale接收。
全部尺度的真实信息允许参与编码，不再固定只输入m8；更好的配套骨干是候选基础设施，不是通信贡献本身。
首版设计固定各组长度、学习SNR条件保护，不强制逐尺度CRC、不加反馈。详见`reports/multiscale_joint_direction_2026-09-10.md`。
本轮仅更新方向与设计；原m8两阶段续训仍暂停，不自动租卡、测速、下载模型或开训。以下保留既有基础和历史结果。

- 官方VAR/VQ-VAE的多尺度token表示和接收端补全。
- 固定3060 complex uses下的m7/m8/m9 + FEC自适应系统。
- prefix-only、VAR completion、Full-VQ配对消融及已有跨数据集验证。
- 同前缀量化cell约束修正已经完成开发验证并停止；停止的是这个候选，不是VAR通信主线。
- next-scale先验辅助译码三关已执行：先验和同观测译码/图像机制通过，当前分组原型未胜过强系统对照。

## 当前实验与运行入口

### 已暂停：两阶段prefix恢复与等预算续训

用户要求先暂停并对齐研究目标；当前新增训练0步，没有存活的实验队列。以下为保留的实现与协议，不代表恢复授权。

协议`reports/prefix_refinement_protocol_2026-09-08.md`；配置`configs/prefix_refinement.yaml`。
同epoch2模型/Adam状态、各新增10000更新，修订分支2000步prefix恢复+8000步联合微调；不改m8/3060预算、模型与hard选择。
每1000步完整1k校准，以LPIPS为主并要求各SNR PSNR相对起点退化≤0.2 dB；最终保留固定数字m8及数字自适应强对照。
实现和CPU检查已完成，不等于训练完成。GPU被其他账户服务占用时仅等待，不自动停止其他进程。
自动入口`bash scripts/run_prefix_refinement.sh`；运行状态`outputs/VAR-PREFIX-REFINEMENT-TRAIN-001/status.json`，日志`outputs/logs/prefix_refinement_pipeline.log`。
实际训练结束、开发评测和分析均完成后，报告生成于`reports/prefix_refinement_result.md`，不能提前引用尚未生成的结果。

### 当前：固定m8全局prefix-JSCC训练与评测完成

按用户新授权真正训练通信Encoder/Decoder，VQ/VAR/官方图像decoder冻结；没有无训练性能前置gate。
20k训练、独立1k校准，两版本各2epoch、40k图像曝光，参数均1,508,750，损失MSE+0.01LPIPS+0.001token CE。
68 header+2992 learned data=3060 uses，header真实译码，data不附加FEC；最终60%更新只用自己恢复的历史。
图像梯度、冻结参数、hard-forward一致性和功率自检已经通过。DINO只在最终评测使用。
训练协议：`reports/learned_prefix_training_protocol_2026-09-07.md`；实时训练记录在`outputs/VAR-PREFIX-JSCC-TRAIN-001/`。
两版本均完成10000更新、40k曝光，独立校准选epoch2；10500行统一评测和两项审计完成。
主1/4/7 dB中，next-scale比同参数parallel的LPIPS改善8.921%、PSNR增加0.716 dB、未优化的DINO增加0.06787。
但未胜过原数字自适应；对感知DeepJSCC是DINO更高、PSNR/LPIPS更差的权衡。旧Deep在5/6条件输入的异常另作透明强对照检查，不算新方法优势。
报告：`reports/learned_prefix_result_2026-09-07.md`；结果`outputs/VAR-PREFIX-JSCC-EVAL-001/`。
源码`learned_prefix.py`实现真实image-gradient、硬前向ST和逐尺度自身历史；原大模型/源码未改。当前任务完成，不自动续训或改超参数。

现有校准端点与固定m8逐SNR复核：`reports/prefix_calibration_fixed_m8_review.md`。
next第二轮校准LPIPS全SNR退化；即使数字对照也固定m8，开发集LPIPS/DINO仍在全部七SNR更好。只读分析已完成，未续训或重选checkpoint。

### 最新：整帧有限前缀/末尺度原型已完成

4/5/6/7 dB、100图×3噪声、8臂共9600行及独立审计完成，原整帧发送/FEC/3060账本不变。
VAR没有胜过普通CRC列表65；6 dB接收API约152.84 ms vs 9.90 ms（不含图像生成），完整波形到RGB约211.97 ms vs 69.38 ms。
预定图像达标点仍为7 dB。
4/5 dB的4个假设没有真前缀；6/7 dB覆盖的151个真前缀都被VAR补成正确整帧，瓶颈是候选覆盖。
状态`STOP_THIS_FINITE_PREFIX_VAR_RECEIVER`：停止本版，不扩大列表/训练/搜索，保留原主基线。

报告：`reports/whole_frame_prior_result_2026-09-07.md`；数据`outputs/VAR-WHOLE-FRAME-PRIOR-001/`。
独立审计核查322万候选和全部图像指标；计算上限与CRC误接受限制见报告，未声称胜过强整帧熵编码。
源码`src/var_comm/whole_list.cpp`/`whole_list.py`、`whole_frame.py`；旧冻结源码未改。
仅为复现已完成实验时使用以下新目录（已存在则继续换号），不是建议再次调参：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 scripts/evaluate_whole_frame_prior.py \
  --output-dir outputs/VAR-WHOLE-FRAME-PRIOR-002
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 scripts/audit_whole_frame_prior.py \
  --run-dir outputs/VAR-WHOLE-FRAME-PRIOR-002 --output-dir outputs/VAR-WHOLE-FRAME-PRIOR-AUDIT-002
```

### 前轮：7 dB CRC失败输出回放已完成

仅改变第一个失败硬候选的图像用途；原波形、m9/FEC/3060预算、资源和模型完全不变。
VAR retain追回64.74%的原LPIPS差距，但其二候选oracle仍落后于整帧m9（0.14303 vs 0.13747）。
同样retain后VAR胜过ML与本次熵编码对照；未取得整帧系统优势，不外推到其他SNR。
状态`STOP_TWO_CANDIDATE_SELECTOR_FIXED_M9`：停止这个二候选选择路线，不追加训练/搜索，保留原整帧自适应主基线。

总体/分层完整表：`reports/crc_failure_replay_result_2026-09-07.md`。
原始表在`outputs/VAR-CRC-FAILURE-REPLAY-001/`，独立审计`VAR-CRC-FAILURE-REPLAY-AUDIT-001`通过。
源码`src/var_comm/failure_replay.py`将可信状态与图像状态分离，旧`progressive.py`及前轮产物未改写。
仅在需要复现本项已完成回放时，显式使用新的输出编号（已存在则继续换号）：

```bash
cd /home/liulu/projects/VAR_COMM
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 scripts/replay_crc_failures.py \
  --output-dir outputs/VAR-CRC-FAILURE-REPLAY-002
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 scripts/audit_crc_failure_replay.py \
  --run-dir outputs/VAR-CRC-FAILURE-REPLAY-002 --output-dir outputs/VAR-CRC-FAILURE-REPLAY-AUDIT-002
```

### 前轮：先验辅助译码三关

2026-09-07（本机Asia/Shanghai）连续完成800行先验诊断、7500行单尺度译码、12000行真实前缀/3060-use图像比较及独立审计。
第二关7 dB正确恢复率87.8%→97.8%；第三关同分组LPIPS改善6.79%，但未胜过原整帧自适应及熵编码强对照。
最终状态`MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN`，暂不替换原主基线。

完整结果：`reports/next_scale_decoding_result_2026-09-07.md`。
第一关结果：`reports/next_scale_prior_result_2026-09-07.md`。
对比图：`outputs/VAR-NEXT-SCALE-DECODING-FIGURES-001/fixed_budget_image_quality.png`。

核心源码在`src/var_comm/`：`scale_channel.py`/`token_trellis.cpp`实现整token MAP；
`progressive.py`区分可信前缀与输出前缀；`entropy.py`实现实际算术编码，不调用旧项目runner。

第二/三关结果已冻结；复跑显式选新编号，不能覆盖001：

```bash
cd /home/liulu/projects/VAR_COMM
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 scripts/evaluate_single_scale_channel.py \
  --output-dir outputs/VAR-SINGLE-SCALE-CHANNEL-002
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 scripts/evaluate_progressive_channel.py \
  --output-dir outputs/VAR-PROGRESSIVE-CHANNEL-002
```

审计分别使用`audit_single_scale_channel.py`/`audit_progressive_channel.py`，显式传入新`--run-dir`和`--output-dir`。
两个runner依赖配置中固定的既有通过收据，不自动选择“最新”gate。
本机依赖包括g++、PyTorch/torchvision、NumPy、PyYAML、Pillow、LPIPS和Matplotlib；不自动安装或下载权重。

### 第一关复跑

新增代码在本项目的`src/`、`scripts/`；配置在`configs/next_scale_prior_diagnostic.yaml`。
只读使用配置中固定的旧模型/数据路径，不依赖旧JSCC runner。运行使用已安装的
PyTorch、torchvision、NumPy、PyYAML与Pillow；完整诊断默认使用CUDA。

```bash
cd /home/liulu/projects/VAR_COMM
PYTHONDONTWRITEBYTECODE=1 python3 scripts/diagnose_next_scale_prior.py --self-check-only
```

已有`DIAG-001`/`AUDIT-001`均已冻结。需要复跑时显式选新目录；以下`002`若也已存在，继续换新编号：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/diagnose_next_scale_prior.py \
  --mode full --output-dir outputs/VAR-NEXT-SCALE-PRIOR-DIAG-002
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_next_scale_prior.py \
  --run-dir outputs/VAR-NEXT-SCALE-PRIOR-DIAG-002 \
  --output-dir outputs/VAR-NEXT-SCALE-PRIOR-AUDIT-002
```

审计只读取已缓存概率，不重跑模型；两种runner均禁止覆盖已存在的输出，禁止新输出落到本项目之外。

## 2026-09-06整理的范围（历史）

当时建立了独立记录，并原样归集七份相关报告；那次归集没有迁移运行代码或重跑实验。
旧源码、配置、checkpoint、数据和原始CSV继续保留原路径，避免破坏复现。

- 旧代码和原始结果：`/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc/`
- 共享VAR模型与ImageNet资产：`/home/liulu/projects/VAR-MAP-GATE0/`
- 既有DINO checkpoint来源：`/home/liulu/projects/CAP-VPR/`

以上是历史依赖，不是新的记录位置。不要直接运行旧报告中的默认命令来产生新结果；
旧runner可能将输出写回旧项目。之后新增代码和输出在本目录按需创建并显式设置路径。

此前的“公平性与贡献审计”没有因目录整理被自动启动；当前已完成用户逐尺度方案的三关开发实验与预算/熵编码对照，
不应把它写成所有近作已经完整复现或已通过全新正式测试。
