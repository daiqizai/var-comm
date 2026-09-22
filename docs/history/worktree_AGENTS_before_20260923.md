# VAR 通信协作规则

## 最新实际状态：阶段B已自动启动，禁止重复开队列（2026-09-17 22:46快照）

- A于17:35到40000安全上限结束，选中38000；`stage_A_v1/completion.json`里的`stage_B_started:false`是当时不可变快照，不修改它来反映后续状态（会破坏B绑定）。B当前真实状态看`stage_B_v1/auto_001/status.json`与`training/status.json`。
- B真实GPU资格和215000帧实际RX缓存均已完成，三臂实际训练每臂约12150步，最新完整校准10000。控制器1605981/trainer1745684是快照，不据旧PID启动新任务。全部失败、原D0、选中Dc和原配方保留。
- 增强512/1024出现相对同Dc后处理的校准收益；不同N尚不作严格系统排名。未做最终development/同资源强数字/Dc对照及完整在线成本，不写成通信效率已成立，不访问新holdout或扩展架构。

## 最新执行：阶段B条件自动接续已接好（2026-09-17 17:01快照）

- 用户明确要求持续监测并自动启动B。独立实现`experiments/var-latent-enhancement-20260917/phase_b/`已完成，条件控制器位于`outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v1/auto_001/status.json`，启动PID1605981仅为快照，不据过期PID重复启动。
- 当前A仍在训练，B为`WAIT_FOR_A`。只在A实际完成且原资格通过后锁定其选中Dc，等待GPU0空闲，自动执行真实GPU自检→实际RX latent缓存→512/1024/receiver-only三臂配对训练。GPU自检/参数更新尚未执行，不把CPU准备写成训练已开始。
- 未改动A的源码/配置/停止规则，SHA核验保持一致。13项CPU测试含三臂真实训练循环的中断/恢复模型、Adam和样本序列逐项一致；20万训练＋1.5万校准真实m8 PHY回放已完成且所有失败保留。
- B loss/预算/功率沿用原登记。有限数字噪声池为每源每SNR两个真实回放，连续噪声每更新独立重采样，明确披露；校准仍用原三个噪声。控制不读新增观测，D0和Dc冻结，RX不读真F/Tx基础/归一化系数。
- 只自动重试资源让卡；资格不通过或实现错误停止留档，不跳过检查、改loss或开新路线。B训练完成也不等于全部科学比较完成，同预算数字/Dc对照、development和完整在线计时仍需交付。详见`reports/latent_enhancement_stage_B_handoff_20260917.md`。

## 当前授权：m8基础层追加连续latent增强（2026-09-17本机登记）

- 用户新授权见`experiments/var-latent-enhancement-20260917/AGENTS.md`及`reports/latent_enhancement_protocol_20260917.md`。仅本新实验允许训练；不重开已停止的固定m7/RGB混合，不改写旧结论。
- 原m8/N3060保持不变，追加512/1024复使用，分别N3572/E7144及N4084/E8168；不对拼接波形整体归一化。
- 先训练独立`post_quant_conv+decoder`副本Dc，确认连续表示的可利用收益后再训练轻量增强E/D；D0、视觉编码器、量化器、VAR均保留冻结。完整强数字与同Dc/仅接收精化对照不得省略。
- HiFi和其自动续跑器按用户优先级暂停，原结果/计划保留。GPU0空闲时使用；遇其他授权任务保存让卡，不干预他人进程或共享环境。
- 以下HiFi/外部定位“禁止新训练”是此前范围，不覆盖本次明确授权。准备、运行与完成必须分清；尚未产生的质量结果不写作完成。

## 最新授权：HiFi耗时排查与20源探索阶段（9月16日登记，2026-09-17继续）

- HiFi不再阻塞研究；原1500计划及全部已完成240帧原位保留，原队列暂停，不改作者采样/权重/噪声/失败规则赶工。
- 新阶段是固定20源×五原SNR×noise2001，共100次/系统；名单为原100序号的等距选择，与质量无关。所有质量对照同20源，所有失败保留，不称原1500已完成或新holdout。
- `experiments/external-baseline-positioning-20260916/performance_branch/`是独立执行诊断，未编辑vendor/原adapter。三校准点无attention checkpoint分支只有约0.04%–0.93%完整RX降幅且显存增加，输出/梯度严格资格未全部通过，未部署到队列。
- 原版自身完整重复也有数值差异；保留AA/AB原数组与失败标记，不事后放宽阈值或把工程差异当算法质量失败。全路径输入梯度未关闭，无训练/共享环境变更。
- 当前唯一GPU质量队列在`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/hifi_stage20_20260916/pipeline_001/status.json`，复用48原帧、用原采样补52帧；原1500收尾等待器已暂停，不重复启动。
- 先读`reports/hifi_runtime_profile_result_20260916.md`与新阶段协议。时间图统一复用20源里已有计时共同覆盖的10源，另报告HiFi全100帧耗时/NFE；不将三记录视图当三次采样，不将单帧外推全体。

## Git远端同步长期约定（2026-09-16）

- 用户要求：每次在已授权任务中创建Git提交后，立即正常推送到对应远端，不再逐次询问是否push。
- 当前发布仓库为`publish/var-comm/`，远端`daiqizai/var-comm`、分支`main`。推送前核对远端变化，不force push、不绕过hooks、不覆盖他人工作。
- 推送后查询远端分支，核实该提交已到达远端；只有本地commit成功不能称为已同步。失败时明确告知原因和未同步状态，保留本地提交。
- 此约定不把每次文件编辑等同于需要立即创建提交，也不授权提交未完成结果、权重、原始数据集、环境或凭据。已授权创建的提交不能仅留在本地。

## 最新授权：外部作者方法实际定位（2026-09-16）

- 用户授权实际运行`semcomm/SwinJSCC`与`wsxtyrdd/diffcom`的公开预训练方法，HiFi须同时保留同ADJSCC、同发送和同观测的无扩散输出。入口`experiments/external-baseline-positioning-20260916/`，计划`reports/external_baseline_positioning_plan_20260916.md`。
- 不重跑三权重/R3/数字大审计，不启动新架构、自己模型训练、CSI题、HJSCC从零训练、自动参数搜索或新最终holdout。已有数字raw/算术、R3和感知Deep强参照不得删除。
- 允许公开权重直连下载、隔离环境和必要兼容/协议适配；历史项目与资产只读。自然预算和作者协议先核实，不能用未训练mask强行N3060，也不能把本地自训Swin当作者预训练。
- 只使用已授权本机GPU0，先检查占用，不影响他人任务或共享环境。先校准冻结设置，再原100 development与配对噪声；DINO不选模。
- 最终交付`reports/external_baseline_positioning_result.md`及实测表/图/成本，不将工程通过或论文数字当实际复现成功。

## 最新收口：结束当前固定m7混合主线，保留结构机制证据（2026-09-16）

- 首先读`reports/hybrid_weight_closure_convergence_20260916.md`。两结构×lambda0.01/0.03/0.1六工作点全部完成；0.01原20000历史合规复用，另四臂各从原暖启动/Adam独立新增20000。六点均按校准选中20000；最终仍有改善，不把封顶当彻底收敛。
- C/0.03在主1/4/7相近LPIPS下PSNR+0.5440dB；机制7/13/19的C/0.1对U/0.03在相近PSNR下LPIPS−0.008817。保留失真—感知结构增量，不说条件读取无效；DINO未全面改善。
- 最低LPIPS混合点主区间仍0.217177，对raw/算术分别差+0.033414/+0.034597，原联合系统目标未过。六点在两个预登记区间均落后校准冻结固定比例选择的三指标期望；后者非部署结果，且4dB有局部PSNR取舍，不能声称全SNR逐点支配。
- 9000真实工作点行、9000冻结参考行、9000期望行、1856配对区间及1500 PHY/9000图像独立复核全部完成PASS，曲线/图/CSV在`outputs/HYBRID-WEIGHT-CLOSURE-20260916/`。旧报告、所有checkpoint/Adam及失败记录保留不改。
- 当前固定m7/1882+1110路线结束，不继续改分配、加大网络或搜索loss，不启动新holdout。这不是停止所有VAR通信，也不是证明所有混合JSCC无效。
- CSI不确定性仅完成三篇已发表原文核查与最小验证草案，见`reports/tx_csi_uncertainty_literature_preparation_20260916.md`；传统鲁棒适配、过期CSI图像传输及channel-blind学习已有先例，具体差异尚不足以定稿。没有新题训练授权/排队，不按下方旧PID恢复任何任务。

## 最新授权：固定混合路线的三权重收尾，不再扩结构（2026-09-16）

- 用户接受原结论，仅授权原两接收器×lambda{0.01,0.03,0.1}；结构、残差域、PHY、m7、N3060/E6120不变。计划`reports/hybrid_weight_closure_protocol_20260916.md`，配置`configs/hybrid_weight_closure.json`。
- 0.01复用已通过初态/Adam/完整实际20000历史/配对数据噪声与校准频率审计；新0.03/0.1四臂都从原共同暖启动独立出发，同旧seed和实际20000新增机会，不从旧0.01终点续训。旧报告/输入只读。
- 本轮统一按完整校准主1/4/7的各自MSE+lambda LPIPS选实际点，不以DINO选模。跨lambda的相近LPIPS/PSNR匹配及帧间选择比例只在校准冻结；没有匹配就报告覆盖不足，不插值伪造已实现模型。
- 只补冻结Deep所缺同口径校准；旧强系统开发结果和结构计时只读复用。固定比例选择是每帧只发一个系统的期望参考，不是双发选优/实际部署测试；驻留和切换成本未测。
- 不扩大lambda、网络或分配，不访问新holdout。若全部工作点无实用优势，结束当前固定m7混合范围，只做三篇已发表工作支撑的发送端CSI不确定性新选题报告；不自动开启该新题训练。

## 最新完成：基图条件读取有像素效应，未达到联合目标（2026-09-16）

- 先读`reports/hybrid_base_conditioning_result_20260916.md`。两臂各新增20000、共享E/D累计30000；在预注册上限停止，最后一段两臂仍改善，不宣称彻底收敛。全部校准、3000正常+1500空间诊断development、960新系统计时及1500 PHY/4500图像独立复核完成。
- 7/13/19条件−同参数控制PSNR+0.36378dB（CI正），LPIPS+0.005325（明确变差），DINO CI跨0。空间条件打乱使PSNR−0.97417、LPIPS+0.040414，确认空间条件被实际使用。保留有限像素机制证据，但未达到预登记LPIPS导向稳定/三指标联合标准；不能仅凭自动no-stable状态笼统说条件信息没用。
- 原1/4/7对raw PSNR+0.80344有区间支持，但LPIPS+0.056129，联合系统目标仍未达到。两臂是通过训练参考资格的新受控点，不是合格系统赢家；旧三通道gain负结果不变。
- 全部训练/队列/复核GPU任务已结束，无自动后续。保留所有结果/模型/Adam/失败；不改资源、loss、不加模块或搜索，不访问新holdout。总体通信方法目标尚未完成。

## 最新授权：实际基图条件连续译码，两臂受控验证（2026-09-15）

- 已实际启动23:05队列2711529/trainer2711561（session97872），先完整初始1k校准，再新更新。8CPU/GPU梯度/90工程输出/20计时通过；回执`reports/hybrid_base_conditioning_launch_20260915.md`。执行代码SHA已锁定，运行中不改源码/配置/协议，不重复启动；图表/读数以实时状态为准。

- 用户接受旧混合负结果，授权仅验证实际RX数字基图作为连续Decoder中间空间条件。计划`reports/hybrid_base_conditioning_protocol_20260915.md`，配置`configs/hybrid_base_conditioning.json`；旧方案、源码和结果只读，不追加原三通道gain。
- 两臂使用同一个5488参数小空间模块，总参数均1648125；公共y_A空间特征＋实际基图或零图，经零初始化投影进入32/64中间层。控制全部共享特征权重仍有y_A梯度，不是挂空的弱参数对照。基础E/D不扩宽/深度。
- 共同使用旧add10000仅作诊断暖启动，复制共享E/D与Adam；新增参数同初始化，训练原MSE+0.01LPIPS。m7、68+1882+1110、E6120、FEC/CRC/失败输出全部不变；不提供TX无误码基图给RX。
- 新增更新最少10000、最多20000，两臂同步；完整1k校准驱动判停，任一臂改善都给两臂同机会。7/13/19机制与原1/4/7系统目标分开，不用DINO选模，不访问新holdout。
- 当前CPU八项检查通过，开始真实GPU工程检查；新输出根`HYBRID-BASE-CONDITIONING-20260915`。这不是已验证收益，不据下方旧PID恢复任何历史队列。

## 最新完成：固定数字基础＋源残差混合实验未达到预设目标（2026-09-15）

- 先读`reports/hybrid_source_correction_result_20260915.md`。三臂各10000更新、两次完整1k校准、原100 development、1440次新系统计时全部完成；另1500 PHY回放和6000图像三指标独立复算PASS。产物`outputs/HYBRID-SOURCE-CORRECTION-20260915/`，最终图在`report_assets_001/`。
- 三臂均无合格校准点；10000只按预注册最小违约规则用于诊断。可靠度分支主PSNR19.6940、LPIPS0.236300；对raw自适应PSNR+0.06735且CI跨0，LPIPS+0.05254。相对同参数SNR gain只有微小PSNR差且LPIPS略差，不能作为新成功机制/强基线。
- 1dB短数字块300/300 CRC失败，完整raw m7有246/300正确接受；但≥7dB短基础全正确后感知差距仍在，不能全部归因于误码。曲线仍小幅改善，不宣称训练已彻底收敛或整个混合架构无效。
- 原训练/队列及报告复核GPU任务已结束，不据下方旧PID重启。保留全部模型/optimizer/失败记录；停止本固定点自动加训、换骨干、改分配和搜索。旧1k holdout不作新独立测试，未访问新holdout；广义VAR通信研究尚未完成。

## 最新授权：一个固定预算的数字基础＋源残差JSCC实验（2026-09-15）

- 实际已启动：19:07正式trainer2436487、受限队列2433287（session17285）；21k缓存完成，8项CPU/真实梯度/90质量/30无缓存计时工程回归通过。入口`reports/hybrid_source_correction_launch_20260915.md`与输出根`HYBRID-SOURCE-CORRECTION-20260915/pipeline_001/status.json`。已锁定执行代码SHA，运行中不修改新源码/配置/协议；不据历史PID重复启动。正式质量结论尚未产生。

- 用户提供新对话并要求理解后推进；新问题与边界见`reports/hybrid_source_correction_protocol_20260915.md`及`configs/hybrid_source_correction.json`。仅激活原官方VAR的m7数字基础＋独立源图残差连续链，68+1882+1110=3060、总E6120。
- 一个小连续E/D，add、同参数snr_gain、reliability_gain三个固定融合臂；原20k训练/1k校准，10000更新/臂的有限配对训练。不得恢复旧WeTok或自动架构/分配搜索。新实现只在VAR_COMM，历史源/模型/结果只读。
- TX残差基于自身无噪声确定性重建，不见RX噪声/CRC；RX只用实际观测与合法可靠度。失败数字候选保留，header失败固定灰图；训练必须包含真实PHY错误。训练缓存不免除在线TX VAR补全时间。
- 已用1000 holdout是形成新假设的历史参考，不再当新独立测试；不访问新holdout。上述新授权覆盖下方“等新明确问题”的暂停，仅限本次固定实验，不等于总体研究成功。

## 最新：本轮强对照及独立验证全部完成（2026-09-15 17:42）

- 首先读`reports/communication_convergence_final_20260915.md`。R3收尾、实际整帧熵编码、1000校准/100 development、1000类各一张新holdout及全部CPU复核均完成；最终数据在`outputs/COMMUNICATION-CONVERGENCE-20260915/`。原GPU和受限finisher均结束，不据下方旧PID重启。
- 最强arithmetic quality与同族BLER策略在全部支持点同动作；holdout主LPIPS相对raw quality差+0.000134，CI跨0。6dB有局部源编码/保护收益，但没有独立的新质量调度增量。不能把development的微小熵编码优势写成独立测试通过，也不说所有学习系统落后。
- 方法/名单/指标在像素访问前冻结，测试后没有重选。保留全部checkpoint、原始数据和失败记录；本轮有限任务完成，但尚未形成有独立机制增量的新方法论文。原raw数字自适应、实际熵编码＋BLER、R3/R2均仅按结果保留为工程候选/参考。
- 不自动再启动训练、新骨干、大网络、复杂机制、模式搜索或新holdout；下一步需用户提出/确认新的明确通信问题。大研究目标未完成，不以报告或工程审计完成冒称目标完成。

## 当前实际阶段：冻结后独立验证运行中（2026-09-15）

- R3质量/自身时间、完整126000条校准及12600条development/CPU审计均完成。报告`reports/communication_development_result_20260915.md`。新熵编码链2016次同端点时间也完成，波形/图像差0；不得重跑七系统benchmark。
- 冻结规则与数据在`outputs/COMMUNICATION-CONVERGENCE-20260915/FINAL_FREEZE_001/`。15:59先写方法/有序名单，再访问holdout作像素查重；最终1000未用源、1个重复被预定规则替换，无模型质量筛选。方法仍由校准选择arithmetic_quality，与普通arithmetic BLER表完全等价；不能调阈值制造新机制。
- 当前GPU矩阵进程2221015，`HOLDOUT_001`（工具session80794）。有且仅有受限收尾进程2230734，`VALIDATION_FINISHER_001`（session78377）：等待真实旧进程结束并核对回执后，只启动四冻结参考、CPU审计与冻结策略统计。不得重复启动、不得运行旧R3 finisher/训练或新架构搜索。
- 以/proc、GPU和completion回执为真，不根据旧status或观察超时重启。冻结文件/模式表/指标/清单不能因holdout结果修改。总研究任务尚未完成；收尾状态只有NUMERICS_COMPLETE仍需最后简明报告与独立验证解释。

## 最新授权：通信问题收敛与必要对照（2026-09-15）

- 用户新授权以两份收敛/计时报告继续：补完既定R3 quality和仅其自身的同端点计时；实现同VAR概率的实际熵编码＋整帧FEC；在校准集上确定可靠性导向与最终图像导向的同候选模式策略；开发完成、协议和清单冻结后再访问未用holdout。
- 执行边界`reports/communication_convergence_execution_20260915.md`。禁止新增骨干、大网络、训练及自动参数搜索；不要求学习模型超过数字VAR，不把校准/开发/holdout混用，不删除旧结果或失败记录。
- R3使用原选定10000 checkpoint，不运行旧finisher/16模型benchmark；原七系统计时只读复用。整帧熵编码不能以旧分组链替代。协议/阈值/主指标/允许取舍在对应评测前登记，普通工程步骤可自主完成。
- 新授权覆盖下方历史“没有R3评测/不访问新holdout”的暂停边界，但仅在明确阶段和冻结条件下开放；不因此恢复旧自主网络扩张任务。

## 当前有限任务已完成：冻结系统同端点计时（2026-09-15）

- 正式`full_001`于13:09完成32图/七方案/五SNR/三次重复共3360行，波形和图像最大差均0，起止权重SHA一致；13:10独立CPU产物/统计复核通过。报告`reports/frozen_system_online_timing_result_20260915.md`，正式图在`outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/figures_full/`。
- 主1/4/7 dB数字adaptive完整TX/RX约10.090/62.284ms；R2约29.388/39.535ms，Deep约2.258/3.023ms。CPU端点、不含空口/排队，不混同旧PHY-only时间，不以此宣称新质量或论文贡献。
- 原1994210进程已结束，无后续GPU队列。本次“开始”授权的有限补测已完成；新增训练/架构/参数搜索仍暂停，R3没有最终质量评测。保留所有checkpoint及三次工程失败；研究总体目标未完成，不自动续跑。

## 最新授权：仅执行冻结系统同端点计时（2026-09-15）

- 用户在CPU准备后明确回复“开始”。可实现并运行现有数字VAR m7/m8/m9/adaptive、R2 residual、感知DeepJSCC及同WeTok数字的统一CPU端点TX/RX计时；协议`reports/frozen_system_online_timing_protocol_20260915.md`，配置`configs/frozen_system_online_timing.json`。
- 固定旧32图、五SNR、seed2001、N3060/E6120和原checkpoint/失败规则，不训练/新架构/参数搜索/新holdout，不运行R3最终质量队列。原暂停仅对此有限计时补测开放，任务完成不自动恢复其他研发。
- GPU0需无其他计算进程，冲突则退出留档、不终止他人任务。所有输出在`outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/`，原报告、源码、checkpoint和失败记录不覆盖。

## 当前收敛审查已完成，暂停仍有效（本机UTC+8，2026-09-15）

- 后续CPU输入核验已完成：`reports/digital_online_timing_readiness_20260915.md`，640条旧数字/2400条旧R2计时记录、10项CPU检查。注意R2旧GPU端点与数字CPU端点不一致；没有新完整计时driver或GPU队列。CPU准备到此结束，不用继续扩写预检代替实际测量；暂停不自动解除。
- 先读`reports/convergence_report_20260915.md`与`outputs/CONVERGENCE-AUDIT-20260915/audit_manifest.json`。窗口截至用户请求时01:28:36，核对4组完成系统比较/15训练版本；R3在窗口后02:00完成训练/校准，没有启动最终传输评测，新GPU队列未运行。
- 暂停新增架构、自动参数搜索、新骨干/大网络/复杂机制；报告完成不自动解除暂停。不再按旧v2自动研发条款另起分支。所有checkpoint/失败/中断结果保留。
- 数字VAR自适应重新作为候选主系统，不要求学习模型必须超过它。R2 residual等为冻结对照；学习方面只保留R3 prediction一个待验证候选资格，不增加训练预算，不把校准优于当最终系统优于。
- 不将有限训练结果泛化为架构无效；旧VAR small/enhanced parallel的失格warmup fallback不入强基线排名。数字完整在线TX/RX计时、独立验证和新颖性仍有明确缺口；整个研究目标未完成。

## 最新用户要求：先收敛审查，暂停新增（2026-09-15，本机UTC+8）

- 用户明确要求暂停新增架构和自动参数搜索，保留全部checkpoint/结果/失败记录。完成收敛报告前，不启动新骨干、新大网络或复杂机制，不默认继续R3后的任何新研究分支。
- 当前任务为过去24小时实际完成实验的只读审计，以及自适应数字VAR与最近next-scale压缩/前缀自适应通信工作的具体差异核查。数字VAR重新列为候选主系统，不把学习模型必须超过它作为唯一目标。
- 以`reports/convergence_scope_20260915.md`为边界：窗口截止01:28:36；现有唯一R3固定上限训练可按原10000结束并留存，但不追加训练或新搜索，也不把未完成评测当结果。新评测/收尾代码尚未排队，不自动启动新GPU队列。
- 不把短期训练结果概括为架构无效，不用不合格回退checkpoint充当强基线。已有报告留存，以新的收敛报告澄清范围，不删除或改写失败历史。

## 用户最新主线反馈边界（本机UTC+8，2026-09-15）

- 用户追问尚未找到主线后，已明确R3只是有限辅助归因，不能替代next-scale通信主方法。按已登记范围完成当前R3训练、质量、独立计时，不自动再衍生通用接收器小变体。随后必须回到单一、可证伪的尺度条件通信问题和系统结构，保留普通迭代与强系统反证，不把普通full-grid重新命名next-scale。
- 具体边界见`experiments/wetok-reencoding-vector-control-r3/docs/mainline_boundary_20260915.md`；不停止通信研究，也不把当前局部结果或完善的评测链当总体研究完成。

## 当前：R3实际训练中（本机UTC+8，2026-09-15）

- R3真实初始化/GPU梯度/4项CPU与断点测试通过，首1000完整校准/审计PASS。00:38从实际1000模型/Adam继续10000，当前trainer1099666/reviewer1099667/observer1099668；00:46核验2000、global9000。入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`，报告`reports/wetok_r3_activation_20260915.md`。下方未实现/未启动状态为历史，不重启。
- 新训练源码/config/protocol与CPU/GPU资格已绑定SHA，不边跑边改。最终23方法质量/16模型计时仅协议配置已登记，驱动/测试/收尾需并行补齐，不能假定已排队。R2全部结果只读，新holdout不访问，研究总目标未完成。

## 最新接续：R2结束，R3向量归因准备（2026-09-14）

- R2的10000训练、46200主质量行、48400图像CPU核验、2400独立计时已全部完成。当前4/8/16主LPIPS显著差于普通full-grid；残差版有增量但仍有强系统差距和额外计算混淆。完整报告`reports/wetok_r2_result_20260914.md`。
- 当前入口改为`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`，只登记同全网格/参数/两次E/训练历史的prediction-vector对照，尚未实现或训练。不能从R2已优化残差权重切换后短训；先核验原7000父点、零融合初始化、Adam和全部数据资格。旧R2/Joint/Grid/R-only的源码与输出只读。
- 原880740/880741/880742/880743及质量/统计/计时子进程均已退出，不据旧status重启。新GPU启动仍需查实时进程和授权。一个空间分层实现失败不停止通信主线，普通全网格迭代也不改名next-scale。

## 当前执行入口更新（2026-09-14 23:12）

- R2四臂10000训练/全校准/模型Adam/数据审计已结束并PASS。single/MS选7500，两个full-grid选10000；当前quality983552、finisher880742、observer880743，原trainer880740/reviewer880741已退出，不重启。入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`，报告`reports/wetok_r2_milestone10000_20260914.md`。
- 新质量与独立计时仍未完整结束；源码/协议绑定不改，自动队列已排好，不开副本。19项CPU测试PASS。后续判断须保留同WeTok数字和其它强系统，区分相同训练机会与实际选中步数，不从校准或部分development结果宣布完成。

## v2持续研究授权（2026-09-12，覆盖下方历史暂停/预算限制）

- 2026-09-14 21:35四臂R2的7500完整校准/模型Adam/数据审计PASS，21:43实际续10000，21:46保存7600总更新。当前trainer880740/reviewer880741，finisher880742已排队质量→CPU分析→独立计时，observer880743；11项CPU/真实driver断点测试通过。入口仍为`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。以下769675/7500队列是历史，不重启；新的训练/收尾源码与协议已绑定SHA，不能边跑边改或篡改哈希放行。

- 当前接续入口为`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`：Grid5000完整结束后，四臂已从真实5000模型/Adam继续至7500阶段；20:26核实总5500即新增500、global12500，未重置optimizer。当前4/8/16未显示独立收益，普通全网格创新量最强但仍有强系统差距。原接口/geometry/RX-only/Joint/Grid已完成源码与输出冻结，不重启旧队列。

- 用户明确要求按`../next_scale_communication_research_brief_v2.md`持续推进。旧金额、半天、固定epoch/候选数量上限不再适用；可在已授权本机GPU上自主研发、训练、消融和校准后续修订，不为正常工程步骤逐项回问。
- 首先推进WeTok A0固定预算通信与A1同骨干逐尺度条件接收，位置`experiments/wetok-comm-v2-20260912/`。复用已经验证的WeTok版本，旧VAR/B/2×2及视觉选型结果只读保留，不重复救当前XQ配对。
- 研发授权不等于新收费/设备/私有数据授权；不租未授权服务，不停他人进程，不改驱动/MPS/时钟。每项主要比较先登记通信问题、因果对照、N/E/CSI/side-information、数据、选模与预计计算量。
- 原生WeTok为32×16×16的±1量化特征、4组8-bit索引；不套4096类CE，不把分组当语义尺度。首轮冻结视觉模型，最终接收接口保持原生hard符号，图像梯度通过冻结Decoder。
- A1中间状态是自定义4/8/16特征金字塔，全部阶段读完整y时只能称多尺度条件接收；不称渐进传输、不提供测试真历史。无合格候选就报告无合格，不能用warmup fallback制造结构增益。

## 最新执行边界（2026-09-12本机UTC+8，覆盖下方历史授权）

- 本轮选型已于01:49完成，09:45独立CPU产物/统计审计通过。结论：**暂不正式迁移到XQ；WeTok仅保留完整codec参照，尚无满足全部条件的next-scale替代**。报告 `reports/backbone_selection_review_2026-09-12.md`。不据候选未通过而停止VAR通信研究，不自动续训/换下一批骨干。
- 用户当前要求一次无训练视觉骨干选型；暂停继续扩大原版VAR训练/模块设计。最新位置授权为**本机既有GPU任务自然结束后**，不连接需认证的开发机，不中断既有2×2流水线。
- 新入口 `experiments/backbone-eval-20260912/`，命名 `ei-liulu-xqvar-eval-20260912-v1`。只比较原官方/已有fidelity、XQ配套MSVR10P2-4096+d17、WeTok指定完整重建，不接FlowAR/FlexVAR、不训练通信。
- 旧100张development/统一评测器/旧源码权重只读。新候选源码/依赖/下载隔离，本机等待监视器检查旧pipeline退出与GPU空闲，不停止他人进程/改变全局依赖。
- 真实XQ两PQ分支必须计费；自然prefix8/9/full静态为2424/3960/6864rawbits，实际索引需复核；不是3060 complex uses实验。
- 完成回执及复核结果以本实验 `docs/` 和 `outputs/ei-liulu-xqvar-eval-20260912-v1/` 为准。未产出结果不能称迁移通过；下文此前训练授权是历史范围，不自动恢复。

## 主线与记录位置

- 2026-09-11固定m8 A/B已全部完成；用户随后明确要求受控升级通信E/D为192维、6头、4层发送+4层共享接收，并补齐小/增强×parallel/next-scale的B配方对照。新入口为`../var-next-scale-comm/scripts/run_token_backbone.py`。
- 本轮允许通信骨干升级及对应parallel训练，不换视觉骨干、不加全尺度/功率分配/FEC/反馈/可变码率。旧B只读固定为工程参考；四个严格比较臂均从头建立一致完整历史（10000共同配方warmup+10000 B），不能只按新增更新声称训练公平。
- 四臂都给定相同名义SNR，并均使用SNR特征调制；这是明示的TX/RX预知SNR假设，不免费提供接收噪声或瞬时CSI。新小骨干含匹配适配，不冒充逐参数不变的旧B。
- 当前项目是 **VAR 用于图像通信**，不是旧 diffusion-JSCC 项目的一个补充章节。
- 当前方向是受信道次数、总能量和合法CSI约束的多尺度通信映射、保护分配与next-scale接收联合训练；全部尺度可参与，不把旧m8/模型规模当永久边界。
- 全尺度/保护分配新方向尚未实现/训练；本次授权仅限已固定m8实验，不授权租卡、下载权重或启动旧工作区的暂停队列。执行入口使用`../var-next-scale-comm/scripts/remote_refinement.py`。
- 开始任务先读 `README.md`、`PROGRESS.md`、`EXPERIMENTS.md`；查相关工作时再读 `LITERATURE.md`。
- 本项目进度更新到本目录的 `PROGRESS.md`；实验更新到 `EXPERIMENTS.md`；报告放在 `reports/`。
- 新代码、配置与输出分别在本目录下按需创建 `src/`、`scripts/`、`configs/`、`outputs/`，
  不再把新结果写回旧 diffusion-JSCC 目录，也不要把新 outputs 软链接到旧输出目录。
- 旧项目自己的“结束时更新旧PROGRESS”规则不决定本项目的记录位置。

## 历史依赖与复现

- `historical_sources.json` 记录归集报告的来源、SHA及历史结果目录。
- 既有代码、模型和数据暂保持原路径，只读复用；本次归集记录不等于运行代码已经迁移。
- 不从这里直接执行旧脚本的默认命令。旧脚本可能按自己的ROOT写回旧目录；
  如需复跑，先检查输出路径、依赖和新目录适配，再明确使用新的输出目录。
- 不覆盖已冻结结果，不把旧报告复制时间当实验执行时间，不改写旧实验状态。
- 未经用户明确要求，不自动提交、push或创建分支。

## 研究边界

- 保留现有 m7/m8/m9 + FEC 自适应系统作为可比较基线。
- 当前基线总预算为3060 complex uses；明确区分raw token bits、FEC和实际信道使用量。
- 新协议另立N/E、header/pilot/反馈账本；旧3060只是锚点。调功率不联动缩放外部噪声；组长不同不把等组能量当等功率。
- 首版无反馈，不给TX免费提供RX状态/噪声/瞬时估计；模型换代须同骨干内部比较并披露DINO等预训练指标暴露。
- 类别、mode、header、校验等side information必须如实说明成本和可靠性，不能偷给新方法额外信息。
- 接收端不得使用原图、未发送token、oracle误码位置或逐图真实指标来选择输出。
- 同时报告失真、感知和语义指标；不以一致性loss下降代替图像质量收益。
- 原ImageNet-100永久作为development。2026-09-04已使用的ImageNetV2 query/gallery不再是全新未见集。
- 停止某个候选，不等于停止VAR通信主线；不得未经用户确认改为旧JSCC/diffusion方向。
- 新方案区分“建议”“已授权执行”“运行中”“完成/停止”，不要把上一轮建议自动当成执行授权。
