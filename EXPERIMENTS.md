> 2026-09-23 source A COMPLETE: actual source streams/roundtrips and20 representation paths for1000 calibration+100 original development sources, calibrated quality targets frozen before development. Full source report: reports/token_channel_efficiency_20260923_source_A.md. P2048/P3060 real-device qualification passed; P2048 is training. B1/B2 communication matrices and merged short-prefix C remain incomplete; no new holdout.

> 2026-09-23 supplement accepted: A actual source representation/bitstreams, B1 P2048/P3060 and QPSK resource curves, B2 real16QAM, then merged short-prefix C. New holdout and content selectors are deferred. Code/CPU qualification is not real-result completion. See experiments/token_channel_efficiency_20260923/README.md and outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/status.json. Existing m6 cache continues unchanged; its controller is intentionally held before training to honor A/B priority.

> 2026-09-23 short-prefix study: user-authorized new plan; v1 protocol, five-arm interface,4500 real-PHY preflight and real-GPU qualification implemented. First20k matrix is being launched; new quality/development,N3060,seed repeats,policy and new500 test are NOT_RUN. See reports/short_prefix_20260923_launch.md and experiments/var-short-prefix-hybrid-20260923/README.md.

> 2026-09-23 phase2 COMPLETE: registered bounded training, full calibration, selected-model quality/timing, strict-precision digital requalification, frozen resource lookup and PCA comparison are complete. See reports/review_20260923_phase2.md and results/review_20260923_phase2/. Original A/B and historical artifacts are preserved; no new holdout or convergence claim.

> 2026-09-23 phase1: local review repairs and affected real-weight reevaluation completed; see reports/review_20260923_phase1.md. Phase2 is separately authorized and not included in this completion.

> 2026-09-23 repository migration: the actual VAR_COMM root is the sole Git worktree. Historical research claims below are not newly certified by this migration. No A/B retraining or GPU quality validation was performed. See reports/repository_migration_20260923.md.

# VAR 通信实验索引

## VAR-LATENT-ENHANCEMENT-20260917（新授权，运行中，非旧混合复活）

2026-09-18更新：训练已按用户决定收口后，选定checkpoint的development质量和在线端点计时均完成。结果报告`reports/latent_enhancement_development_result_20260918.md`；质量61500行、41方法、无新holdout，计时650调用。当前证据支持保留latent增强候选，但低SNR和DINO取舍仍需如实披露，外部/R3/Deep只复用既有结果做后续定位，不重新训练本轮模型。

2026-09-18更新：已启动选定checkpoint的development评测，协议`reports/latent_enhancement_evaluation_protocol_20260918.md`，配置`experiments/var-latent-enhancement-20260917/evaluation/config.json`。100图×五SNR×三噪声，raw/算术各N3060/3572/4084、m7/8/9及D0/Dc全部保留；不访问holdout、不重新选checkpoint。当前评测结果尚未完成。

2026-09-17 17:01更新：独立`phase_b/`三臂训练及条件接续已实现，13项CPU测试/21.5万真实PHY回放通过；控制器等待A完成及其所选Dc资格，再自动GPU自检→实际RX缓存→配对训练。报告`reports/latent_enhancement_stage_B_handoff_20260917.md`。当前尚未发生B参数更新；不重复启动旧A/HiFi或把准备冒称全链完成。

入口`experiments/var-latent-enhancement-20260917/`，配置`configs/experiment.json`，协议`reports/latent_enhancement_protocol_20260917.md`，启动`reports/latent_enhancement_launch_20260917.md`。固定原m8/N3060，再追加512/1024连续latent观测；先适配Dc，再根据真实连续参考决定是否进入轻量通信E/D。原D0与数字基础不变，校准选模、不访问新holdout。8项CPU及真实GPU接口自检通过；当前精确缓存→Stage A流水线运行，Stage B/同总资源数字/Dc/refiner/development评测尚未完成。新缓存和模型只写`outputs/VAR-LATENT-ENHANCEMENT-20260917/`，不更改旧结果。

## HIFI-RUNTIME / EXPLORATORY-STAGE20（9月16日登记，2026-09-17执行）

协议`reports/hifi_runtime_stage20_protocol_20260916.md`；性能结果`reports/hifi_runtime_profile_result_20260916.md`；有界AA补查`reports/hifi_repeatability_check_scope_20260916.md`。三固定校准点原版/直接attention分别完整计时和拆分；未改权重/精度/完整采样，未训练，未切换原实现。

原240已提交帧及1500计划只读保留，旧全量收尾等待器暂停。新`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/hifi_stage20_20260916/`固定20源、五SNR、noise2001；复用48旧合法结果、用原版补52，之后同子集所有系统统计。小样本探索不冒称1500完成。原脚本/vendor的六个绑定不变，性能诊断和scope wrapper独立留存。

## 只读推进：不等待HiFi的研究定位

阶段一页报告`reports/research_positioning_one_page_20260916.md`，原文差异`reports/external_related_work_difference_20260916.md`。新增CPU脚本`experiments/external-baseline-positioning-20260916/scripts/position_completed_results.py`只复用已完成输出，三类图及失败后果分层在外部输出根的`research_positioning_interim_001/`。没有重跑历史测量或开新支线。

`finish_hifi_positioning.py --wait`是CPU收尾等待器，不占GPU、不改原推理。完整回执后才出`external_baseline_positioning_result.md`和`research_positioning_one_page_final.md`；出错留存/停止，不自动新训练或Git推送。真实同数据观测与同预算系统对照分开，独立ADJSCC不重复计数。

## 已完成阶段：外部非扩散质量、共同预算数字与计算补充

`reports/external_baseline_non_diffusion_result_20260916.md`；主要产物`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/non_diffusion_analysis_001/`。Swin/ADJSCC公开模型完成；N4204/4498仅原数字编码族和m7/m8/m9进行18000校准＋6000冻结开发；1920新完整CPU端点计时逐字节对齐质量。119表行、1624源图级配对指标区间和固定图片原尺寸/等预算拼图就绪。

执行登记`reports/external_baseline_priority_work_20260916.md`，新入口`evaluate_digital_budgets.py`、`time_digital_budgets.py`、`summarize_available.py`和`audit_non_diffusion.py`均在外部定位实验`scripts/`内。9项CPU测试、6000新PHY和183图像/指标子集复核通过。HiFi原队列19:05恢复，尚未完成全量；没有新训练或holdout，也没有重新打开已结束支线。

## 运行中：EXTERNAL-BASELINE-POSITIONING-20260916

作者SwinJSCC、ADJSCC及同ADJSCC的HiFi-DiffCom，只使用公开预训练权重。五权重核验、作者commit锁定、独立兼容环境、小样本完整HiFi推理与Swin/ADJSCC收发一致性通过；内部四系统6000参考只补SSIM。入口`experiments/external-baseline-positioning-20260916/CONTINUE.md`。

作者测量队列17:34启动；程序实际保留原100 development×五SNR×三噪声。默认HiFi单张235步约78.7秒，长任务不等于训练。原假设/实际计费分表，预算不强塞3060，生成seed固定23。还需新公共预算数字校准/评测、成本与最终报告；此时没有外部胜负结论。已完成/停止的旧研究不重跑。

## 已完成：HYBRID-WEIGHT-CLOSURE-20260916

六个实际工作点均获得同源20000新增训练机会；四新臂实际训练、0.01两臂经严格审计复用。完整校准五节点/固定子集曲线、校准冻结选点及比例、9000开发工作点、9000参考与9000期望行、1856配对区间和完整PHY/图像独立复核均完成。新训练校准会话3.06138 GPU小时；不访问新holdout、不重跑无关计时。

结论见`reports/hybrid_weight_closure_convergence_20260916.md`：有相近感知/像素条件下的局部结构增量，但最低混合LPIPS仍较算术自适应差0.034597，原系统目标未过。结束当前固定m7混合主线，不扩大loss/网络/分配搜索；保留全部数据与模型。帧间选择仅期望参照，不是新的部署系统；4dB局部取舍与训练预算边界如实保留。CSI新题只是文献报告，未启动实验。

## 启动快照：HYBRID-WEIGHT-CLOSURE-20260916（已结束）

原两接收器×lambda{0.01,0.03,0.1}，仅改变MSE＋lambda LPIPS权重；每点从原共同暖启动/Adam获得20000新增更新机会。0.01原历史通过审计后只读复用，0.03/0.1四臂实际新训。m7/68+1882+1110/E6120、五SNR、实际RX基图和失败规则不变。完整校准在0/5k/10k/15k/20k、固定子集其余每1k；DINO只报告。校准冻结实测checkpoint、跨lambda匹配及固定帧间选择比例后才评测原100 development及三噪声。

协议`reports/hybrid_weight_closure_protocol_20260916.md`，配置`configs/hybrid_weight_closure.json`；09:44队列3440348/trainer3440380，四臂各700/20000。缺失的冻结Deep校准15000行已补齐，旧强系统结果/计时不重复。最终只判断结构增量、系统取舍和当前固定路线去留；期望帧间选择不是实际部署，也不访问新holdout。任何新题仅允许文献报告，尚未启动。

## 已完成：HYBRID-BASE-CONDITIONING-20260915（2026-09-16收口）

两臂同1648125参数、同功能/Adam起点，原loss及m7/68+1882+1110/E6120不变。每臂新增20000更新（共享参数累计30000），0/5k/10k/15k/20k完整1k校准和固定子集曲线完成；按上限结束，不宣称收敛。3000正常开发输出、1500空间打乱诊断、9000冻结参考、960新系统计时及完整PHY/图像复核均结束。

报告`reports/hybrid_base_conditioning_result_20260916.md`。7/13/19条件对控制PSNR+0.36378有支持，但LPIPS+0.005325，DINO未确认改善；空间打乱显著损伤输出，保留有限像素侧机制证据。原1/4/7对raw PSNR+0.80344，但LPIPS+0.056129，联合目标未过。原负结果不改写，不自动续训/改资源/加模块，不访问新holdout。

## 当前：HYBRID-BASE-CONDITIONING-20260915

仅训练unconditioned/conditioned两臂，实际基图或零图通过同一个小空间模块，在连续Decoder的32/64恢复层注入；公共y派生特征保证控制的共享模块也可学习。同参数1648125，零投影同初态，旧add10000/Adam仅诊断暖启动。E/D骨干、MSE+0.01LPIPS、m7、N3060/E6120、原FEC不变。

计划`reports/hybrid_base_conditioning_protocol_20260915.md`。CPU/GPU/质量/计时工程检查通过，23:05正式受限队列启动；详见`reports/hybrid_base_conditioning_launch_20260915.md`。新更新10000–20000由固定完整校准规则有界判停，两臂同预算。机制与系统判断独立，另有不训练的固定空间条件打乱诊断；不新测holdout、不自动换资源或加模块。

## 已完成：HYBRID-SOURCE-CORRECTION-20260915（预设目标未通过）

三臂各10000更新及完整模型/Adam保存，原20k/1k/100角色不变；126000校准记录、4500新开发行、9000冻结参考、1500短基础消融、1440完整新系统计时全部结束。1500数字PHY与6000图像指标独立复算PASS。报告`reports/hybrid_source_correction_result_20260915.md`，正式图`outputs/HYBRID-SOURCE-CORRECTION-20260915/report_assets_001/`。

没有合格校准点；可靠度gain主LPIPS0.236300，对raw差+0.052536，PSNR仅+0.06735且CI跨0；同参数SNR gain控制未显示有实用价值的可靠度增量。局部4dB有PSNR收益但LPIPS退化，不能包装为联合优势。完整限制/学习曲线/失败分层见报告。停止当前固定点自动追加，不抹除失败、不否定全部混合JSCC、不启用旧holdout作新测试。

## 新授权：HYBRID-SOURCE-CORRECTION-20260915（固定分配，不搜索）

实际激活：2026-09-15 19:07正式三臂配对训练开始，trainer2436487/队列2433287。21k缓存、8项CPU测试、GPU梯度/90条质量和30条计时工程回归已完成；详见`reports/hybrid_source_correction_launch_20260915.md`。实时状态以队列`status.json`为准；仅工程回归完成，不是新方法性能通过。

协议`reports/hybrid_source_correction_protocol_20260915.md`；配置`configs/hybrid_source_correction.json`；缓存/训练入口`scripts/prepare_hybrid_correction.py`、`scripts/train_hybrid_correction.py`。原官方冻结m7基础与1110-use真实源图残差同占N3060/E6120，接收端实际PHY错误进入训练；add、snr_gain与reliability_gain仅比较融合信息，不另建大骨干。

本次先完成工程回归和确定性缓存，再做每臂10000的有限配对训练及校准。原20k/1k/100 development角色不变；上一轮1000 holdout仅历史参考，无新holdout授权。主目标、LPIPS0.005工程非劣容差及0.5dB期望增量已先于新性能登记；没有可行checkpoint时如实标不合格，不回退冒充强基线。不自动扩展资源或模型候选。

## 已完成：COMMUNICATION-CONVERGENCE-20260915 全部收敛与独立验证

R3：2100新development质量行与480自身计时；实际整帧算术码：126000校准、12600 development、2016新链计时；独立holdout：1000源126000 VAR候选与84000冻结强参考，所有PHY/资源/源codec/图像统计审计完成。最终报告`reports/communication_convergence_final_20260915.md`。

主结论不通过“独立新质量调度方法”：arithmetic质量与BLER完全同表，主LPIPS对raw无确认增益；局部6dB源编码/保护收益明确。没有把失败记录删除、短期训练泛化为架构无效或使用失格fallback。全部源码/配置/数据/图像/统计链保留，新的训练/架构/搜索/holdout不自动启动。

## 当前：COMMUNICATION-CONVERGENCE-20260915

- R3冻结收尾已完成：2100新质量行＋46200旧参考，100个development源，原三噪声与七SNR；仅其自身480条同CPU端点计时完成。见`reports/r3_completion_result_20260915.md`，没有训练或重选checkpoint。
- 实际整帧VAR算术码已实现并CPU/GPU资格通过。两族×三个m、1000校准源×七SNR×三固定噪声的126000行实际运行中，PID2117550；不是新增架构或参数搜索。协议`reports/whole_entropy_and_mode_policy_protocol_20260915.md`与配置`configs/communication_decision_study.json`先于结果冻结。
- 三种预定校准规则为目标BLER最大源率、最终LPIPS且PSNR最多退0.25dB、最大源索引goodput；最后一项只复用同一批数据，不新增传输。完成全部校准再拟合并固定，不用development逐图质量选模式。
- 开发/独立holdout尚未开始；只登记可用文件路径，不读holdout像素。最终方法及数据清单冻结后才进入独立验证；不据本阶段更新称研究已完成。

## 当前有限执行：FROZEN-SYSTEM-ONLINE-TIMING-20260915

用户新授权仅完整计时；七个现有冻结系统、32个既有development源、五SNR、原seed2001，每条件三次计时重复，3360行已于2026-09-15 13:09完整结束，13:10独立CPU复核通过。实际波形/图像差0，所有模型起止SHA相同。70行`smoke_004`通过，前三次归档/接口工程失败完整保留。报告`reports/frozen_system_online_timing_result_20260915.md`，协议`reports/frozen_system_online_timing_protocol_20260915.md`。

主1/4/7 dB数字adaptive CPU端点TX/RX10.090/62.284ms，R2为29.388/39.535ms，Deep为2.258/3.023ms；补齐成本不等于质量创新或新holdout验证。不启动任何训练/新架构/搜索或R3评测，有限任务完成后GPU队列为空。正式图用`figures_full/`，工程smoke图用正确样本标注的`figures_smoke/`，不将其作为正式性能图。

## 当前补充：DIGITAL-ONLINE-TIMING-PREPARATION-20260915（CPU只读）

固定32个既有development源/五SNR/seed2001，m7/m8/m9/adaptive共640条旧传输输入与R2原2400条计时记录核验通过；10项CPU回归检查、155个输入SHA。不计作新传输实验，没有重新渲染/计算质量或启动GPU。报告`reports/digital_online_timing_readiness_20260915.md`。

查清旧R2设备驻留RX与数字CPU输出端点不一致，旧R2 TX也非连续端到端计时。未来需要现有冻结模型同端点测量；当前仅输入准备完成，没有GPU计时driver或自动队列。收敛报告revision 2不变，新增架构/搜索/训练仍暂停。

## 当前：CONVERGENCE-AUDIT-20260915（只读收敛审查）

窗口2026-09-14 01:28:36—2026-09-15 01:28:36（本机UTC+8），4组已完成系统比较/15个模型训练版本；输出`outputs/CONVERGENCE-AUDIT-20260915/`，报告`reports/convergence_report_20260915.md`。没有新传输/模型推理/训练或搜索，完整源图、checkpoint和失败记录保留。合法完整校准早期最优与失格warmup fallback分开。

数字VAR自适应重新作为候选主系统，学习支线仅保留R3待验证资格，其他冻结为控制。R3窗口后02:00已按原10000结束训练/校准，但没做最终系统评测；当前VAR_COMM无活动GPU队列，不干预其他项目。新增架构/自动参数搜索/新大网络/复杂机制继续暂停，不按下面历史计划自动启动。

## 当前：WETOK-REENCODING-VECTOR-CONTROL-R3（已激活）

本机UTC+8 2026-09-15：实现/CPU原始初始化/GPU图像梯度/4项CPU与实际driver断点测试通过。首1000完整校准和模型Adam/数据/功率/选模审计PASS，00:38实际续到10000，1099666/1099667/1099668；00:46核验2000。唯一融合向量q对y−q，均保留标量残差能量门控，整体向量大小/方向都可能变化，不单独宣称方向效应。`reports/wetok_r3_activation_20260915.md`。

最终23方法/48300主质量行和16模型/2560独立计时只完成协议/配置登记，驱动与收尾尚待补齐；当前没有新R3 development或正式test结果。旧R2完整参考只读，后续入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`。

## 当前：WETOK-REENCODING-VECTOR-CONTROL-R3（登记，未训练）

R2于2026-09-14完整结束：主MS−普通full-grid LPIPS+0.002350且CI全正，残差版−0.009803优于普通full-grid但仍输主区间数字自适应/Deep。46200质量主行、48400图像CPU审计和2400独立计时完整；报告`reports/wetok_r2_result_20260914.md`。不重启旧R2队列。

后续只补向量归因对照：三次16×16读取、两次E、同参数/门控/10000训练机会，仅融合q而不是y−q。两者门控都保留残差能量，不能称完全无残差；Joint训练后不同s/y只匹配源/噪声/N/E。原7000父点和完整初始化/Adam/数据资格需核验，当前未实现或启动。入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`。

## 当前：WETOK-JOINT-SUFFICIENCY-R2的10000系统评测

2026-09-14 23:09四臂10000训练/全校准/模型Adam/数据与选模审计PASS。single/MS部署选择7500，两个full-grid选择10000；各臂均拥有10000机会。当前质量983552/收尾880742/观察880743，23:12提交23/100原development源图；完整质量和独立计时尚在流程中。19项CPU测试PASS；报告`reports/wetok_r2_milestone10000_20260914.md`，入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。不重启已结束训练，不访问新test，不省略同WeTok数字强参考。

R2辅助报告（2026-09-14 22:09）：7500完整训练/校准曲线与可对账新增成本已封存，`reports/wetok_r2_training_curves_20260914.md`；原15模型质量/计时准备不变，新增只读绘图工具通过4项测试，全套15项CPU测试PASS。实时训练至8400，并未执行新的R2 development或正式test。

## 当前：WETOK-JOINT-SUFFICIENCY-R2，7500→10000

2026-09-14 21:35，四臂7500完整校准与实际模型/Adam、历史数据、功率、选模审计PASS，四臂均有LPIPS改善；21:43从真实7500保存点共同续到10000，21:46已保存7600。trainer880740/reviewer880741，自动质量/统计/独立计时finisher880742与被动GPU observer880743已运行；不重开旧769675队列。

评测/计时与断点测试准备完成，11项CPU测试PASS。最终四R2＋18旧参考＝46200主行，78个比较/3744主配对区间；独立15模型32源×五SNR×seed2001＝2400计时行/70区间。模型选择使用完整校准，100图monitor不选模；多训练、同机会结构、历史强系统三类比较明确分开。质量可显式共享、计时单独准入，未有新R2 development结果。`reports/wetok_r2_milestone7500_20260914.md`；`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。

## 当前准备：WETOK-JOINT-SUFFICIENCY-R2

2026-09-14 20:09已进入真实续训，四臂继承model+Adam5000到7500；20:26核实5500总更新、500新增、global12500，旧5000历史不变。trainer769675/reviewer785157/observer785158，CPU恢复与真实driver中断等价通过，损失/学习率/预算未改；无R2新开发结论。`reports/wetok_r2_activation_20260914.md`。

在Grid5000完整质量/计时结果后，准备四臂相同新增5000更新的训练充分性对照。四个实际5000模型/Adam、global12000、原数据/噪声/冻结身份已CPU核验；没有启动GPU训练，driver尚待实现。`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。

前轮Grid的18方法质量与11模型独立计时完整完成，4/8/16未展示独立LPIPS/PSNR/时延优势；Joint全网格innovation最佳但仍落后强系统LPIPS。`reports/wetok_joint_grid_result_20260914.md`。所有旧结果与模型保留。

## 当前：WETOK-JOINT-GRID-CONTROLS-R1

2500更新/完整校准审计于2026-09-14 18:29完成，两臂实际续5000（650391/650392）。新评测保留18方法、37800质量主行，新增模型推理与冻结参考复用分明；与固定32图/11模型的1760行独立计时分开。18项CPU测试通过，自动finisher689944已挂载；当前尚无新Grid开发结果，不把准备/校准当系统胜出。`docs/evaluation_preparation_20260914.md`位于该实验目录。

2026-09-14从原7000父点开始两臂匹配Joint训练：full_grid_state_history和full_grid_innovation，仍N3060/E6120、原loss/数据/噪声/预算。6项CPU测试及真实GPU零更新图像梯度检查PASS，17:27实际激活，已核实300步模型/Adam；trainer570837/reviewer579794。新Grid还没有development结果，不将参数相同冒充计算量相同。入口`experiments/wetok-joint-grid-controls-r1/docs/CONTINUE.md`。

原Joint5000完整评测/统计已于16:48结束，33600主行全部保留。条件版对single有小幅LPIPS增量，但对强全网格差值CI跨0，仍不及数字/Deep的LPIPS。结果`reports/wetok_joint_result_20260914.md`。

## 当前：WETOK-JOINT-SENDER-R1

2026-09-14 16:01原评测因GPU竞争退出，49张已提交、5000训练与校准完整保留。恢复监督器473779/观察器473780于16:12启动，等待空闲再原协议`--resume`，之后自动统计；资格核验和31项CPU检查通过，不重新训练、不变更原源码/参数/样本。`experiments/wetok-joint-sender-r1/docs/evaluation_recovery_20260914.md`。

5000更新与全校准审计于2026-09-14 15:50完成，三Joint均选5000。条件版校准LPIPS .194679优于Joint single .197125，但R-only full-grid .187311更强。当前自动回放三Joint＋六R-only＋七系统，evaluator443167/finisher348503，尚未得到完整Joint development结论。报告`reports/wetok_joint_milestone5000_20260914.md`。下方训练中状态为历史，不重复启动。

2026-09-14续训正常，15:00已核实3100保存点。被动GPU并发记录从14:44开始，不改变质量实验；新增5项观察器测试，共27项CPU测试通过。HJSCC小型源码/协议核查单独完成但尚无模型或性能复现，详见`reports/hjscc_source_protocol_review_20260914.md`，不取代当前注册的强控制。

2500完整校准及审计已PASS；2026-09-14 14:37三臂从真实末端模型/Adam继续5000（348501/348502/348503），全量评测与CPU统计已排队。single/state选中2500，no-history仍选0；对匹配R-only有小幅校准改善，条件结构尚未胜single。报告`reports/wetok_joint_milestone2500_20260914.md`。新增收尾工具后22项CPU测试通过，当前未有新Joint development或holdout结果。

2026-09-14 12:39已激活Joint三基础结构，首1000里程碑；相同原7000父点和fresh Adam规则、相同数据/噪声与5000新更新机会，唯一训练因素为E是否也优化。原R-only5000控制、全量评测/分析、reference资格、GPU梯度与16项CPU检查均PASS。训练尚无新性能结果。`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`。
原WETOK-INNOVATION-R1已完成冻结，结果见`reports/wetok_innovation_result_20260914.md`；以下运行条目为历史，不据旧PID重启。

## 当前：WETOK-INNOVATION-R1

2026-09-14 11:10已从恢复点继续到4100；原5000评测仍由恢复后的finisher自动接续。后续Joint评测准备完成：16方法、九模型新测、全部六R-only对照，16项CPU测试PASS；实际Joint训练与GPU评测尚未执行。

2026-09-14恢复：旧队列中断于3000保存点附近；旧进程消失、系统已重启。核验六臂真实模型/Adam/数据噪声后，从3000续至5000并重新挂上审计/评测队列，3100已保存。`experiments/wetok-innovation-r1/docs/recovery_20260914.md`。不把中断时间计为训练，不误称5000已经结束。

2500完整校准/审计已结束；full-grid .196310优于多尺度innovation .200449。六臂共同续到5000，自动收尾评测进程已排队（3211225）。`reports/wetok_innovation_milestone2500_20260913.md`。Joint仅准备与真实GPU零更新profile通过，未训练，须等待完整控制与结果归因。

后续准备（未启动GPU训练）：`experiments/wetok-joint-sender-r1/`只改变通信E是否接受梯度，与现有R-only三基础结构同起点、同新训练预算；7项CPU检查及真实权重初始SHA核验通过，等待实际完整控制与GPU profile。动机来自父点训练曝光与校准曲线审计，不把多训练本身或可训练参数变多当论文创新。

首1000更新/完整校准/审计已于本机06:00完成，创新量选中LPIPS .202062，尚不胜single .198920或prediction .201300。六臂已从实际末端模型/Adam共同继续2500；no-history虽选中0，也从真实1000末端续，不重置训练历史。`reports/wetok_innovation_milestone1000_20260913.md`。

geometry同预算开发评测已完成并保留负交互结果。当前固定同一发送器，进行single/no-history/state-history/prediction-feature/multiscale-innovation/full-grid-innovation六臂RX-only比较；相同y、N/E、数据/噪声和训练机会。CPU/真实GPU检查通过、首1000训练/审计已完成，尚无新机制系统成功结论。`experiments/wetok-innovation-r1/docs/protocol.md`。

评测准备已补齐：27300主行、600固定支持点、1968主配对区间和108支持点区间，逐帧共享s/y与冻结E/选模核验、全部失败保留。15项CPU/合成整链测试通过，旧参考441条PSNR和21波形只读预检通过；训练/profile绑定源码和配置未改。`experiments/wetok-innovation-r1/docs/evaluation_protocol.md`；尚未执行新接收器development评测。新时延评测轮换六臂次序，并保存源图前后温度/时钟/节流记录。

## 当前：WETOK-GEOMETRY-20260913

总4500/image2500全校准审计已完成，geometry收益保持但条件结构未确认。当前三臂从真实末端继续总7000/image5000；旧控制历史不变，之后运行匹配完整预算开发评测。

首总3000已完成，匹配image1000预算下全校准LPIPS约.207，对旧geometry约.256–.261改善；只作校准证据。现继续总4500(image2500)，两个geometry的比较训练资格保持一致。新最终评测入口只接受总7000匹配预算，CPU检查通过，未提前访问新模型development。

本机UTC+8 2026-09-13启动204×30三结构训练，固定6120实坐标/N3060/E6120、连续接口、视觉模型与原loss；153×40完整2000+5000历史控制经过资格核查后只读复用。31项CPU与真实GPU无更新profile通过，初始目标总3000，未形成新geometry性能结论。入口`experiments/wetok-comm-v2-20260912/docs/geometry_execution.md`。
前接口5000比较已完成：主区间出现同接口条件增量，但仍不胜强系统；`reports/wetok_interface_milestone5000_2026-09-13.md`。以下v2接口运行条目均为历史，不再默认重开。

## v2当前执行：WeTok原生通信A0/A1（2026-09-12）

20:47接口2500开发评测/统计完成，33600+600行、独立3024区间复算通过；条件continuous LPIPS不胜无历史或强系统。20:57九臂从真实末端共同恢复到总追加5000，配置和N/E不变；不把继续训练当收益保证。报告`reports/wetok_interface_milestone2500_2026-09-12.md`。

18:16接口九臂完成1000更新+全校准/审计并继续至总追加2500；完整calibration的continuous LPIPS single/no-history/conditioned为.259012/.261040/.256377，不是development系统成绩。25项CPU测试及Deep支持点补充检查通过；当前运行入口train_interfaces --until-additional 2500 --resume。详见docs/interface_continuation_2500.md。

16:06已启动接收接口九臂流程（首1000检查点）：三结构×native identity-ST/native有界ST/continuous。相同2000模型父点、统一fresh Adam、原joint loss、固定N/E，19项CPU及九臂真实GPU无更新profile通过。正进行完整起点校准；后续按完整校准共同续训，最终development评测尚未做。协议`experiments/wetok-comm-v2-20260912/docs/interface_protocol.md`，接续`docs/interface_execution.md`。

此前bit权重六臂15:52因固定监控继续严重退化而探索性提前停止；共同追加1000现场保存、计划5000未完成，不把该停止当整个研究终点。报告`reports/wetok_bit_support_stop_2026-09-12.md`。

以下为之前的启动记录：

首5000原配方里程碑已完成，联合图像训练出现感知/bit退化，未支持条件历史的明确增量。已于15:06启动单变量bit-BCE 0.01→1的六臂配对续训：同5000模型/Adam、同新增预算/数据/噪声，不改网络/物理/接口。协议`experiments/wetok-comm-v2-20260912/docs/bit_support_protocol.md`；首250追加更新审计通过。

原生±1 Fq全局发送，single-pass／三阶段无历史／三阶段条件三臂，同N3060/E6120/训练机会；不是codec换代本身的创新，不是逐包渐进。
原生缓存、100图identity回放与真实image-gradient检查通过；13:00已正式训练，5000为首里程碑而非总上限。旧暂停/200元/半天限制由根目录v2任务书覆盖。
入口`experiments/wetok-comm-v2-20260912/`，配置`configs/study.yaml`，协议`docs/protocol.md`；native cache/8PSK数字对照及日志位于本项目outputs，旧结果只读。

## 已完成并复核：ei-liulu-xqvar-eval-20260912-v1（无训练骨干选型）

- 授权：本机已有GPU整条流水线自然结束后；不再使用需要认证的开发机、不停止任何已有任务。
- 数据：同100张development，旧预处理/PSNR/SSIM/LPIPS/DINO；非新test。
- 全重建：旧官方、旧fidelity、XQ实际配套tokenizer、WeTok指定8192bit完整codec。
- 条件恢复：XQ真实自然prefix8/9（静态2424/3960rawbits）直接/配套VAR补全/full；原3060/5088raw点只作离散源表示参照，非复信道等资源。
- 状态：01:46自动启动、01:49完成，1800正式行；独立CPU产物/统计复核PASS。**暂不正式迁移到XQ；WeTok完整codec得到保真/感知共同改善，但不直接接通信。** 本轮新增训练0步。
- 协议：`experiments/backbone-eval-20260912/docs/protocol.md`；结果解释：`reports/backbone_selection_review_2026-09-12.md`；输出 `outputs/ei-liulu-xqvar-eval-20260912-v1/`，不覆盖旧记录。

## 已完成通信骨干2×2：VAR-TOKEN-BACKBONE-20260911

2026-09-12 01:43（UTC+8）完成四臂各20000更新、16800行development评测及统计。增强对小next-scale的LPIPS差-0.001586，95% CI跨0；仍未追平固定数字m8的LPIPS。
两条parallel无合格PSNR候选，回退warmup；不将该回退对照的巨大差值当作B配方结构增量。摘要`reports/token_backbone_result_2026-09-12.md`，原主表和receipt保留不改写。以下为启动时记录。

小/增强×parallel/next-scale均按完整10000 warmup+10000 B配对训练；保留旧B为冻结工程参考，不用其继承历史替代严格比较臂。只改通信E/D。
实现、CPU合成检查与真实GPU无更新检查已通过；增强5139662参数、匹配SNR小骨干1559054参数。新结果尚未形成；入口`../var-next-scale-comm/scripts/run_token_backbone.py`，新协议在该仓库`reports/token_backbone_protocol_2026-09-11.md`。

## 2026-09-11本机运行完成：VAR-M8-LOCAL-20260911

固定m8的A同预算原配方续训/B两阶段通信训练于13:42:19（UTC+8）完成训练、12600行development评测及统计；每分支新增10000更新，B为2000+8000，不引入其他方法变量。
执行位置`../var-next-scale-comm/`，配置`configs/m8_local_20260911.local.yaml`；选中A step2000、B step6000。主1/4/7 dB的B相对A LPIPS改善27.60%，但仍未超过固定数字m8/自适应/感知Deep的LPIPS。摘要`reports/m8_local_result_2026-09-11.md`；所有旧准备/暂停条目为历史记录，不自动追加实验。

## 当前独立工作区实验

| 实验 | 状态 / 主要作用 | 本地报告 |
|---|---|---|
| 固定m8原配方续训 vs 两阶段通信训练（远端准备） | 当前指定步骤，准备与CPU自检完成；未训练/未GPU测速，代码在新仓库 | `../var-next-scale-comm/docs/REMOTE_M8_HANDOFF.md` |
| 多尺度通信×保护分配×next-scale条件接收（2026-09-10方向更新） | 用户更新方向；设计口径已记录，尚未实现/训练/测速/租卡；旧m8续训仍暂停 | `reports/multiscale_joint_direction_2026-09-10.md` |
| 两阶段prefix恢复对照（2026-09-08） | 用户要求暂停；实现/CPU检查完成，新增训练0步，无存活队列；重新启动须用户明确授权 | `reports/prefix_refinement_protocol_2026-09-08.md` |
| 已有校准与固定m8对照复核（派生分析） | 完成，无训练/推理；next第二轮校准LPIPS全SNR退化，固定m8数字链在开发七SNR的LPIPS/DINO仍更好；72组原配对复算差0 | `reports/prefix_calibration_fixed_m8_review.md` |
| m8全局prefix-JSCC训练（2026-09-07） | 两版本完成各10000更新/40k曝光，冻结VQ/VAR/image decoder，最后6000更新全self-history；均选epoch2 | `reports/learned_prefix_result_2026-09-07.md` |
| 图像梯度工程自检（同日） | 通过；MSE/LPIPS到E/D梯度非零、梯度穿过冻结VAR、train/eval硬前向像素差0；不作科学性能gate | `outputs/VAR-PREFIX-JSCC-GRADIENT-001/selfcheck.json` |
| 训练预算/校准选择审计（同日） | 通过；相同初值/更新数，独立image-only校准，DINO未优化，冻结权重不变 | `outputs/VAR-PREFIX-JSCC-TRAIN-AUDIT-001/audit.json` |
| 学习prefix开发评测与审计（同日） | 10500行完成；next相对parallel有8.921% LPIPS增益，绝对性能未胜数字自适应；7348图/4200硬前缀重建复核 | `outputs/VAR-PREFIX-JSCC-EVAL-AUDIT-001/audit.json` |
| DeepJSCC条件支持检查（同日） | 主结果冻结后补充；实际5/6 dB、NN固定条件4/7，600行，无训练/物理噪声变化；不把原off-grid异常算优势 | `outputs/VAR-PREFIX-DEEP-COND-SANITY-001/summary.csv` |
| 整帧4前缀/末尺度辅助译码（2026-09-07） | 完成9600行；未胜列表65，条件真前缀恢复饱和但覆盖不足；`STOP_THIS_FINITE_PREFIX_VAR_RECEIVER` | `reports/whole_frame_prior_result_2026-09-07.md` |
| 整帧列表独立审计（同日） | `AUDIT_PASS`；322万候选/3984概率表/1240图复核，638新图旧实现像素差0 | `outputs/VAR-WHOLE-FRAME-PRIOR-AUDIT-001/audit.json` |
| 完整接收API时延补测（同日） | 20固定图×4SNR×6方法=480调用，输出全复现，不修改主统计 | `outputs/VAR-WHOLE-RECEIVER-LATENCY-001/summary.csv` |
| 波形到RGB时延补测（同日） | 20固定图×2SNR×3核心方法=120调用，像素差0；6 dB VAR/list65约212/69 ms | `outputs/VAR-WHOLE-IMAGE-LATENCY-001/summary.csv` |
| 7 dB首个CRC失败候选图像回放（2026-09-07） | 完成；固定m9/3060/原波形与FEC，VAR追回64.74%差距但二候选oracle仍输整帧；`STOP_TWO_CANDIDATE_SELECTOR_FIXED_M9` | `reports/crc_failure_replay_result_2026-09-07.md` |
| CRC失败回放独立审计（同日） | `AUDIT_PASS`；1200条原轨迹、219张新图用旧实现复渲染像素差0、580张图完整重算指标 | `outputs/VAR-CRC-FAILURE-REPLAY-AUDIT-001/audit.json` |
| 第9尺度同观测译码（2026-09-07） | `PASS_SINGLE_SCALE_GATE_ONLY`；100图×3SNR×5噪声×5先验=7500行，7 dB正确恢复率87.8%→97.8% | `outputs/VAR-SINGLE-SCALE-CHANNEL-001/completion.json` |
| 单尺度独立审计（同日） | `AUDIT_PASS`；波形、CRC、7500行统计与9个独立NumPy MAP最优值复核 | `outputs/VAR-SINGLE-SCALE-CHANNEL-AUDIT-001/audit.json` |
| 真实逐尺度与强对照（同日） | 12000行完成；同分组机制通过，整帧自适应/熵编码强对照未胜出 | `reports/next_scale_decoding_result_2026-09-07.md` |
| 完整链路独立审计（同日） | `AUDIT_PASS`；12000条可信状态轨迹、1555张独特重建全部重算指标 | `outputs/VAR-PROGRESSIVE-CHANNEL-AUDIT-001/audit.json` |
| Next-scale先验诊断（2026-09-07本机日期） | `PASS_PRIOR_GATE_ONLY`；100图×2尺度×4先验，完整token log-loss过门槛；没有FEC/CRC或图像质量结果 | `reports/next_scale_prior_result_2026-09-07.md` |
| 独立缓存审计（同日） | `AUDIT_PASS`；重算800行指标/CI、静态频率及1200图内容SHA，不是第二遍模型推理 | `outputs/VAR-NEXT-SCALE-PRIOR-AUDIT-001/audit.json` |

最新整帧原型预注册：`reports/whole_frame_prior_preregistration_2026-09-07.md`；未训练/改变发送/FEC或5.5 dB阈值。
4/5 dB无真前缀进入4候选，6 dB仅65/296；条件先验有效但不胜普通CRC列表的图像收益/成本。
大列表受候选/堆上限限制且有误接受，未声称严格等算力或直接替换主系统。
强整帧熵编码未实施，不把保留旧分组熵编码记录当成完成了该强对照。

前轮回放预注册：`reports/crc_failure_replay_preregistration_2026-09-07.md`；产物`outputs/VAR-CRC-FAILURE-REPLAY-001/`。
3000行包含900个分组接收记录的discard/retain/oracle输出与300个原整帧对照，不是3000次独立传输。
100张原development、7 dB、原三噪声，未训练/换FEC/重分资源/新搜候选。两候选选择路线按预定oracle界限停止。

第一关预注册：`reports/next_scale_prior_preregistration_2026-09-07.md`。
主输出：`outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/`；同类donor明确作为development，旧calibration仅用于频率拟合。
第二/三关已接续执行，详见上表；先验门槛通过不等于系统门槛通过。

第二关协议：`reports/single_scale_channel_preregistration_2026-09-07.md`。
第三关协议：`reports/progressive_channel_preregistration_2026-09-07.md`。
该前轮三关状态`MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN`，不替换原整帧自适应主基线。

工程记录：首个C++自检在穷举例中发现初始化顺序错误，修复后SELFCHECK-002通过；
第三关smoke001后补全mode误接受计数，smoke002通过，实验参数不变。失败/smoke目录保留，不计科学样本。

## 历史归集

2026-09-06归集。下列结果此前已在旧工作区执行，**不是本次重新跑出的结果**。
历史原始输出根目录及各报告SHA见 `historical_sources.json`。

| 实验 | 状态 / 主要作用 | 本地报告 |
|---|---|---|
| Decoder-only full/m8/m9微调 | 已完成；失真提高但感知/语义退化 | `reports/var_decoder_only_m89_result_2026-09-03.md` |
| VAR-m8真实无线诊断 | 已完成；分离接收误码与补全损失 | `reports/var_m8_wireless_diagnostics_result_2026-09-04.md` |
| 固定3060-use自适应scale/FEC | 已完成开发机制验证；保留m7/m8/m9基线 | `reports/var_fixed3060_adaptive_scale_result_2026-09-04.md` |
| 同前缀语义/码率贡献消融 | 已完成；同payload下比较prefix-only与VAR completion | `reports/var_semantic_rate_contribution_result_2026-09-04.md` |
| Receiver-prior跨数据集验证 | 已完成历史正式验证；不得再用该数据声称未见测试 | `reports/var_receiver_prior_cross_dataset_result_2026-09-04.md` |
| 同前缀cell约束修正 | 完成并停止；6400行argmax主消融、3600行三种子诊断 | `reports/var_prefix_consistency_result_2026-09-06.md` |

cell约束修正预注册：`reports/var_prefix_consistency_preregistration_2026-09-06.md`。
该候选最终状态为 `STOP_THIS_LATENT_CORRECTION_CANDIDATE`，不是VAR通信主线整体停止。

## 后续记录约定

新实验在本目录登记数据角色、共享信息、真实预算、模型/源码版本、随机种子、对照、判据和输出位置。
新输出写入本项目的 `outputs/`，不覆盖历史输出，不默认写回旧项目。
未运行的建议不得记为已完成；只做目录整理不记作新算法实验。
