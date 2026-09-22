> 2026-09-23 repository migration: the actual VAR_COMM root is the sole Git worktree. Historical research claims below are not newly certified by this migration. No A/B retraining or GPU quality validation was performed. See reports/repository_migration_20260923.md.

# VAR 通信当前进度

## 最新：latent增强后续对照全部完成（2026-09-19收口）

- 数字校准动作完成：N3572/N4084、raw/arithmetic、m7/m8/m9、D0/Dc，360000 calibration rows；m10按当前header mode域不纳入。动作及development结果位于`outputs/VAR-LATENT-ENHANCEMENT-20260917/followup/digital_policy_v2/`与`digital_policy_development_v1/`。
- 修正计时完成：650调用，旧数字重建最大像素差0；纯数字不再计算Fb_TX，Dc不再重复D0，噪声移出RX计时。产物`followup/online_timing_v2/`。
- N4084折中`3504 digital + 512 continuous`完成1500帧，五SNR均值PSNR21.5841、LPIPS0.147276、DINO0.870658；1dB body CRC失败299/300，4dB以上保留明显连续增强收益。
- 1024 Decoder适配两臂均完成10000步安全上限；development适配臂22.2190/0.131421/0.873083，控制22.0985/0.135576/0.874250，配对提升约+0.1205dB、LPIPS−0.00416，DINO区间跨0。
- 当前结论：保留1024连续增强和Decoder适配；折中分配只作为低SNR候选，不自动组成SNR自适应系统。没有新holdout、不追加训练、不启动新骨干。

## 当前：latent增强development测评与在线计时完成（2026-09-18 11:47 UTC+8）

- 100张原development图×5 SNR×3噪声、41个方法共61500行质量结果完成；未访问新holdout，冻结模型SHA未变化，所有失败保留。协议`reports/latent_enhancement_evaluation_protocol_20260918.md`，结果`reports/latent_enhancement_development_result_20260918.md`。
- 关键结果：512增强N3572为21.3479/0.157741/0.853351，1024增强N4084为22.0674/0.136225/0.874412（PSNR/LPIPS/DINO）；同资源raw m8/Dc分别20.1531/0.171924/0.867911和20.3061/0.166890/0.873181。512/1024质量优势有源图配对区间支持，但512 DINO略低，1024 DINO差异跨0。
- 1dB仍受m8基础body失败主导：m8/增强均300/300 body CRC失败；增强层不能保证低SNR感知质量。4dB及以上新增观测收益明显。
- 固定10张development图×5 SNR×seed2001的650次原始TX/RX计时完成，但发布复核发现纯数字路径统一计算Fb_TX、Dc/算术路径存在重复工作、噪声生成落在RX区间；因此计时只作诊断，不能作速度排名。原始产物`outputs/VAR-LATENT-ENHANCEMENT-20260917/online_timing_v1/`和复核说明已保留。
- 本轮不追加训练、不改checkpoint、不访问holdout。新增预算的校准自适应/可行全尺度数字对照和统一在线计时仍是缺口；当前latent增强候选保留为固定模式下有正面development证据但尚未最终定稿的系统。

## 当前：阶段B按用户决定收口，保留训练与结果（2026-09-18 10:25 UTC+8）

- 用户认为近期提升已很小，要求到当前为止。B trainer在配对更新边界安全退出，最后保存更新44510；没有强制kill、没有删除checkpoint/日志/校准结果。原controller的exit130另以`training/user_stop.json`登记为用户主动停止，不是训练失败。
- 训练固定选中点：512为30000步、1024为40000步、receiver-only为35000步；最后完整校准为40000。阶段A此前已到40000安全上限并选中38000步Dc。
- 收口报告：`reports/latent_enhancement_stage_B_training_closure_20260918.md`。B校准已显示新增512/1024观测相对同Dc无新增传输精化器有收益迹象，但同总N数字对照、development质量、配对区间和在线成本尚未做，不能宣布通信效率结论。
- 不自动续训、不扩大预算、不启动新loss/骨干/功率搜索。若继续，只做既定选中checkpoint的development和同资源对照；不访问新holdout。

## 当前：A已完成，B自动接续成功、每臂约1.2万步（2026-09-17 22:46 UTC+8快照）

- A于17:35:10达到40000安全上限结束，不宣称完全收敛；按原规则选中38000步Dc。原VAE保持不变，1000源校准连续参考Dc(F)相对D0(Fq)为PSNR+3.632995dB、LPIPS−0.052360，源图配对区间支持改善；这些不是有限带宽通信成绩。
- B控制器1605981实际自动接上。17:35:40真实GPU/原PHY/图像梯度检查通过，原基础RGB差0，D0/Dc均未改变；18:18:18实际RX缓存完成，保留全部215000帧，43954次新VAR补全。18:18起进入B训练流程，现trainer1745684每臂约12150步；旧PID仅为快照，不据此重复启动。
- 最新完整校准是每臂10000步，1000源×五SNR×三噪声、每方案15000传输，全部失败保留。五SNR平均PSNR/LPIPS：原m8+D0(N3060)18.7973/0.196227；仅Dc(N3060)19.4677/0.183585；receiver-only(N3060)19.6872/0.184131；增强512(N3572)20.9967/0.156736；增强1024(N4084)21.7291/0.137339。不同N不作等资源排名，数据是校准而非development。
- 5000→10000步，512/1024的LPIPS分别再降0.002284/0.003753；控制PSNR小增但LPIPS略退。三臂均按已登记规则继续、更新机会相同，尚不能称充分收敛。无失败记录，GPU约99%/6.6GiB，温度实时83–84℃，控制器仍监控。
- 同N3572/4084的强raw/整帧算术数字及同Dc对照、development三指标/源图配对区间、完整在线TX/RX成本尚未完成。当前只能说有新增观测的校准收益迹象，不能宣布资源效率或系统胜出。实时入口仍为`stage_B_v1/auto_001/status.json`及`stage_B_v1/training/status.json`。

## 当前：B自动接续真正挂上，等待A结束（2026-09-17 17:01 UTC+8快照）

- A实际38530步、最近选中38000，源码/配置绑定未变；此前“B未实现/未接好”是历史状态，现在已补齐独立`phase_b/`代码与条件控制器。
- 新控制器PID1605981，实时状态`outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v1/auto_001/status.json`，当前`WAIT_FOR_A`。它等A真实完成、资格通过并锁定其选中Dc，依次运行GPU工程检查、实际接收latent缓存、三臂配对训练，不需再次请求开始。A未结束，因此此时B参数更新仍为0。
- 13项CPU测试通过，含真正三臂训练循环在让卡后恢复的模型/Adam/数据序列与不中断运行逐项一致。实际数字回放：训练200000帧，校准15000帧；所有CRC/header失败保留，不修复候选。
- 模型参数：512增强527716，1024增强532328，receiver-only263776。RX公共主干相同，控制无TX和观测stem；不冒称总参数完全匹配。原loss、N/E、SNR、原两预算及Dc冻结输入梯度规则均未改。
- 独立控制器每30秒记录运行状态/GPU和异常，旧A观察器仍在。只读等待控制器曾在尚无B子进程时刷新以载入最终通过测试的版本，回执已留存；没有停止A或其他任务。任何资格/实现失败停止留档，不无限重试或自动改配方。
- 待完成项仍包括实际GPU资格、B训练与校准、development/强数字同资源对照及完整成本；不把接续完成写成通信机制成功。

## 当前：Stage A超过20000步，持续监控已挂上（2026-09-17 10:03 UTC+8快照）

- 20k训练＋1k校准FP32缓存全部完成；Stage A实际20030步，20000步完整1000源校准成为当前选中点。Dc(F)为PSNR26.1106、LPIPS0.051777，混合校准效用0.0118028；是连续信息参考，不是有限带宽通信成绩。仍按原曲线/停止规则继续，Stage B未启动。
- 用户要求持续监控，新增独立CPU观察器`experiments/var-latent-enhancement-20260917/monitoring/watch_stage_a.py`，每60秒检查进程身份、训练/校准活动、loss/梯度、选模、GPU、磁盘与冻结源码SHA。观察器不在训练源码目录，不改动已经注册的训练实现/配置/优化器，不占GPU、不停止其他任务。
- 状态`outputs/VAR-LATENT-ENHANCEMENT-20260917/monitor_001/status.json`，历史`history.jsonl`，校准/异常事件`events.jsonl`。启动PID1081377仅为快照，以实际命令及状态为准；6项监控测试通过，首检无告警、源码绑定未变，GPU100%/19690MiB/78℃。
- 异常或阶段A结束会保留本地告警/完成记录；没有配置邮件/消息/聊天自动推送，不声称已向用户发送离线通知。观察器不自动重试一般失败、不更改loss/批量/停止阈值，也不未经真实实现资格就启动阶段B。

## 当前：新latent增强实验已占用空闲GPU，缓存→阶段A（2026-09-17 01:58 UTC+8快照）

- 用户明确授权独立的m8/N3060基础＋512/1024连续latent符号，不改旧m7/RGB混合结论。协议与训练/停止规则见`reports/latent_enhancement_protocol_20260917.md`及新实验`docs/stage_A_execution.md`。
- GPU0确认空闲后，真实模型资格检查通过：F形状32×16×16，全部真token累计Fq与官方映射差0；1dB body CRC失败和19dB成功两例基础RGB均与原实现像素差0。Dc副本训练后D0/原VAE未改，冻结Dc仍有输入梯度；四样本反向实际峰值分配约14.92GiB。工程检查更新已丢弃，不是正式训练成绩。
- 八项CPU测试通过，正式流水线PID466481/cache PID466513（仅为启动快照，不据旧PID盲目重启）。已完成1000校准缓存和300/20000训练缓存，FP32源latent、全量量化latent和确定性m8基础latent；缓存后自动执行Stage A。状态入口`outputs/VAR-LATENT-ENHANCEMENT-20260917/pipeline_001/status.json`。
- Stage A使用独立post_quant_conv+decoder副本，固定50/25/25输入和MSE+0.1LPIPS。每500步固定子校准、每2000步完整校准；基于曲线判停，40000安全上限不作收敛证明。尚无实际训练/科学收益结论；Stage B及开发/数字资源对照仍待后续，整个实验未完成。
- HiFi及其自动续跑器按新任务优先级暂停，保留原1500/20源记录。之前一次非持久后台启动未留下缓存或训练更新，正式运行已改为受监控常驻session。新流水线遇其他授权GPU进程会保存让卡，不终止他人任务。

## 20源阶段91/100已保存，等待外部GPU任务释放（2026-09-17 01:01快照）

耗时排查和独立分支验证已完成，报告`reports/hifi_runtime_profile_result_20260916.md`。20源阶段在91/100帧后遇到外部CAP-VPR GPU进程，原保护规则主动退出；不是采样失败，没有删除/重选图像或修改作者参数。

新增`resource_resume_001/status.json`空闲续跑监视器（启动PID382116，实时状态为准）：只在GPU持续空闲时用原版本补余9帧，若其他GPU任务出现只中断我们自己的worker，不操作其他项目/共享环境。原1500计划及240帧仍保留暂停，不冒称完成。

停止附近19dB源47/52的原时延保留并标注资源不确定性，二者不在原共同10源时间比较中；后续争用事件继续留痕。质量仍全100计划，不因资源问题丢失败输出。说明`reports/hifi_stage20_gpu_contention_20260917.md`。分析代码仅增加该披露和标记，旧准备SHA保留，修订记录`analysis_contention_disclosure_revision.json`；物理、采样和质量统计不变。

## HiFi耗时诊断完成，20源探索队列执行中（2026-09-17）

按用户最新授权，原1500计划与240已提交帧保留并暂停；新阶段固定20×五SNR×noise2001，复用48帧，仅补52帧。入口`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/hifi_stage20_20260916/pipeline_001/status.json`，没有改采样来加速，也没有部署新分支。

性能报告`reports/hifi_runtime_profile_result_20260916.md`：原240帧=720记录视图但只有240次扩散，元数据额外恢复0。三校准工作点关闭attention checkpoint仅约0.04%–0.93%完整RX降幅，allocated显存增加约300MiB。1dB仪器轨迹的重算正向约0.98秒，已含于约51.9秒根输入反传；不能重复相加或误算整次扩散。

9个匹配状态梯度均有限/非零，6/9通过全部严格阈值；三个完整AB输出未过原RGB阈值，AA原版重复也有同量级差异。所有数组/记录保留，不改阈值，不把数值问题当科学质量负结果。原版保持。

性能图在新阶段`runtime_figures_002/`（001布局尝试保留），23项CPU检查通过。阶段分析程序已准备，完整100帧与指标后才出`hifi_stage20_positioning_20260916.md`；所有系统同20源质量，时间另统一用共同已有10源，HiFi100帧全部时延单列。无新训练、骨干或holdout。

## 本次定位材料已提交并推送（2026-09-16 22:12）

按用户“这次的先提上去”授权，一页定位、原文差异表、HiFi收尾说明/代码/测试及三类图与失败后果表已提交并立即推送到`daiqizai/var-comm`的`main`。提交`4be9faa9ec5bcb397e382c81dc34981ebaba195f`，远端SHA核对一致，发布工作树干净；23文件变更，未强推。

公开入口`results/external_baselines/positioning_interim/README.md`，首页和研究状态已加链接。9项公开测试、15项实验检查通过；用暂存区的独立副本只读公开CSV复算，三张新增数表逐字节一致，包含56项配对指标区间。发布文件/双SHA/敏感信息检查通过。

HiFi和CPU收尾等待器未修改，未发布未完成的性能结论。每次授权commit后立即push的长期约定也已同步到公开`AGENTS.md`。

## 不等HiFi：先行研究定位与失败后果复算（2026-09-16）

按用户要求，已完成`reports/research_positioning_one_page_20260916.md`阶段一页定位、`reports/external_related_work_difference_20260916.md`原文差异表，以及`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/research_positioning_interim_001/`的等资源/质量—信道次数/质量—完整RX三类图与数表。只读已有结果、CPU运行，没有新模型推理/训练或holdout。

新复算：同N4204、1dB、m8的300次传输，算术可靠正确率81.67%、raw16.67%，但全传输LPIPS算术0.19949、raw0.18613；raw−算术的源图CI为[−0.021745,−0.005649]。保留“成功率不等于全部输出图像质量”的限定证据；不把不同失败集合的均值当因果，也不改写旧算术quality=BLER的负结果。

建议数字VAR暂为限定需求的工程候选，R3保留高SNR工作点，Swin/Deep保留像素/计算取舍；当前不定稿为新增算法。普通误差韧性熵码只是登记的最小证据缺口，没有新实验队列。不把m9饱和当数字整体上限。

HiFi原worker未动，另挂CPU-only收尾等待器，状态`closure_queue_001/status.json`；完整1500帧/指标/原队列回执齐后才做同观测归因、同N4204比较并输出最终报告和最终一页定位。15项CPU测试通过。ADJSCC四固定样例的同观测SHA严格一致，RGB有约2.4–3.6e−7差异，不能假称bit-exact；按原功能容差作全量核查，详见`reports/hifi_completion_readiness_20260916.md`。

## 外部非扩散结果已同步GitHub（2026-09-16 19:53）

用户授权“都传到仓库里”后，新增代码、报告、25500开发/协议/参考行、18000校准行、完整计时与120张展示图已推送到`daiqizai/var-comm`的`main`，提交`1eb526f3c01c0891244cb46855b0346539a69289`。本地/远端一致、发布工作树干净，未强推。9项公开测试及干净克隆的119汇总/1624配对区间CPU复算通过，最大差0。入口公开`results/external_baselines/README.md`；详情`reports/github_external_update_20260916.md`。

未上传权重、完整数据集、源像素数组、环境或密钥。HiFi只有运行快照，仍按原协议继续，不冒称外部最终定位完成。

## 非扩散外部定位先行完成，HiFi已断点恢复（2026-09-16 19:05）

用户要求先完成其他部分后，已交付`reports/external_baseline_non_diffusion_result_20260916.md`。作者Swin/ADJSCC各4500次传输、原四系统6000参考、新N4204/4498数字18000次校准＋6000次冻结开发、1920对完整TX/RX计时完成。25500协议/参考/开发行汇总为119表行、1624配对区间；等N4498原图拼图、逐SNR/质量—资源/计算曲线在`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/non_diffusion_analysis_001/`。6000新PHY及183固定图像检查PASS。

相同N4498下，主区间raw数字相对Swin RA32的LPIPS−0.126653，但PSNR−5.2319dB、RX约59.64vs5.58ms；是明确取舍，不是全面胜出。R3高SNR低LPIPS参考保留。旧R3/Deep单模型峰值仍有空项，不能从旧共驻值宣称显存优势。

HiFi第52帧提交后为短任务让卡，19:05:20已用原SHA绑定队列恢复；全部提交帧与中断记录保留，没有改权重/采样/数据或启动训练。实时PID以`author_queue_001/status.json`为准。最终外部定位报告仍待HiFi全量与同ADJSCC归因，不把本阶段当所有任务结束。

## 外部作者方法实测已启动（2026-09-16）

五份用户转存作者权重全部SHA核验通过，扩散权重另与官方MD5一致。独立torch1.12.1/cu116环境建立完成；Swin三档与ADJSCC已严格加载，并与作者收发路径达到约1e-7像素一致性。HiFi完整默认235步校准推理通过，固定参数未更新；最初冻结autograd标志引发的自定义checkpoint兼容错误已保留并修复，不当方法负结果。

17:34启动有限作者测量队列，入口`experiments/external-baseline-positioning-20260916/CONTINUE.md`，实时状态`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/author_queue_001/status.json`。Swin/ADJSCC各4500帧，HiFi1500帧，原100 development、五SNR、三原噪声。HiFi10dB单图实测约78.7秒，完整默认设置预计30–36 GPU小时；没有擅自缩步数弱化基线。

四已有系统6000条只补共同SSIM，未重跑原模型。作者免费功率/mask假设与实际计费协议分表；裸ADJSCC没有被强加header。N4204/4498数字适配CPU测试通过，但必要校准、实测、时延补充和最终定位报告仍未完成，不能宣称外部定位已完成。运行输入SHA已绑定，不边跑边改源码；不启动训练或新holdout。

## GitHub发布完成，远端提交已核对（2026-09-16）

用户明确授权整理并推送到`daiqizai/var-comm`。独立发布工作树`publish/var-comm/`已提交`16dd8d7`（main，500文件、约21.8 MB），保留通信源码/配置、研究报告、失败结论、关键数据/图与CPU复算入口，排除大模型、数据像素、vendor、venv和凭据。原始文件原位只读，发布副本仅做机器路径脱敏，并记录双SHA。

5项公开结果测试、8项模式策略测试、314个Python语法检查、全部暂存blob核对及干净Git克隆后的1856配对区间复算均通过。完整Git bundle约6.1 MiB已备份。

用户提醒后找到已有非默认密钥`id_ed25519_github`，已认证为`daiqizai`。只设置本发布仓库的SSH认证命令，未新建密钥或改全局SSH配置。普通push成功，远端main/default HEAD均与本地`16dd8d78989537d80218660c21b032407798e7ad`一致，工作树干净。之前将默认SSH失败概括为需重新配置认证不完整，现已纠正；过程保留在`reports/github_publication_status_20260916.md`。没有新训练或holdout访问。

## 最新完成：三权重收尾，当前固定m7混合主线结束（2026-09-16）

报告`reports/hybrid_weight_closure_convergence_20260916.md`。四新臂各20000新增更新，旧0.01两臂同历史只读复用；全20000数据/SNR/噪声、初态/Adam及最终计数独立核对一致。六点都在校准选20000；最后5k仍改善，封顶不作为彻底收敛证明。新增训练/校准共同会话3.06138 GPU小时。

结构方面，C/0.03在主1/4/7相近LPIPS下PSNR+0.544004（CI正）；机制区间C/0.1对U/0.03在PSNR−0.05333dB时LPIPS−0.008817（CI负），保留条件读取机制。系统方面，最佳混合LPIPS0.217177，仍较raw/算术差+0.033414/+0.034597，未过原联合要求。两个预定区间六点三指标均落后校准固定比例期望参考；其非部署属性、区间越过0.005以及4dB局部PSNR例外均披露，不夸称全面支配。

9000实际工作点行＋9000旧强参考＋9000期望行完成；1856配对区间、1500 PHY及9000图像独立复核PASS，图表/CSV已导出。全部模型、Adam、失败记录与原报告保留。正式队列12:37结束，独立复核12:38完成，图表12:39完成，无后续训练/搜索或新holdout。CSI不确定性只交三篇已发表原文核查与未执行的最小验证草案，具体差异尚不够清楚，不自动定稿或启动训练。

## 本轮历史执行：固定m7混合路线三权重收尾（2026-09-16）

09:44（UTC+8）实查：正式队列3440348/trainer3440380运行正常，四个新臂已各完成700/20000新增更新。两种原接收结构与PHY不变，仅新增lambda0.03/0.1；lambda0.01原两臂完整20000步通过共同初态、Adam、训练机会及配对数据噪声审计后复用，不用旧终点作为新臂起点。固定选模、实测点匹配容差与校准拟合的帧间选择比例规则已在运行前登记。

计划`reports/hybrid_weight_closure_protocol_20260916.md`；输出`outputs/HYBRID-WEIGHT-CLOSURE-20260916/`。已完成复用资格、冻结Deep所缺15000条校准、四臂工程测速及180行质量/分析smoke；这些不计科学效果。自动流程仅训练→校准冻结→原development→数值汇总，不自动扩lambda/结构或启动CSI新题。旧报告与所有失败记录保持不变，最终收敛判断尚待完成。

## 最新完成：基图条件读取证实像素效应，未获得联合收益

2026-09-16 01:24:47注册流程结束，01:27后的独立产物核验亦PASS。两臂各新增20000，共同E/D累计30000，最后完整校准仍改善，因此按预登记预算上限停止而非宣称收敛。报告`reports/hybrid_base_conditioning_result_20260916.md`；图表在`outputs/HYBRID-BASE-CONDITIONING-20260915/analysis_001/`和`report_assets_001/`。

机制7/13/19：条件22.08019dB/0.194206/0.812539，对照21.71641/0.188881/0.816989。PSNR+0.36378、CI[+0.32971,+0.39767]；LPIPS+0.005325、CI明确为正；DINO CI跨0。空间打乱使PSNR−0.97417、LPIPS+0.040414，说明空间条件真实参与。保留有限像素/取舍证据，不把自动未通过LPIPS稳定/三指标状态等同“条件完全没用”。

系统1/4/7：条件20.43010/0.239892/0.769932；raw19.62665/0.183763/0.886743。PSNR+0.80344有CI支持、超过0.5dB点目标，但LPIPS+0.056129远超0.005，联合目标仍失败。条件比本轮控制主PSNR+0.62050而LPIPS+0.007316，不能仅和旧诊断起点比较夸大感知增量。

只加5488参数（每臂总1648125），新增空间卷积28.18M MAC/图；完整条件TX/RX约68.386/61.944ms，与同结构控制同量级，小差异不当加速证据。训练+校准两臂合计约2.271 GPU驻留小时。原资源、loss和视觉/PHY未变；3000正常+1500空间诊断行、960计时、1500 PHY/4500图片指标复核完成。

旧结果/checkpoint/失败记录均保留；无新holdout。原队列和训练/审计GPU任务均结束，没有自动扩展；当前固定条件实现停止追加，保留机制观察而非新系统赢家。

## 最新执行：实际基图中间条件读取，两臂有界训练

2026-09-15新授权与计划`reports/hybrid_base_conditioning_protocol_20260915.md`。保持m7/68+1882+1110/E6120/FEC/视觉/源残差域/loss不变，仅比较连续Decoder是否读取实际基图的空间特征。两臂同1648125参数、同起点功能与共享参数Adam，新小模块5488参数；不重开三通道gain训练。

8项CPU、真实GPU梯度、60正常+30空间打乱工程输出、20条完整计时均通过；基图与旧链像素差0。23:05:12队列2711529/trainer2711561启动（session97872），23:09正在完整初始1k校准，尚无新增更新。回执`reports/hybrid_base_conditioning_launch_20260915.md`。实时看`outputs/HYBRID-BASE-CONDITIONING-20260915/pipeline_001/status.json`及`training_001/status.json`。

新增最少10000、最多20000，任一臂改善都让两臂同机会推进；最后按完整校准选模，DINO不参与停止/选择。机制7/13/19和系统1/4/7分开，不筛失败，不访问新holdout。工程完成不等于机制成立；旧全部负结果/模型/失败只读保留。

## 最新完成：混合源残差固定分配实验及报告复核

2026-09-15 20:30:13注册流水线全部完成：每臂10000更新，126000校准记录、4500新development行与9000强参考、1500基础消融、1440新系统计时。后续CPU完整训练/Adam审计、1500 PHY重放和6000张图像PSNR/LPIPS/DINO独立复算通过。完整报告`reports/hybrid_source_correction_result_20260915.md`；图表`outputs/HYBRID-SOURCE-CORRECTION-20260915/report_assets_001/`。

主1/4/7：可靠度gain 19.69400dB/LPIPS0.236300/DINO0.782826；raw自适应19.62665/0.183763/0.886743。ΔPSNR+0.06735 CI跨0；ΔLPIPS+0.05254、CI[+0.04920,+0.05591]，未达到+0.5dB/LPIPS退化≤0.005的预设目标。三个分支均无合格校准点，10000仅是已登记的违约诊断点，不列作新强基线。

残差链相对缩短基础PSNR+2.198dB/LPIPS−0.02100，但没有追回完整系统差距。同参数可靠度输入未提供实用增量。1dB缩短数字块300/300 CRC失败，原完整raw m7正确246/300；≥7dB新基础全正确后LPIPS仍明显落后，不能仅归因于误码。匹配校准曲线仍小幅改善，不宣称已彻底收敛/混合架构无效。

新TX/RX约69.336/62.681ms，和132.017ms，主要多付TX VAR生成成本；不含空口/排队。当前所有GPU/队列已退出，无自动追加训练/搜索/新holdout。原数据/checkpoint/失败保留，通信研究总体目标仍未完成。

## 最新实际状态：源残差混合通信已进入正式配对训练

19:18更新：每臂1509/10000；首1000 checkpoint三套Adam更新计数一致，固定100校准源/4500行完整，CPU状态与覆盖审计PASS。所有运行源码绑定未改动，继续固定预算；尚未得到完整训练后的科学结论。

2026-09-15 19:07:36正式trainer2436487启动（受限队列2433287，session17285）；19:08:51每臂145/10000更新，GPU占用5326 MiB、利用率94%。21k确定性缓存完成；8项CPU回归、真实图像E/D梯度、90条工程质量及30条无缓存在线回放均通过，后者像素误差≤1.67e−6。启动回执`reports/hybrid_source_correction_launch_20260915.md`。

实时入口`outputs/HYBRID-SOURCE-CORRECTION-20260915/pipeline_001/status.json`，代码SHA锁定的有界队列含后续校准/原100 development/仅新系统计时/配对统计；出错停止，不自动换架构/额度或加训。不访问新holdout。当前仍没有新方案的科学质量结论，下方“训练尚未启动”为此前缓存阶段快照。

## 最新：固定数字基础＋源残差连续支路，工程准备已进入执行

2026-09-15用户新附件及“理解后推进”授权形成一个明确问题：同N3060/E6120下，实际传输基础图未表达的实例残差，能否提升PSNR并满足LPIPS非劣。协议`reports/hybrid_source_correction_protocol_20260915.md`，固定m7/68+1882+1110；一个小连续E/D，硬相加、同参数SNR gain、可靠度gain三个融合臂，每臂10000配对更新，不进行架构/功率/分配搜索。

18:41六项CPU协议/能量/失败候选/信息边界测试通过；4个train/calibration工程样本的无噪声数字往返正确、TX与RX旧VAR图像逐像素相同。18:42启动20k训练＋1k校准的确定性TX基础缓存，PID2407172（session77047），`outputs/HYBRID-SOURCE-CORRECTION-20260915/cache_001/status.json`。此条记录时训练尚未启动，不把缓存/梯度代码准备当成学习增益。

TX不见RX噪声/CRC；RX真实译码失败候选不删除，header失败灰图。已有完整raw/arithmetic数字、固定m7/m8/m9与感知Deep是强对照；缩短基础不能冒充强基线。旧1000 holdout已经用于提出本假设，只作历史外部参考；本轮不访问新holdout。所有旧结果/失败/权重只读保留。

## 最新完成：通信强对照、独立验证及最终收敛结论

2026-09-15 17:42（本机UTC+8）所有数值阶段结束：126000条新holdout VAR候选、84000条四冻结系统参考完成；两项CPU审计和冻结策略统计均PASS。1000张/1000类独立源严格在方法与有序名单冻结后访问，1个历史像素重复按预定规则排除，没有据holdout调整策略。最后报告`reports/communication_convergence_final_20260915.md`，正式图`outputs/COMMUNICATION-CONVERGENCE-20260915/HOLDOUT_FIGURES_001/`。

独立主1/4/7：raw质量/goodput/旧adaptive LPIPS0.179154，arithmetic三规则0.179288；后者−前者+0.000134，95% CI[−0.000262,+0.000542]。新质量策略没有超出最强arithmetic BLER控制：两者在全部七SNR同动作。6dB raw质量比raw goodput降0.022053，但arithmetic普通BLER进一步降0.010240；5dB相对arith goodput的感知改善伴随PSNR−0.3246dB，不能称无代价全指标优势。

高13/19：R3 LPIPS0.105112、PSNR23.5885，优于原VAR的0.133193/20.9821；同WeTok数字LPIPS0.086295更强，Deep PSNR26.330更高。不能说所有学习方法输。新算术链完整处理约135.33ms，raw约72.43ms，成本主要增加在TX；计时只用旧32源、不同session，不含空口/排队。

本轮任务完整结束，不自动加训或再搜索；原PID2221015/2273403/2230734及CPU审计均已终止。保留所有原报告、checkpoint、失败/中断和完整图像。新增的是有边界的通信实验发现，不是已经验证新颖性的独立调度方法；大研究目标仍未完成。

## 当前：独立holdout已启动，受限后续流水线已接续

主矩阵PID2221015（session80794）按冻结方法/名单运行1000源；16:17已204/1000、25704行，只看进度不据部分holdout改方法。受限finisher2230734（session78377）等待该实际进程，再仅运行四冻结参考、两项CPU审计和策略统计，入口`VALIDATION_FINISHER_001/status.json`。不重复队列，不重新训练或搜索。

新整帧熵编码时间2016行完成，实际TX/RX波形图像差均0、参数不变；主质量策略TX/RX约60.28/75.05ms，处理和135.33ms。raw质量策略复用旧各m计时约10.07/62.36ms；不同session/co-residency限制保留。完整开发收敛判断见`reports/communication_development_result_20260915.md`，尚未把holdout当成完成结果。

## 当前：校准/开发完成并冻结；即将按锁定方案独立验证

2026-09-15：1000源×六固定候选×七SNR×三噪声的126000条校准完成；CPU逐条重放全部PHY并核对源codec完整性/PSNR/DINO，100128个完整恢复payload全部正确还原source，审计PASS。100源12600条development也完成并同口径审计；旧raw数字复放三指标最大差≤3.82e−6，原参考未改写。

`POLICIES_001`按校准固定：raw质量规则[7,8,8,9,9,9,9]，BLER规则[7,8,8,8,8,9,9]，goodput[7,8,8,8,9,9,9]；arithmetic质量与BLER均[7,8,8,9,9,9,9]，goodput仅5dB改为m9。SNR顺序1/4/5/6/7/13/19。全局校准选择arithmetic_quality，但它与同族普通BLER策略完全等价，不得称独立质量调度增量。

Development主LPIPS raw_quality/goodput/旧adaptive同为0.183763，arithmetic三规则同为0.182580；6dB raw质量比raw goodput改善0.022872，但arith质量/BLER又优于raw质量0.012850。没有调阈值找优势，仍按原主区间与强对照报告。

15:59:05先封存`FINAL_FREEZE_001/method_core.json`与全部按类排序的source路径计划，再开始像素完整性检查；15:59:09形成1000张/1000类未用holdout清单。历史排除覆盖20k通信训练原图及flip、1k校准和3342个已用val源；预定顺序排除1个重复，未计算holdout模型指标。数据与方法SHA均封存，不可据holdout改方法。

当前新熵编码链2016条CPU端点计时运行中，PID2210452（工具session67630），只测新链、不重测七系统。随后只按`FINAL_FREEZE_001/frozen_method.json`启动1000源VAR候选矩阵和四个冻结系统参考，不训练/新架构/搜索。来源根`outputs/COMMUNICATION-CONVERGENCE-20260915/`；本轮总任务尚未完成。

## 当前：R3收尾完成，整帧熵编码/模式策略校准运行中

2026-09-15新授权见`reports/communication_convergence_execution_20260915.md`。R3原10000选定点已完成2100新development记录、原22参考复用、CPU归档/配对统计；仅自身480条同端点计时也结束，没有重跑七系统benchmark。主R3−R2 LPIPS−0.001811、PSNR−0.125949dB，DINO CI跨0；高13/19改善更多但非所有指标最优。完整阶段报告`reports/r3_completion_result_20260915.md`，不混用校准0.172232。

新整帧算术编码不是旧分组链：94 header＋2966 data，单一连续r1:m码流和一个data FEC块，溢出raw回退计费，CRC失败实际候选仍渲染。完整VAR概率/CDF、GPU无损往返、复用状态补全和失败候选与旧实现图像差均0；8项算术/FEC CPU、5项模式完整性CPU检查通过。

校准协议已固定：raw/arithmetic两族×m7/8/9，1000原独立校准图、固定4101/4102/4103噪声、七SNR，共126000记录。BLER目标10%下最大源率、LPIPS+0.25dB PSNR约束、最大源索引goodput三个规则均在结果前登记，不搜索阈值。主SNR1/4/7、高13/19，5/6过渡单列，DINO不进入新选择。

14:51核实实际校准进程2117550（工具session73155），已107/1000源、13482行，约2.7秒/源。输出`outputs/COMMUNICATION-CONVERGENCE-20260915/CALIBRATION_001/`；当前只运行这项GPU校准，尚未拟合策略或重新评测development。新holdout仅盘点文件路径，未读其像素/指标，需完成开发后冻结方法与清单再执行。不启动训练/新骨干/自动候选扩张。

## 当前完成：冻结系统完整在线代价已补齐

2026-09-15 13:09:46（本机UTC+8），七方案/32图/五SNR/seed2001/三次计时重复共3360行完整结束；每条实际TX波形、RX图像与旧结果逐字节相同，权重起止SHA不变。13:10:18独立CPU重新核对产物、故障保留、均值/P95/配对bootstrap，最大统计差1.42e−14ms。实际正式进程约283.78秒，不等于GPU利用积分。

主1/4/7 dB：数字adaptive完整TX/RX为10.090/62.284ms，R2 residual为29.388/39.535ms，Deep为2.258/3.023ms；adaptive处理和比R2多3.451ms，比Deep多67.092ms。旧数字1—2ms只是PHY，当前成本应与质量一起判断，不再称轻量或实时。计时仅32源一个噪声，不能改写100源三噪声的旧质量结果。

报告`reports/frozen_system_online_timing_result_20260915.md`；全部表格/哈希/三次保留工程失败在`outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/`，正式PNG/PDF在`figures_full/`。21项CPU检查通过；`smoke_004`旧示意图标题范围误用正式模板，数据回执正确，旧图保留且另以真实2图/一次重复重绘至`figures_smoke/`，不得用于系统排名。

原正式进程1994210已结束，无追加队列；本次有限计时授权完成，训练/架构/自动搜索仍暂停，R3只保留待验证资格，研究总目标未完成。

## 当前有限执行：冻结系统完整CPU端点计时

用户于CPU准备后明确授权“开始”。2026-09-15 13:04 `smoke_004`完成70组GPU回放：发送波形和RGB最大差均0、起止模型SHA一致。动态模块归档、错误引用旧未配对WeTok数字归档、单帧batch维度适配的前三次工程失败均保留；未改旧源码、原FEC、噪声或checkpoint。11项新CPU检查和10项旧回归通过。

13:05核实正式进程1994210，已提交2/32源、210/3360条记录；同一GPU仅该任务。数字m7/m8/m9/adaptive、R2 residual、感知Deep及WeTok数字全部冻结，CPU RGB→CPU波形、CPU波形→CPU RGB连续计时。输出`outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/full_001/`；尚无正式汇总结果。新增架构/训练/参数搜索及R3质量流水线继续暂停。

## 当前补充：数字计时输入已CPU核验，未启动GPU测量

2026-09-15 11:38（本机UTC+8），固定32图/五SNR/seed2001的640条旧数字记录与R2的2400条原计时记录完成输入核验，155个输入SHA留存、10项CPU测试通过；全部失败保留，没有新传输或模型推理。报告`reports/digital_online_timing_readiness_20260915.md`，产物`outputs/DIGITAL-ONLINE-TIMING-PREPARATION-20260915/`。

新增明确限制：R2旧RX是GPU波形到GPU RGB，数字`complete_image`返回CPU；R2旧TX为两段计算之和，中间表示转换未计入。不能把旧数字PHY时间或新CPU端点时间直接与它横排。CPU准备已结束，新的同端点完整测量尚未实现/排队；暂停不解除，不扩写新架构或预检。

## 当前：收敛报告完成，数字VAR重列候选主系统

本机UTC+8 2026-09-15：`reports/convergence_report_20260915.md`完成过去24h审查（窗口截止请求时01:28:36），4组系统比较/15模型版本、64事件与22个缺少明确完成时间的旧产物分开记录。原数据只读，补做数字adaptive对固定模式/Deep/已完成学习链的图像级配对统计，没有新训练或推理。adaptive对m8主LPIPS差−0.031164，CI[−0.034927,−0.027676]；但这是开发集策略/条件系统证据，不是已完成新颖性/独立验证。

关键修正：不把有限训练判为架构无效；旧VAR parallel失格warmup回退不入强比较。数字历史receiver_seconds仅PHY译码，不含VAR补全/图像Decoder，完整在线代价是缺口。ARPC/Ada已有大量概念重合，具体单次N/E/失败输出/计费差异不能直接等同新贡献。

R3按既定上限于02:00完成10000训练和校准，属于窗口后补充、不是完成系统评测。当前VAR_COMM没有GPU任务，不干预其他项目，不启动新架构/参数搜索/大网络/复杂机制；报告完成不自动解除暂停。学习方面只留R3一个待验证候选，R2与其余合法checkpoint冻结保留；全部失败/中断记录不删除。

## 当前：按最新用户要求转入收敛报告

本机UTC+8 2026-09-15 01:28：暂停新增架构/自动搜索，保留全部checkpoint与失败记录；先审查过去24小时完成实验和数字VAR相关工作。自适应数字VAR重新作为候选主系统，不再把学习模型胜过它当唯一目标。现有R3到6700/10000，仅按预登记上限继续和保留审计，不追加新训练/搜索/新GPU队列。窗口与边界见`reports/convergence_scope_20260915.md`；当前不把尚未完成的R3系统评测列为已完成。

## 当前：R3首1000完整审计PASS，真实续10000

本机UTC+8 2026-09-15：新q向量控制保持相同2,933,775参数/三次full-grid/两次E/残差能量门控，真实原7000父点及全部10000配对批次已核验。CPU初始函数与GPU波形/RGB差0，图像损失确实回传E/R；4项CPU和实际driver断点测试PASS。00:35首1000全校准/模型Adam/数据/功率/选模审计PASS，同1000校准prediction LPIPS0.198349、residual0.202011，尚非最终结论。

00:38从实际1000模型/Adam续到10000，1099666/1099667/1099668运行中；00:46读取实际2000、global9000，原1000完整历史和已绑定源码/校准未改。报告`reports/wetok_r3_activation_20260915.md`，入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`。23方法48300质量行与16模型2560独立计时的协议已登记，**评测驱动和自动收尾仍需实现，尚未排队**。不访问新holdout，不称全网格为next-scale。

## 当前：R2完整结果封存，R3向量归因登记

2026-09-14 23:30，全部46200质量行、48400图像/8800特征CPU审计、3744主/270支持点区间与2400独立计时/70区间完成并复核。MS−普通full-grid主LPIPS+0.002350、95% CI[+0.000808,+0.003969]，DINO更低，无明确速度优势；停止当前固定4/8/16的机械续训，不否定所有层级通信。

残差版主LPIPS0.220543，较普通full-grid0.230345改善，但仍差于数字自适应0.183763/感知Deep0.205291；相对同WeTok数字的高SNRLPIPS仍差。它另增6417参数、两次内部E与1.68758ms RX成本，不能直接宣称残差独立或next-scale贡献。完整报告`reports/wetok_r2_result_20260914.md`。所有旧进程已退出，19项CPU测试PASS。

新入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`：仅登记同参数/回算次数/完整训练历史的q vs y−q向量对照，门控保留同一标量残差能量。尚未实现或启动R3，不从R2最优残差点切换短训，不修改旧权重或哈希。研究仍缺强系统优势、充分机制归因、训练重复与新holdout，不标记完成。

## 当前：R2全部10000训练完成，进入22方法质量评测

2026-09-14 23:09:03完整审计PASS，实际四模型/Adam10000、global17000，原数据/损失/资源不变。single/MS完整校准最优仍是7500；ordinary full-grid和innovation选10000，LPIPS分别0.190455/0.174380。不能将相同10000机会误写为全部使用10000权重。当前quality983552、finisher880742、observer880743，23:12真实提交23/100源；原880740/880741已正常退出，不重启。`reports/wetok_r2_milestone10000_20260914.md`。19项CPU测试PASS，当前无完整R2 development/新holdout结论。

2026-09-14 22:09：四臂R2至8400总更新，880740/880741/880742/880743继续正常运行。7500封存端点已生成完整校准、逐SNR、训练分项和成本四类PNG/PDF及CSV；15项CPU测试PASS，绑定训练/评测源码不变。成本对账1.414581h分项＋0.010964h未细分开销＝1.425545h进程时间，不是GPU利用率积分。逐SNR也保留1 dB部分PSNR回退与MS19 dB微小LPIPS回退，不称全面改善。`reports/wetok_r2_training_curves_20260914.md`。

## 当前：R2的7500审计PASS，四臂实际继续10000

2026-09-14 21:35完成全部1000图校准与独立CPU审计：单次/多尺度state/普通full-grid/innovation的LPIPS为0.192001/0.192695/0.193974/0.176768，分别较5000改善0.005124/0.001984/0.000994/0.007951；是五训练SNR的校准均值，不与development主区间混排。实际模型/Adam7500、global14500、新2500数据/噪声配对、功率与完整校准选模全部通过。

21:43由实际7500模型/Adam继续到10000，trainer **880740** / reviewer **880741**，finisher **880742**与被动observer **880743**已实际运行，21:46保存7600总更新。质量与独立计时实现完成，11项CPU/真实driver断点测试通过；后续22方法46200主行、旧18参考不变，32图15模型2400独立计时行。共享只用于显式质量准入，计时遇竞争等候。报告`reports/wetok_r2_milestone7500_20260914.md`，入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。10000/质量尚未完成，不把排队当成结果；下方769675等均为历史。

## 当前：四臂R2真实续训已启动

2026-09-14 20:09从四个实际5000模型/Adam启动至7500，trainer **769675** / reviewer **785157** / observer **785158**。CPU逐值恢复与真实driver断点测试通过；20:10保存5001，20:26读取实际5500保存点确认四臂Adam5500、global12500、前5000历史及来源不变。20:32已到5800总更新。没有重跑起点全校准、重置Adam或更改loss/lr/N/E。`reports/wetok_r2_activation_20260914.md`；入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。当前没有R2完整校准或development结论，下方“尚未启动”是历史。

## 当前：Grid完整结果已结束，准备匹配充分性续训

2026-09-14 Grid5000的37800质量主行、39600图像行CPU复核和1760条独立计时全部完成。4/8/16 state对普通full-grid state的LPIPS差−0.000230、95% CI[−0.001220,+0.000759]，DINO更低，时延也无可靠增量；不能将原对single的改善直接称为next-scale优势。最佳学习链Joint full-grid innovation LPIPS .231342，仍弱于Deep .205291/数字自适应 .183763。报告`reports/wetok_joint_grid_result_20260914.md`。

下一步不加模块，准备single、multiscale state、普通full-grid state、full-grid innovation四臂从实际5000模型/Adam共同继续至10000，保留相同data/loss/lr/N/E以检查充分训练和适配。四端点CPU核验PASS（同global12000、同末批次/噪声/冻结视觉及Adam5000），**新训练尚未实现/启动，当前无活动训练队列**。入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。以下训练/评测PID均为历史，不能按旧等待状态重启。

## 当前：Joint网格控制已实际训练

2026-09-14 18:29 Grid2500完整校准/审计PASS，普通full-grid .198209、Joint innovation .194304；均选2500，仍只是校准。两臂已从实际末端续5000，trainer650391/reviewer650392，19:05核实4600。新18方法质量/统计与11模型独立计时链路通过18项CPU测试，原100源图像素和33600参考行核验通过；finisher689944/observer689945已排队自动收尾，不能重复启动。共享只用于新质量阶段，速度另行独占测量。`experiments/wetok-joint-grid-controls-r1/docs/evaluation_preparation_20260914.md`。

2026-09-14 17:55首1000全校准/审计PASS：普通full-grid state LPIPS .200014，Joint innovation .202011；同1000机会R-only full-grid .199371，尚无强结论。两臂已从真实1000模型/Adam共同续2500，trainer **612129** / reviewer **612130** / observer **612131**；旧首1000进程均结束。报告`reports/wetok_joint_grid_milestone1000_20260914.md`，设置和配对规则不变。

2026-09-14原Joint5000于16:48完整结束：主1/4/7 dB条件版LPIPS 0.236337，对Joint single差−0.002615、95% CI[−0.003608,−0.001680]，对R-only full-grid差−0.000295且CI跨0；仍落后数字自适应/感知Deep的LPIPS。全部33600主行、配对统计与CPU波形/图像复核完成，原49张回执保留。报告`reports/wetok_joint_result_20260914.md`。旧恢复器/评测/观察器均结束，不再重启。

17:23新两臂真实GPU零更新profile PASS，6项CPU/断点测试通过。17:27启动普通16/16/16 state及既有full-grid innovation的Joint控制，先1000更新；trainer **570837**、reviewer **579794**、观察器 **585355**。实际读取300步保存点核实E/R、Adam和原配对数据，非空进程。当前入口`experiments/wetok-joint-grid-controls-r1/docs/CONTINUE.md`。旧三Joint不重训，不把普通迭代称为next-scale；新开发结果尚未产生。

## 当前：Joint E/R同起点控制已激活

16:25最新核实：第一次恢复已从49张推进至**55/100**。16:19新evaluator480961因另一GPU任务482008再次触发原保护退出，恢复器473779/观察器473780仍在等空闲，未重启训练、未改校验规则。原49张回执及冻结源码复查不变；当前不根据部分样本给最终结论。

2026-09-14 16:01，评测在第50张源图末端发现其它GPU进程455090，按原保护退出；前49张/16464主行已提交且保留，训练5000和校准结果不受影响。16:12恢复监督器**473779**与观察器**473780**启动，资格和31项CPU测试通过，等待GPU空闲后自动用原`--resume`继续，再做统计。未停止其它任务、未改原代码/hash、未重启训练。入口`experiments/wetok-joint-sender-r1/docs/evaluation_recovery_20260914.md`。**旧evaluator443167/finisher348503已退出，下方运行中条目为历史。**

2026-09-14 15:50，三臂5000训练/完整校准/审计PASS，均选中5000。校准LPIPS single .197125、no-history .197853、state .194679；条件版开始优于两个基础版，但仍不及R-only full-grid .187311，不能据校准宣称方法成功。原trainer/reviewer已结束，evaluator **443167**、finisher **348503**、观察器 **357923**正在运行，首张336行已保存。报告`reports/wetok_joint_milestone5000_20260914.md`，配对学习曲线`reports/wetok_joint_calibration_figures_20260914.md`。**下方训练中状态均为历史，不重启原trainer。**

15:10已实际核实3600保存点，3500监控完成；state监控LPIPS有波动，不据子集改配方或选模，仍等5000完整校准与原自动评测。三主进程与只读观察器存活；当前GPU仅有本训练，已观测到的早前并发记录不抹去。

2026-09-14 15:00核实保存3100/5000，trainer/reviewer/finisher仍为348501/348502/348503。运行期间其它GPU任务加入，新增只读观察器357923记录并发与硬件背景；不停止其它任务、不改原训练。最终时延若受并发影响，不作独占速度结论。`experiments/wetok-joint-sender-r1/docs/passive_timing_context_20260914.md`。新增5项观察器测试后，共27项CPU测试通过，原训练与收尾绑定源码均未变。

等待训练期间完成HJSCC作者8个小型源文件的只读协议核查，未下载权重或运行新GPU模型。已定位变长CBR、布局/归一化尺度传递、示例固定10 dB与RGB导出要求；不把未适配的示例冒充3060-use强对照，也不据此删除HJSCC。报告`reports/hjscc_source_protocol_review_20260914.md`。这些是强对照接入准备，不是新性能结果或已完成复现。

2026-09-14 14:32完成Joint2500全校准及审计PASS。选中LPIPS single .197908、no-history .200233（仍选0）、state .198812；相对同机会R-only分别−.001012/0/−.002979，但Joint state仍不及single，不能称尺度结构成立。14:37从实际2500模型/Adam共同续5000，trainer **348501** / reviewer **348502** / finisher **348503**，自动评测/统计已排队，勿再手动启动重复任务。曾检测到其它GPU任务，待空闲才启动，未抢占。报告`reports/wetok_joint_milestone2500_20260914.md`；原13:32及更早条目为历史。

新增自动收尾工具与6项CPU测试，连同原16项共22项通过，训练/评测数学和旧绑定SHA不变。5000完成后保留三Joint、全部六R-only及七系统参考，共33600主行＋600支持点＋900无噪诊断；当前没有新development或holdout结果。实时入口`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`。

2026-09-14 13:32 Joint1000全校准/审计PASS：single/no-history仍选中0，state选中1000，尚无明确整体收益。三臂从实际末端模型/Adam共同续至2500，trainer **274550** / reviewer **274551**，13:43保存1100。`reports/wetok_joint_milestone1000_20260914.md`。旧188608/188609已退出；一次GPU占用拦截后重新核实空闲才启动，未抢占别人。
原RX资料已新增逐SNR/配对图和固定样本原图，Deep离支持点异常单列，72图块验证内容不变。入口`reports/wetok_receiver_figures_20260914.md`。这些不是新推理结果。

2026-09-14 12:51已完成三臂起点完整校准并实际保存Joint step1；复读断点确认三臂E都已更新、Adam/训练历史位置一致，源码绑定通过。仍在首1000里程碑训练，不据首个更新宣布质量改善。

2026-09-14原RX-only六臂5000已完成全部训练、27300主行＋600支持点回放与CPU复核。主区间multiscale innovation .242830未胜single .242495，full-grid .236632有局部改善但仍不及数字自适应/感知Deep的LPIPS；高SNR可胜旧骨干参照，仍弱于同WeTok数字链。停止继续扩展该固定E多尺度候选，保留全部强对照。`reports/wetok_innovation_result_20260914.md`。
12:39从与R-only相同原父点激活三结构Joint E/R，trainer **188608** / reviewer **188609**，先step0完整校准再1000更新。只让通信E也优化，loss/结构/数据噪声/新更新机会/N/E不变；原控制与reference资格已冻结。当前入口`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`。未有新Joint性能结论，未访问新holdout。

## 当前：固定共同发送器的接收创新量实验

2026-09-14 11:10恢复训练已保存4100/5000，三个新进程均正常。等待5000及其自动评测/分析，Joint未启动。后续Joint严格评测代码已补齐并通过16项CPU检查，保留全部六个冻结接收器与强系统，不声称所有方法同y；`experiments/wetok-joint-sender-r1/docs/evaluation_preparation_20260914.md`。现有冻结绑定仍通过。

2026-09-14 10:15实际核查：旧三个进程已消失，未完成5000或评测；本机10:10重新启动。已完整核验3000保存点（六模型/Adam/数据噪声/冻结E与源码），10:22恢复原5000流程，trainer **25991** / reviewer **25992** / finisher **25993**，10:26保存3100。不要将旧WAIT状态当成仍在跑。`experiments/wetok-innovation-r1/docs/recovery_20260914.md`，新日志和原现场在`outputs/WETOK-INNOVATION-R1-RECOVERY-20260914/`。未改实验设置或访问新test。

07:28完成2500与全校准审计：多尺度innovation .200449，优于state .201791/prediction .201300，但仍不及single .198920和full-grid .196310。不能称next-scale成立。六臂已从真实2500末端共同继续5000，trainer **3205198** / reviewer **3205199**，07:44保存2700。收尾进程 **3211225** 已排队自动做5000评测/统计/绘图，不再手动启动重复评测。报告`reports/wetok_innovation_milestone2500_20260913.md`。
Joint准备于07:32真实GPU零更新检查PASS，图像梯度到E/R、权重不变、分项梯度/成本已记录；**没有启动Joint训练**，先等当前完整控制及自动评测结果。训练、profile和排队评测源码已绑定SHA，不边跑边改。

新增只读训练审计：7000父点共28000次曝光，但图像阶段20000次曝光只覆盖15197张不同训练图，4803张仅接受过表示监督；R-only续训不更新E。single/no-history末段校准仍改善，conditioned近乎持平，不能统一宣称充分收敛。强Deep为30751643个通信参数、40000次ImageNet微调曝光，另有COCO历史，仍保留为强系统参照而非等参数控制。`reports/wetok_training_sufficiency_audit_20260913.md`。
据此准备`experiments/wetok-joint-sender-r1/`的同起点、同数据/更新机会Joint与R-only控制；7项CPU测试和真实初始化权重核验PASS，新的GPU训练未启动，不抢当前六臂任务。RX图像评测已补去no-history不影响图像的辅助读取，并逐帧核对原输出；17项CPU检查通过，当前训练/校准图没有改变。

06:00首1000更新、完整校准和共享y/固定E/Adam/选模审计PASS。选中校准LPIPS single .198920、no-history .200233（选中0）、state .201941、prediction .201300、innovation .202062、full-grid .199371；创新量尚未体现图像增益，不把初始化偏移修复当成功。已从实际1000末端模型/Adam共同续至2500，trainer **3136686** / watcher **3136687**，原配置与物理预算不变。报告`reports/wetok_innovation_milestone1000_20260913.md`；完整校准/监控/训练曲线已生成。旧3072510/3077070已正常结束。

geometry完整开发评测已完成：新204×30显著改善旧映射，但新geometry条件对无历史LPIPS +.003614、CI[+.002575,+.004700]，DINO也更差；仍未胜强系统LPIPS。不能把共同geometry收益当next-scale成功。完整解读`reports/wetok_geometry_result_review.md`。
基于训练源波形解释误差诊断及ISEC先例核查，新实验只改RX：冻结同一已选中的single发送器和视觉模块，六臂读取逐样本相同y，设置预测特征与全网格创新量控制。CPU/真实GPU无更新检查及首1000训练均已完成，尚无新机制系统成功结论。
入口`experiments/wetok-innovation-r1/docs/CONTINUE.md`，状态`outputs/WETOK-INNOVATION-R1-TRAINING/status.json`。重编码/迭代本身不是原创，冻结E保留输入梯度但不更新其参数。最终27300主行+600支持点评测、配对统计与实际波形/图像CPU核验执行器已补齐；15项CPU测试及441条旧图/21父点信号只读检查通过，未访问新模型development或新holdout。GPU负载约94%并出现间歇温控降频，不提高负载、不改硬件设置；新时延评测增加次序平衡和硬件状态记录。

## Geometry总4500完成，当前推进匹配完整预算总7000

image2500完整校准/审计PASS：204×30三结构LPIPS .203079/.204214/.203916；同预算153×40为.257791/.254603/.256377。改善首先是geometry共同效果，当前条件版未稳定优于同geometry的其它结构；不与不同数据/SNR的development强基线混排。
已从实际4500末端模型/Adam共同续训至总7000=image5000，trainer **2973427** / watcher **2974929**；设置不变，结束后做已准备的21000主行+600支持点匹配开发评测与交互/成本分析。`experiments/wetok-comm-v2-20260912/docs/geometry_continue_7000.md`。

## Geometry同预算校准改善，继续总4500

总3000（表示2000+image1000）已完整校准及审计PASS；204×30的新LPIPS single/no-history/conditioned为.206725/.206953/.207850，对同预算153×40为.259012/.261040/.256377，同时PSNR提高约1.14–1.34dB。这是calibration，不能与另一批development强基线混排，也尚未证明新geometry的条件增量。
已从实际总3000模型/Adam共同继续至总4500，trainer **2941742** / watcher **2943268**，配置和N/E不变。决定`experiments/wetok-comm-v2-20260912/docs/geometry_continue_4500.md`。
新geometry最终评测/统计入口已实现：两组共六模型同场重测、四强参考、21000主行+600支持点、1008配对+48交互区间。34项CPU/合成整链测试及441条旧图PSNR/配对复核通过，正式新模型development GPU评测仍未执行。

## 本机UTC+8 2026-09-13：接口5000完成，转入受控geometry训练

原接口5000开发评测完成：continuous条件主1/4/7 LPIPS .279289，对无历史差-.007812、95% CI[-.009552,-.006112]；PSNR/DINO配对也支持改善，但仍输数字自适应.183763和感知Deep.205291，noiseless约.190仍远离native约.0868。不是系统成功；报告`reports/wetok_interface_milestone5000_2026-09-13.md`。
新204×30三结构已在00:47启动，trainer **2902777** / watcher **2907659**，目标先总3000(表示2000+image1000)，再按证据到同控制image5000。当前已完成2000表示并fresh Adam重置，正在image0全校准。保持6120实坐标、连续接口、视觉/loss/SNR不变。
153×40原控制的真实初始化、全部7000步数据/噪声、阶段优化器与选择机会审计通过，不重复训练它们。初始化2线程曾因QR数值差被严格拒绝，已恢复历史8线程而完全匹配，没有改SHA或阈值。31项CPU检查及真实GPU梯度/功率profile通过，新geometry没有开发集结论。
新入口`experiments/wetok-comm-v2-20260912/docs/geometry_execution.md`；当前状态`outputs/WETOK-GEOMETRY-20260913-TRAINING/status.json`。独立holdout和训练重复仍未完成，研究目标active。

## 2026-09-12：v2持续研发已激活，WeTok A0/A1三臂正式训练

21:36九臂仍向5000运行，已保存3100；当前训练代码与资源不变。只准备了一个后续发送geometry候选153×40→204×30（同6120实坐标），旧控制参数/前向完全一致、新行正交构造及CPU图像梯度检查通过，共30项测试。候选参数2927358对控制2895944、计算也不同，尚未GPU profile/训练或测图像收益，不因准备好就自动切换。详见`experiments/wetok-comm-v2-20260912/docs/geometry_candidate_preparation.md`。

21:12向5000续训的实际2700断点审计PASS（global data4700、Adam2700，存储独立/采样配对），进程继续。2500结果另有13/19dB条件相对无历史的局部LPIPS改善-.00313/-.00564，但PSNR/DINO区间跨0且仍未赢强系统；不把主区间负结果说成所有SNR无效，也不据此改主评价区间。

**20:57已进入总追加5000的共同续训，2500开发结果为接口改善但非结构/系统成功。** 新trainer **2730582**、watcher **2733118**，从实际2500模型/Adam继续原配方和账本；未改网络/packing/功率或增加数据。决定`experiments/wetok-comm-v2-20260912/docs/interface_continuation_5000.md`。
20:47完成33600主无线行、600 Deep支持点及3024+162配对区间。主1/4/7 continuous条件LPIPS .296130、无历史 .290503，差+.005627、95% CI[+.003947,+.007379]，条件版更差；无历史也仍明显输数字自适应.183763和感知Deep.205291。未把接口收益当next-scale胜利。报告`reports/wetok_interface_milestone2500_2026-09-12.md`。
独立统计复算3024区间最大误差<4e-15，12600个选中hard与原父点图像hash比较一致；这是统计/文件审计，不是第二次神经推理。600支持点GPU重放与旧图像素差0，未用5/6异常制造优势。新holdout仍未访问。

19:21的2000固定监控重新改善：continuous三结构LPIPS .256390/.254504/.253020，较1000同源分别-.001166/-.004966/-.002473。此前1500回退不能视为持续恶化的定论；保持九臂到2500完整校准，不改配方、不从监控选模。`experiments/wetok-comm-v2-20260912/docs/interface_monitor2000_review.md`。

19:00补充CPU训练梯度监测：1600点、4训练源×3个continuous，MSE/LPIPS局部2/12反向，但注册总梯度对LPIPS均正向；范数比不构成回退因果证明，Adam方向亦未据此假定。未更新参数/占GPU/访问新test，继续原2500检查点。结果`experiments/wetok-comm-v2-20260912/docs/objective_probe_result_1600.md`。

**18:49：1500固定监控的continuous LPIPS回退，不能宣称训练继续单调改善。** same-source的1000→1500变化为single +.003228、无历史+.008029、条件+.005612；当前值.260784/.267499/.261105。既有最佳不变，仍保持九臂配置等机会至2500全校准；若非有限按既有安全规则停并保留现场。`experiments/wetok-comm-v2-20260912/docs/interface_monitor1500_review.md`。
CPU整链评测/源图恢复测试已加入，修复续跑后的预热计时，并累计session耗时、检查外来GPU竞争；27项测试通过，训练源码/参数和物理账本未改。新模型development及新holdout仍未执行。

18:24续训恢复审计PASS：九臂共同保存追加1100（global data3100、Adam1100），模型/优化器存储独立、数据/增强/SNR/噪声完全配对；当前仍向2500目标推进。见`experiments/wetok-comm-v2-20260912/docs/interface_resumed_checkpoint_audit.json`。

**18:16：1000完整检查点已结束并通过审计，九臂继续到总追加2500。** 三个continuous全1000图/五SNR校准LPIPS为0.259012/0.261040/0.256377，选择追加1000；六个hard仍选择合法追加0，不能当作新条件结构成功。条件continuous对无历史的校准差-0.004663，不是独立development结论。
从实际1000末端模型与完整Adam继续，不从最佳点回退。新trainer **2614845**、watcher **2616385**；保持LR/loss/接口/网络/N/E/数据与噪声，追加1500到2500，1500/2000监控、2500全校准。决定`experiments/wetok-comm-v2-20260912/docs/interface_continuation_2500.md`。
最终评测补上600条固定支持Deep5/6单列：原始曲线不改，NN条件4/7而物理SNR5/6。600旧图PSNR/预算检查、4次实际CPU模型重放通过，最大像素差3.40e-6；25项CPU测试通过，GPU新模型评测仍未执行。另完成16训练源的只读TX占用诊断，记录线性skip/实际发射信号方向问题，但未据此改packing或宣称容量界。

**17:09首500追加步监控出现连续接口改善信号，训练继续。** 固定100 calibration图×五SNR，continuous三结构LPIPS分别0.257093/0.257975/0.257031；native有界-ST约0.301–0.306，identity约0.330–0.335。原native同源起点约0.287–0.288；不把这批监控与development强基线混排。
条件continuous对无历史差仅约-0.000944，对single-pass约-0.000062，尚未确认next-scale独立增量。保持九臂等机会，继续当前1000检查点的完整校准，不按monitor选模或改loss。细节`experiments/wetok-comm-v2-20260912/docs/interface_monitor500_review.md`；23项CPU测试及441条旧参考复核通过，新模型最终development评测仍未执行。

**16:44接口九臂已完成并审计共同100追加更新。** global data step2100、fresh Adam逐参数step100，九臂模型/Adam状态存储独立、数据/增强/SNR/噪声完全配对，最大能量误差5.96e-7；原始训练/profile源码未变化。训练及watcher真实存活，不根据100步loss宣称有效。
新development执行器`evaluate_interfaces.py`和统计`analyze_interfaces.py`已补齐：continuous保持连续解码，16方法/33600行、63组/3024个配对区间、PSNR/LPIPS/DINO图表；旧7方法原图只读引用。CPU完整合成网格与441条真实旧对照复核通过，尚未执行新模型development评测，不能把准备完成当方法成功。详见`experiments/wetok-comm-v2-20260912/docs/interface_execution.md`。

**16:06起接口九臂流程实际运行。** 三结构×hard_identity/hard_bounded/continuous_mean；19项CPU测试和真实九臂零更新profile通过。两个hard前向像素相同、train/eval一致、image梯度到E/D、视觉权重冻结；此时只是工程通过，不是模型收益。
trainer **2518473**、校准监督器 **2518976**，实际日志`interface_train_0001000_launch2.log`。首1000流程从完整起点校准开始，profile估计九臂更新主体0.944小时，完整校准另计；首1000只是检查点，初步目标5000再按走势调整。
实时`outputs/WETOK-COMM-INTERFACE-20260912-TRAINING/status.json`；详细恢复和下一步`experiments/wetok-comm-v2-20260912/docs/interface_execution.md`。当前不访问development，首里程碑自动完成CPU配对/选模审计和calibration曲线。最终新接口development评测入口仍需实现，不冒用旧hard-only输出。

**15:52更新：bit权重六臂已探索性提前停止，不是完成5000。** 固定100张calibration monitor在追加1000时，原joint/bit权重1的LPIPS为single_pass 0.68509/0.83198、无历史0.72811/0.90610、条件版0.53653/0.76036，均继续恶化。保留六臂共同global6000的完整模型/Adam；SIGINT时已执行至共同1113追加步，未提交部分不纳入比较。没有新development结论，不声称所有bit权重无效。
自有trainer与监督器均已退出，不触碰其它任务。回执`outputs/WETOK-COMM-BIT-SUPPORT-20260912-TRAINING/exploratory_early_stop/termination.json`；决定`experiments/wetok-comm-v2-20260912/docs/bit_support_early_stop.md`。原监督器中断状态保留，不伪造完成。
接续准备`configs/interface_study.yaml`：从真实2000模型出发、各臂统一fresh Adam，三结构×native identity-ST/native有界均值-ST/真正连续接口，冻结视觉/网络/物理预算/原joint权重。下一步先做真实接口与梯度profile再配对训练；尚无新接口收益结果。

以下bit权重条目是此前启动记录，已由上述停止决定取代：

**bit监督修订对照已启动。** 2026-09-12 15:06起，三种结构×原joint/bit-BCE权重1，共六臂从同一5000模型/Adam续训；父点SHA已锁定，optimizer深拷贝，唯一变化为bit权重0.01→1。首个追加5000更新仍是研究里程碑。
已验证六臂各250追加更新、global/Adam step5250；各模式内模型/Adam初值相同，所有batch/SNR/噪声指纹一致，源代码未漂移。训练PID2473858；评测/分析监督器PID2483685。
新结果将同时比较同预算原joint、原实验2000最佳、同配方条件/无历史、以及强数字与Deep参考。16项CPU测试及378条历史最佳/旧对照CPU回放通过；尚无本轮质量增益结论。
实时状态`outputs/WETOK-COMM-BIT-SUPPORT-20260912-TRAINING/status.json`，日志`outputs/WETOK-COMM-V2-20260912-LOGS/bit_support_*`；代码入口`experiments/wetok-comm-v2-20260912/scripts/train_bit_support.py`。

**首5000里程碑已完成，研究继续。** 三臂完整校准均选择2000步，joint阶段到5000反而恶化LPIPS/bit恢复。主1/4/7 dB条件版LPIPS0.328912，对无历史差+0.000860、95% CI跨0，未证实条件增量，也未赢旧强系统。
noiseless仍约10.1%bit错误/LPIPS0.210，而同图native参考0.08681；19dB噪声只再增加很少损失。按v2优先修订通信恢复目标而不是先改功率。
下一项已登记`configs/bit_support.yaml`：从共同5000模型与Adam起点，仅改joint bit BCE权重0.01→1，原配方/修订配方在三结构内等新增训练配对，旧2000最佳另保留。详见`reports/wetok_a0a1_milestone5000_2026-09-12.md`；修订尚未产出结果，不将首里程碑当研究完成。

### 初始启动记录

用户指定根目录`next_scale_communication_research_brief_v2.md`为持续执行任务书（SHA 27648fbc73485deb74a0695f3f7840bd01c561c1b0b4f6f67284fefcfb6d2e3f）。旧研发时间/金额/轮数上限取消，但只使用已授权本机GPU，不租新服务、不动其他用户任务，通信公平与数据隔离不变。
新实验`experiments/wetok-comm-v2-20260912/`。选定WeTok是确定性CNN Encoder/Decoder，原生32×16×16±1量化特征、4组8-bit；不套VAR的4096类CE、不另加生成噪声，也不把分组当语义尺度。
三臂single_pass／multiscale_no_history／multiscale_conditioned，每臂2895944参数、相同初值和配对训练。全部读取完整y，不称逐包渐进；中间4/8状态是连续通信估计，最终16网格为native硬符号。新增模型不需要类别/动态mode，0 header+3060 data，总能量6120；与旧68+2992分账同N/E比较。
20000训练图×两种真实翻转、1000校准及100 development的原生组索引缓存完成；100图native重建回放像素差0；校准/development源像素精确去重检查无重叠。实际梯度通过冻结Decoder到E/D，全部冻结权重不变，表示阶段不调用Decoder。
2026-09-12 13:00启动三臂训练，PID2380048；已完成2000表示恢复并进入图像联合训练。首个5000步只是评估里程碑，后续按固定校准走势继续/调度，不自动因达到上限停研究。
8PSK+soft-bit FEC对应数字参考已在CPU生成：8192payload+CRC16+tail6、9180coded bits=3060 8PSK符号，恒定每复符号能量2，CRC失败候选仍渲染；这不是最优数字系统宣称。
首里程碑自动评测/统计监督器PID2400192。结果将比较同WeTok基线/无历史控制、WeTok数字、原数字固定m8/自适应与感知Deep，统一PSNR/SSIM/LPIPS/DINO与严重失真事件；不访问新holdout，不采用不合格warmup fallback。
实验协议`experiments/wetok-comm-v2-20260912/docs/protocol.md`；实时训练`outputs/WETOK-COMM-A0A1-20260912-TRAINING/status.json`，日志`outputs/WETOK-COMM-V2-20260912-LOGS/`。当前尚无最终通信增益结论。

## 2026-09-12：有界骨干选型已完成，暂不正式迁移到XQ；WeTok完整codec参照为正面结果

旧2×2流水线于01:43:30自然完成；自动队列确认空闲后，选型评估01:46:16启动、01:49:15完成（均UTC+8），GPU已释放，新增训练0步。
同100张development，四模型1800正式逐图记录全部完成：旧官方full PSNR22.8359/LPIPS0.10575；fidelity23.8304/0.12089；XQ配套full22.1412/0.13194@6864rawbits；WeTok23.7393/0.08681@8192rawbits。
WeTok相对旧官方8160bits，源表示仅多32bits，PSNR+0.9034dB（95%CI[0.7915,1.0171]，95/100图提升）、LPIPS-17.91%（100/100图提升）。相对fidelity，PSNR均值低0.0911dB、CI跨零，LPIPS改善28.19%；不是重复更平滑工作点。但WeTok仅测完整codec，未证明next-scale支持。
XQ直接prefix有真实收益，但补全没有形成支持迁移的证据：2424bits补全19.7391/0.20490，对旧3060bits补全19.6916/0.17714，PSNR近似但LPIPS恶化15.67%；3960bits补全20.9408/0.16594，对旧5088bits21.2437/0.13590也有感知退化。XQ3960bits优于旧3060bits，但多29.41%源bits，非等预算证据。
XQ2424bits补全相对自身直接解码PSNR反降0.2783dB；3960bits仅增0.0792dB。作者三seed采样未扭转结果。不同自然预算不能当完全匹配，也不能据此断言XQ所有码率都差；本轮结论是**不足以支持付出正式迁移成本**。
独立CPU产物/统计复核PASS：1836个含smoke输出tensor哈希、67份源码/config、7模型资产一致，20组条件统计/80组配对指标及10000次按图bootstrap复算差0；没有第二遍GPU推理/指标网络重算。XQ配套tokenizer与独立checkpoint EMA552/552张量完全相同，非配错权重。
旧基线重算与历史非bit-exact（包括旧p9直接个别图1.6552dB差异），未受控归因；本轮选型用本次统一重算基线，不混历史均值。DINO暴露、类别side信息、无信道/非3060complex uses边界保持。
**决策：本轮不正式迁移到这对XQ；WeTok仅保留高保真完整codec参照；尚未选出满足全部要求的next-scale替代。不是停止VAR通信研究，不自动续训或增加其他模型。**
复核报告：`reports/backbone_selection_review_2026-09-12.md`；自动结果、原始float、图表及审计：`outputs/ei-liulu-xqvar-eval-20260912-v1/`。

## 历史排队记录：2026-09-12凌晨选型准备（下述为当时状态）

用户先指定公司开发机，SSH两次认证前TCP超时，未操作远端文件/进程；随后明确有认证，改为本机既有GPU任务结束后运行。
未向任何GPU任务发送停止信号；原2×2仍按原计划自然运行，不在train/evaluate间隙抢卡。之后不自动新增原版VAR训练/模块设计。
新隔离入口 `experiments/backbone-eval-20260912/`，协议 `docs/protocol.md`，旧项目/checkpoint/基线保持只读。
已核对官方XQ源码配置与配对权重清单；双PQ的自然prefix8/9/full静态raw为2424/3960/6864bits（必须进一步实际索引核验）；WeTok四256码本、16×16对应8192bits，指定公开checkpoint已定位。
新候选权重单流限速下载、私有venv不改变当前训练环境。原100图清单及官方/fidelity模型定位完成；全重建四臂，XQ两个自然prefix直接/argmax及预定三seed作者采样敏感性，无训练/信道试验。
CPU-only队列安全测试41项通过。01:02:14启动监视器PID1856483，命名 `ei-liulu-xqvar-eval-20260912-v1`；实际队列当前 `WAITING_ASSETS` 且旧supervisor仍存活，**新模型GPU评测尚未开始**。
新队列状态 `outputs/ei-liulu-xqvar-eval-20260912-v1/queue-status.json`；资产完成需源码/配置/模型校验receipt，旧整条pipeline终止+3次空闲才启动。遇外来GPU任务只退出自己的评估进程组。
没有新PSNR/LPIPS结果、没有迁移决定；XQ DINO训练暴露明确披露，不用论文指标代替同协议测量。接下来只执行这一次有界选型，不自动重训通信。
01:22代码/配置/SHA锁定完成；78项CPU-only测试通过（含tiny随机模型API/补全审计，不是预训练质量）；独立XQ权重已通过LFS SHA和CPU元数据读取。处理了作者checkpoint的YAML元数据兼容性，只用精确SHA限定的受限loader；未退回不受限pickle。CPU准备任务PID1869580串行等待后续资产，monitor PID1856483仍在等待。详见`reports/backbone_selection_queue_2026-09-12.md`。

## 2026-09-12：通信骨干2×2已完成，未证实增强骨干的稳定增益

本轮01:43:30（UTC+8）训练/16800行development评测/统计全部完成，四臂各20000更新、80000曝光；三阶段产物和源代码SHA复核通过，GPU已释放，不自动追加训练。
两个next-scale均选总16000步；增强对小骨干LPIPS差-0.001586，95% CI [-0.004106,+0.000725]，PSNR/DINO差的区间也跨0。增强LPIPS 0.236827，仍差于数字m8 0.214927。
两条parallel没有满足共同PSNR约束的候选，按协议回退到10000步warmup端点；不能把主表相对这些回退项的巨大差值当作B配方下的next-scale结构增益。旧B继续只读参考，增强版不自动升为主模型。
解读与全部边界：`reports/token_backbone_result_2026-09-12.md`；原始结果在新代码仓库`outputs/VAR-TOKEN-BACKBONE-20260911-ANALYSIS/`。整个正式训练均为microbatch2，测速4未正式启用。本节不更改上方另行授权的视觉骨干选型任务。

### 本轮历史启动记录（已结束）

用户给定的token域Transformer方案已实现：增强192维/6头/4发送+4共享接收，实际5139662参数；小骨干128维/2层并匹配SNR调制，1559054参数。旧B固定只读；不换视觉模型、不改m8/3060预算、不加功率/全尺度/FEC。
四个严格比较臂统一从头10000步warmup+10000步B、每臂80000曝光；同一家族parallel/next-scale初始化相同，fresh Adam、teacher=0、数据/噪声/增强配对。旧B仅工程参考，训练历史不混排。
CPU检查及16800行合成统计测试通过（合成数据不是实验结果）；真实GPU四臂hard输出与原接收算法差0，image/state梯度到E/D非零，prefix阶段不构建后缀/RGB，frozen权重未变，profile新增更新0。
实测四臂训练主体约8.17 GPU小时，计划校准1.85小时，合计10.02小时，另计optimizer/I/O/最终评测；不是精确工期。
代码和配置在`../var-next-scale-comm/`，协议`reports/token_backbone_protocol_2026-09-11.md`，入口`scripts/run_token_backbone.py`。当前没有新架构收益结果，不按参数规模推断必胜。
14:44首次启动仅遇到空历史分支目录未创建，新增更新0次；目录创建逻辑已修复并补回归检查，失败现场保留。14:46以`configs/token_backbone_20260911_r2.local.yaml`重启，只改训练输出到`TRAINING-R2`，科学设置不变。
14:53四臂各250更新已共同保存，Adam均step250，数据/噪声/增强指纹相同，loss/梯度有限、teacher=0，最大功率误差5.96e-7，旧B及新训练源码snapshot均核验未变。监督器PID1408947；流水线完成后自动评测/统计，异常停止。
启动说明`reports/token_backbone_launch_2026-09-11.md`；最新步数见新仓库`outputs/VAR-TOKEN-BACKBONE-20260911-TRAINING-R2/status.json`。主体约10小时，粗估2026-09-12凌晨完成，不是时刻保证。

## 2026-09-11 13:42：本机m8 A/B训练、开发评测和统计全部完成

`VAR-M8-LOCAL-20260911`于13:42:19（UTC+8）完成全部流水线；训练13:34:52结束，GPU已释放，不自动追加训练或新变量。
双方各新增10000更新/40000图像曝光；按预定完整校准及PSNR约束选A step2000、B step6000，并非双方只训练到选中步数。
主1/4/7 dB、100张development×3噪声：B相对A的PSNR +0.31210 dB、LPIPS -0.090893（相对改善27.60%）、DINO +0.11685，逐项配对95%区间支持改善。
B的LPIPS 0.238479，仍差于固定数字m8 0.214927、数字自适应0.183763和感知DeepJSCC 0.205291；不能宣布强系统胜出。主区间原起点对数字m8的LPIPS差距追回79.59%，仅为描述性比例。
三阶段产物及源码SHA复核通过，12600行评测无重复键，全部3060 uses；216组配对统计复算差0。没有重新做全量独立模型推理/图像指标计算，不访问新test。
结果摘要`reports/m8_local_result_2026-09-11.md`；完整报告`../var-next-scale-comm/reports/prefix_refinement_remote_result.md`，图表和CSV位于新运行ANALYSIS/EVALUATION目录。

## 历史启动记录：2026-09-11首个250步共同断点通过

用户明确表示本机显卡空出并要求开始；此前只准备/暂停限制对本次固定m8实验解除，不扩展至全尺度、功率分配或parallel重训。
RTX 4090 D已确认空闲；原13类资产及官方VAR源码SHA校验通过，不复制大型资产、不改历史结果。
新配置`../var-next-scale-comm/configs/m8_local_20260911.local.yaml`，各新增10000更新，B前2000 prefix/后8000 joint，科学设置与已准备协议一致。
新运行根名`VAR-M8-LOCAL-20260911`；无更新GPU profile/工程检查通过，1000图×5 SNR起点校准相对旧结果MSE/LPIPS差均为0。
2026-09-11 10:30:38（UTC+8）进入正式更新；10:33:23双方各新增250步、各1000图曝光并保存共同断点，Adam step均10250。首250个batch指纹逐一配对，loss/梯度有限、teacher=0，功率误差最大4.77e-7，训练源码snapshot未变。
后台依次执行训练→development评测→分析，异常停止；监督器PID 1113754。日志`../var-next-scale-comm/outputs/VAR-M8-LOCAL-20260911-PIPELINE/train.log`，实时更新数以TRAINING/status.json及共同resume为准。
实测两分支训练加计划校准约2.98 GPU小时，不含optimizer.step/I/O/最终评测，不是完成时刻保证；尚无新方案最终效果结论。详情`reports/m8_local_launch_2026-09-11.md`。以下准备阶段记录为历史状态。

## 当前准备：固定m8配对A/B，后续在其他服务器训练

按用户最新要求，论文主线保持有限带宽/总能量下的next-scale学习通信；当前先准备现有m8映射的训练修订，不同步改全尺度/功率/FEC/网络，也不重训parallel。
准备入口`../var-next-scale-comm/docs/REMOTE_M8_HANDOFF.md`，配置`configs/prefix_refinement_remote.yaml`；同epoch2/Adam状态，每分支新增10000更新初始计划，B为2000+8000。
每500步固定100图监控、预定6个步骤完整1000图选模，监控不选模；真实header、自身历史、硬选择/ST和冻结骨干不变。
资产清单587文件、8.025 GiB已校验，不复制大型资产；核心/配对采样/状态目标/optimizer副本/CLI防误启动CPU检查通过。
正式训练、预训练模型推理、GPU测速均未在本机执行；新服务器须先验证环境和profile/step0，然后由用户显式开训。准备不等于新训练结果。

## 2026-09-10：研究方向更新，不恢复旧m8续训

用户将主线调整为：受带宽、总能量和合法CSI约束的多尺度通信编码、保护分配与next-scale条件接收联合训练。
全尺度信息可参与发送，旧m8和数字自适应继续作为参考/强基线；不把换tokenizer的共同提升当通信贡献。
首版固定组长度、学习SNR条件保护，无反馈、不强制逐尺度CRC；补充噪声固定、等功率/等能量区别和TX/RX信息边界。
方案及小范围原始资料核查见`reports/multiscale_joint_direction_2026-09-10.md`；代码整理仓库的研究说明和协作规则已同步。
**本轮只更新方向文档，不改训练源码或配置；未实现全尺度链、未租卡、未测速、未下载checkpoint、未训练/推理。**旧两阶段实验仍暂停，不自动作为新方向起点。

## 专用资料：VAR原理、通信动机和四个研究问题

用户给出的第二至四部分已整理到`../var-next-scale-comm/outputs/var_report_core_materials_2026-09-10/`。
含修订正文、18张逐段配图、完整五方法主表、CSV及24条口径核对；无PPT、无新增训练/推理。
重点区分源payload与信道次数、Deep的0+3060预算、不同旧流程的m8数值，以及联合训练增量与固定y接收器消融。
已保留全部七SNR原记录和Deep5/6补充，两阶段训练仍暂停。原实验结果未改写。

## 图片素材入口：用户要求只整理对比图，不继续做PPT

可用图片集中在`../var-next-scale-comm/outputs/figure_library_2026-09-10/`，双击其中`index.html`浏览。
检索11个相关项目根，整理124张主图、14张精选：完整猫图m1–m10、当前学习链与强对照、数字尺度/SNR和旧跨数据集图片。
中止产物、无效管线和NOT RUN占位图不纳入可用图；实际历史负结果另设分类。来源与使用边界随图附带。
这只是离线素材整理，不运行模型、不新增训练；两阶段方案仍暂停。代码整理版的`reports/figure_library_2026-09-10.md`记录详情。

## 2026-09-08：两阶段prefix恢复（用户要求暂停；新增训练0步）

用户随后要求“先暂停一下”，先对齐VAR通信研究目标。已检查本轮队列和孤立子进程，当前没有存活的本轮实验进程，
没有训练resume或完成receipt；运行状态已记为PAUSED_BY_USER。未经用户新的明确授权，不重新启动、恢复队列或推进训练。

用户确认两阶段和公平续训对照并要求开始。协议`reports/prefix_refinement_protocol_2026-09-08.md`，配置`configs/prefix_refinement.yaml`。
两分支同next-scale epoch2模型及10000步Adam状态；各新增10000更新，B前2000步CE+0.1state，后8000步image+0.01CE+0.01state。
固定3e-5、全程自身历史、m8/3060预算；每1000步完整1k校准，LPIPS选模且每SNR PSNR相对起点退化不超过0.2 dB。
新实现仅记录和监督前八尺度累计state，不改旧模型/源码；prefix-only不生成后缀/RGB。训练、评测和统计报告入口已实现。
CPU自检已验证原输出/原loss/梯度/官方state目标差0、state梯度到E/D非零、prefix-only仅8次VAR尺度前向/0次图像Decoder。
额外CPU协议检查验证阶段边界、PSNR guard、header屏蔽、optimizer值相同且不共享存储、168组配对统计聚合。它们不是新增训练结果。
当前GPU被另一账户常驻服务占用，禁止擅自停止；正式GPU训练开始前等待至少11000 MiB可用显存。
自动入口`bash scripts/run_prefix_refinement.sh`，具有重复启动锁和断点续接，训练完成后接开发评测及报告。
准确运行状态见`outputs/VAR-PREFIX-REFINEMENT-TRAIN-001/status.json`；日志`outputs/logs/prefix_refinement_pipeline.log`。
不能把已实现、CPU自检通过或GPU排队记成完成10000次更新；正式结果尚不存在。

## 现有训练结果补充：校准端点与固定m8数字对照（复核完成）

仅分析已冻结产物，没有训练/模型推理、改checkpoint或新数据。校准只有epoch1/2两个端点，不是密集学习曲线。
next-scale第二轮校准MSE下降0.8986%，但LPIPS从0.29041462变为0.29998300（恶化3.2947%，五SNR都变差）；
MSE+0.01LPIPS目标仅均值改善0.3143%，新增事后配对区间跨零。原epoch2选择符合协议，未擅自替换为epoch1。
100图开发对照双方固定m8/3060：next在全部七SNR的LPIPS、DINO落后数字m8；4—19 dB三指标均落后，1 dB只有PSNR占优。
19 dB next/digital LPIPS=0.262843/0.177141；不能把差距只归因于数字自适应发m9。next对parallel的逐SNR LPIPS增量仍成立。
核对20000行校准、6300行相关开发数据及72组配对结果（最大差0）；原源码、CSV、配置与选择记录SHA不变。
报告：`reports/prefix_calibration_fixed_m8_review.md`；两张新图及配对表：`outputs/VAR-PREFIX-CALIBRATION-M8-REVIEW-001/`。
本项只读分析任务完成，不据两个校准点宣称收敛，也不自动续训或调整loss/架构。

## 2026-09-07：固定m8全局prefix-JSCC（训练、评测与复核完成，最新）

用户改为真正训练通信Encoder/Decoder，VQ/VAR/官方image decoder全部冻结；不要求无训练方法先成功。
已完成20k训练/1k校准prefix缓存，包含实际重新tokenize的水平翻转；与100图development无精确内容交叉，类别覆盖1000。
图像MSE、LPIPS对E/D梯度和通过冻结VAR的suffix→prefix梯度均通过；训练/评测hard forward像素差0。
两版本通信参数完全相同，均1,508,750；仅global E/D及融合模块训练，DINO不参与loss/选择。
按`reports/learned_prefix_training_protocol_2026-09-07.md`完成各2epoch/40k曝光、10000更新，多SNR、68+2992=3060。
最后6000更新全部self-history；VAE/VAR/LPIPS权重前后完全相同。训练与选择审计通过，两个版本均由校准选择epoch2。
独立校准image MSE+0.01LPIPS目标：parallel 0.02289528，next_scale 0.02077703；不是开发集结果。
100图×7SNR×3噪声、5方法共10500行完成。预定1/4/7 dB主区间，next-scale相对同参数parallel：
PSNR +0.71616 dB，LPIPS -0.03235360（相对8.921%，95% CI=[-0.04079135,-0.02418770]），DINO +0.06787497，三项CI均支持改善。
但当前学习链仍落后于原数字自适应三指标；相对感知DeepJSCC，DINO更高而PSNR/LPIPS更差，不能宣称整体胜出。
这证明了本实现中next-scale参与联合训练的增量，不是宣称学习通信整体已经优于强系统。

两项审计通过：预算/选择/冻结权重，10500行、7348图重算指标、4200个hard-prefix官方重建像素差0，80个学习RX/40个Deep/200个错配body重放一致。
发现旧Deep高频Fourier SNR条件在未训练5/6 dB异常；补充固定最近训练条件（5→4、6→7）且真实噪声不变，避免把基线异常当优势。
主1/4/7结论不变，补充对照不是事前主实验的一部分。错配body显著恶化图像，说明模型利用了源相关数据。
完整报告：`reports/learned_prefix_result_2026-09-07.md`。
本轮固定训练任务已完成，不重启旧state纠正，不擅自改loss/网络或续训；保留训练型next-scale增量证据，原数字自适应仍为主基线。

## 2026-09-07：整帧末尺度辅助搜索（完成，停止本候选，最新）

按用户“开始吧”完成4/5/6/7 dB、100图×3噪声、8臂共9600行；发送/FEC/3060预算、模型及5.5 dB阈值未变。
28组穷举、16个边界、真实码长及神经自检通过；4/7 dB的旧m8/m9共1200条接收记录完全复现。
主区间5/6 dB，VAR−列表65的ΔLPIPS=+0.00005630，95% CI=[-0.00040866,+0.00059500]，无独立优势。
6 dB完整接收API补测（20图/一噪声）约152.84 ms vs列表65的9.90 ms；VAR未把预定达标SNR从7降到6。

前缀瓶颈：4/5 dB真r1:8覆盖0；6 dB为65/296，7 dB为86/100。被覆盖的151帧VAR全部恢复正确整帧，
但不能补回不在4候选中的前缀。固定m9诊断5 dB有1次CRC误接受；冻结adaptive_var该点发m8，未受此样本影响。
大列表效果更好但有44次误接受，且710次受候选/堆上限限制，不能包装成严格等算力或无风险升级。

独立审计通过：3,222,659条候选、3984个重算概率表、1240张图全部重算指标，638张新图旧实现像素差0。
480次完整接收API补测均复现主输出；全模型冻结，未训练/访问新正式测试。
另120次波形到RGB连续调用像素差0；6 dB VAR/list65约211.97/69.38 ms，完整图像链路约3.1倍，不是API的15.4倍。
报告：`reports/whole_frame_prior_result_2026-09-07.md`。
最终`STOP_THIS_FINITE_PREFIX_VAR_RECEIVER`：停止当前4前缀/末尺度候选，不扩大L、训练或搜索，原整帧自适应保留。
本轮结束，不代表停止VAR通信主线；强整帧熵编码未实施，不作总体或新颖性优势声明。

### 此前只读核查

用户提供新方向：整帧FEC/发送预算不动，ML数据CRC失败才启用有限r1:8假设下的r9先验辅助搜索。
这不是继续选择上轮的两张输出。已核查现有600条正确m9接收记录：同口径clean LPIPS=0.13589562、
PSNR=21.24372801；整帧7 dB仅差0.00157537 LPIPS / 0.07404158 dB，支持把关注点移到4—7 dB交界。
历史冻结候选阈值5.5 dB意味着4/5/6/7 dB原模式为m8/m8/m9/m9；不需要根据新目标结果重挑阈值。

技术边界及实施顺序：`reports/whole_frame_prior_review_2026-09-07.md`。
目前仅只读产物/代码及历史文献记录核查；本轮外网访问被拒绝，未完成原文核实。
此前没有新信道搜索、神经推理、训练或资源调整；随后获得实施授权，现已按预注册完成，结果见本节开头。
上轮二候选输出选择路线的停止决定保持不变。

## 2026-09-07：7 dB CRC失败输出回放（已完成）

用户限定只复用100图、7 dB、原三噪声和固定m9，回放首个CRC失败硬候选是否应仅用于图像。
按`reports/crc_failure_replay_preregistration_2026-09-07.md`完成900条分组记录回放及300条整帧对照，
共3000条输出/诊断行，仅新生成219张独特图。没有训练、换FEC、搜索资源或新增信道候选，可信前缀不变。

VAR的33次失败输出全部改善，LPIPS从0.15322436降至0.14302625，追回原整帧差距的64.74%。
但整帧m9仍为0.13747099，VAR retain−整帧为+0.00555526，95% CI=[+0.00281896,+0.00866255]。
VAR二候选oracle与固定retain相同，也不能追平。4次基础失败占VAR追回量52.47%，说明丢弃有用候选的代价很大但非唯一问题。
双方均retain后，VAR比ML低0.02282401 LPIPS，比熵编码低0.00798687，两个配对CI均完全低于0；仅限此7 dB开发回放。

独立审计通过：1200条可信接收轨迹与波形核查、219张新图全部用旧实现复渲染像素差0、580张图全部重算指标；四模型冻结。
完整总体/失败分层表：`reports/crc_failure_replay_result_2026-09-07.md`。
决定`STOP_TWO_CANDIDATE_SELECTOR_FIXED_M9`：停止当前两张输出间的选择路线，不训练选择器或扩大搜索；
固定m9分组原型保留为机制证据，原整帧自适应继续保留。本轮结束，不自动启动新实验，不代表停止VAR通信主线。

## 2026-09-07：三关及独立审计均完成（较早记录）

用户明确要求继续并执行到全部完成。已完成第9尺度7500行同观测译码对照与独立审计：
4 dB正确接受恢复率从0提高到9.6%；7 dB从87.8%提高到97.8%；10 dB各主臂均为100%。
VAR臂未观测到CRC误接受。状态`PASS_SINGLE_SCALE_GATE_ONLY`，不是整体系统收益结论。

独立审计重算波形似然、GF(2) CRC、所有指标/CI，并核对9个实际码长NumPy MAP最优分数。
第三关完成分离20图资源选择和100图完整3060-use对照，八臂共12000行，header/class/mode/CRC/tail均计费。
4/7 dB主比较中，VAR相对同分组ML的LPIPS改善6.79%，PSNR提高0.6778 dB，DINO提高0.01809；
但明显落后于原整帧自适应，与同概率实际熵编码＋FEC也没有显著优势。

最终状态：`MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN`。
保持原整帧自适应主基线；保留整token译码器和机制证据，不将当前固定m9分组原型替换主系统。
没有看到目标曲线后继续调参，也没有训练或访问新测试图像。

第三关独立复核12000条接收轨迹和1555张独特重建的全部图像指标；四个模型冻结，
4个旧argmax（包括有误码前缀）回放像素差0。两关审计均为`AUDIT_PASS`。
完整报告：`reports/next_scale_decoding_result_2026-09-07.md`；对比图在`outputs/VAR-NEXT-SCALE-DECODING-FIGURES-001/`。

## 2026-09-07：第一关先验诊断（较早记录）

按用户“可靠恢复一个尺度，再帮助译下一个尺度”的方向，完成第一关；时间按本机Asia/Shanghai记录。
新代码、配置、原始输出及报告都在VAR_COMM，旧JSCC/VAR资产只读复用，没有训练或新正式测试。

- 100张既有development目标＋100张同类不同图donor；独立于它们的1000张旧calibration拟合静态频率。
- 第8/9尺度真实token NLL：真实前缀9.44293/8.97015，静态频率11.96977/11.91021 bit/token。
  相对静态改善2.52684/2.94006，配对95% CI为[2.36202,2.70979]/[2.79774,3.09341]。
- 两尺度都胜过均匀、静态、同类错误前缀三个对照，超过事前门槛；状态`PASS_PRIOR_GATE_ONLY`。
- 独立bit边缘只保留相对均匀NLL改善的3.32%/2.79%，下一步需保留整token评分。
- 800行指标、频率表、输入内容SHA和131项输出哈希经独立NumPy/PIL缓存审计，状态`AUDIT_PASS`。

结果与边界：`reports/next_scale_prior_result_2026-09-07.md`。
当时据此进入第9尺度同FEC/同观测/同搜索预算对照，先穷举校验，再做后续两关。
第一关本身的正确前缀是oracle条件；第三关已改用真实PHY恢复的前缀，结果见最新记录。
第一关单独不代表通信、图像、实际码长或新颖性结论。

## 2026-09-06：独立工作入口确立

用户明确：当前主要研究 **VAR用于通信**，不要继续把记录写入旧 diffusion-JSCC 目录。
本目录从该次整理起承接VAR通信的新进度、方案、代码和实验记录。
七份已有报告原样归集到 `reports/`；历史代码和原始输出不搬动、不覆盖。

## 已有基础

| 实验线 | 实际状态 | 在当前主线中的位置 |
|---|---|---|
| VAR同前缀补全价值 | 已完成prefix-only / completion / Full-VQ消融 | 已有基础证据，不重复当新机制 |
| 固定3060-use m7/m8/m9 + FEC | 已跑通，策略为1dB→m7、4dB→m8、7/13/19dB→m9 | 保留基线 |
| Receiver-prior跨数据集验证 | 2026-09-04已完成ImageNetV2/CLIC/Kodak结果 | 保留历史正式结果和使用记录 |
| Decoder-only微调 | PSNR提高但LPIPS/DINO退化 | 负结果/权衡，不作为无损成功 |
| 同前缀量化cell约束修正 | 20张自检、100张主消融和3种子诊断已完成，未过继续投入门槛 | 停止该具体候选 |

cell约束修正的主消融LPIPS相对改善仅0.64%/0.92%，不如简单prefix混合；
三种子诊断改善1.30%/1.69%，仍低于预注册5%门槛。完整结果见
`reports/var_prefix_consistency_result_2026-09-06.md`。

## 当前边界

- 已完成先验辅助FEC/CRC、真实逐尺度接收和同概率熵编码对照；未做新的神经训练或全新正式测试。
- 新接收器、PHY和审计代码独立写入本项目；历史完整系统未搬迁，旧模型/数据保持原路径。
- 自身公平预算已核查；优化LDPC/Polar、完整近作复现和系统新颖性审查尚未完成。
- 按用户追加授权完成7 dB首失败硬候选的图像回放；二候选oracle仍未胜过整帧，该选择路线停止，不训练选择器。
- 不根据本次100图结果继续追调配置；其他模式/失败输出机制尚未执行，不是本轮已取得的收益。

后续判断和实验都围绕VAR用于通信展开；某个模块停止不代表自动退回旧diffusion/JSCC主线。

## 数据边界

原ImageNet-100永久为development；2026-09-04的ImageNetV2 2000 query和10000 gallery已经使用。
后续若需要全新正式测试，先核对各项目历史曝光记录。
本轮另有100张同类donor作为development曝光；旧manifest虽标为test，也不能再声称未见。
第一关没有访问ImageNetV2/CLIC/Kodak的新测试图像。
