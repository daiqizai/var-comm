# 当前：5000步训练和固定500图测试已闭合

固定检查点对原始NeST进行了同预算真实终止码流/独立接收比较。源码容量591、1273、1911、2548bit，另收41bit信令与CRC；完整680tokens单独作为无预算重建上限。没有新增无线PHY或恢复旧研究臂。
正式入口：formal_chain_metrics_v4.py、formal_owner_v3.py、formal_worker_v1.py（TX/RX）、formal_worker_v2.py（score_v2）；不应重新执行已有角色。
注册、全部指标与正负解释在../../results/nestcom_rd_transfer_20261106_v1/formal500_v1/。
resource_v1/v2仅启动前资源检查，resource_v3在资格指标导入前失败；真实编码/接收成功结果原样保留复用，指标定义隔离导入后从缺口接续。
训练全记录在training5000；历史启动/200步保护停止/继续训练证据继续保留。旧四臂和640M仍暂停。

--- 以下为原始阶段记录 ---

# 当前：用户授权200步接续，延长观察

最新用户“多训点试试”已执行。活动入口train_continue_v1.py/owner_continue_v1.py，
continuation_v1.json绑定代码、数据、恢复源。原八卡完整状态已恢复并实际执行201步；
目标累计5000，最多3小时，截止2026-11-06T13:24:30.035577+08:00。质量阈值改为监测，不因早期下降停训。
旧STOP和旧协议不覆盖；本段STOP_CONTINUATION_V1单独控制。模型/目标/数据不变。
详情：../../results/nestcom_rd_transfer_20261106_v1/continuation_v1/REPORT_zh.md。

--- 以下为原段历史说明 ---

# Original NeST rate-distortion transfer — 2026-11-06

User authorizes implementation, necessary checks and training of RDVQ-style
differentiable rate optimization in original VAR/NeST. This is a new experiment,
not a continuation of the stopped 639.6M RDVQ model. The four old P2 arms remain
paused. Original WCL code, weights and results are untouched.

Initialization: original VAE encoder/quantizer, VAR-d16 and adapted continuous
decoder, all verified against the relocation manifest and tensor hashes.
Ten scale sizes `(1,2,3,4,5,6,8,10,13,16)` and 4096-entry codebook are retained.
Codebook and scale-specific residual transforms are frozen. Encoder + quant-conv,
VAR prior, and continuous decoder are trained. No RDVQ weights are mixed in.

## Training and gradient meaning

Hard token assignment preserves the original residual quantizer's forward.
Soft assignment (distance temperature 0.1) supplies a differentiable rate target
to the encoder. The original hard coarser contexts supply the prior's
teacher-forcing inputs. Those contexts are detached: the rate-to-encoder path is
the soft assignment, not a hidden same-scale or future-token input.

Actual scale-wise unconditional class1000/no-CFG probabilities create integer
CDFs. The original arithmetic encoder measures independently terminated streams
at all 651 legal positions (30..680). Longest feasible source prefix is selected
under source capacities 591,1273,1911,2548 bits; body length13 + CRC16 are charged
in addition. Profile12 identifies the endpoint in the preconfigured directory.
The packet adapter here is a clean source-body training harness, not a new radio
test or claim that changed weights have completed LDPC/QAM qualification.

RX separately reconstructs CDFs and decodes canonical hard token prefixes.
Greedy completion starts only from those decoded tokens. For image gradients,
received embedding values use an identity straight-through derivative and keep
hard forward values. Completion argmax has **no image gradient into the prior**;
the prior is trained by soft-rate and hard-token cross entropy. The decoder sees
the actual hard partial completion and full hard representation every update.

Loss (RGB in [0,1]): partial MSE + .1 LPIPS-VGG + full MSE + .1 full LPIPS-VGG
+ .05 soft prefix bpp + .01 mean hard-token CE + .25 commitment MSE.
This is the registered first design, not an optimized or proven RD objective.
Encoder/decoder LR1e-5, prior3e-6, AdamW(.9,.95), weight decay .01, 100-step warmup,
global batch8, one image/device, gradient clipping1, FP32/TF32off/math attention.
The differentiable teacher-forcing prior uses block activation checkpointing.
The rate-gradient-stopped switch exists for a subsequent matched control; no
matched-control training result is claimed by launching the main model.

## Data and validation

Reuse the verified 116238-image COCO training corpus, already excluding the old
500 test sources. Before inference, hash-order selection reserves 32 development
images and two qualification images; 116204 remain for training. No old heldout
test images choose hyperparameters or checkpoints. Train uses deterministic
resize/crop/flip; development uses fixed short-side256 + center crop.

Every 100 updates evaluate full reconstruction on the fixed32 and four real
source budgets on the fixed first8. Stop after two consecutive checks with
full-PSNR below initialization by >.15dB or LPIPS-VGG worse by >.01. These are
development guardrails, not formal statistical tests or guarantees. All adverse
measurements are retained. Each update logs separate losses and actual lengths.

Finite first tranche: at most5000 updates; the absolute registered deadline is
preserved. Two rolling checkpoints contain optimizer, all eight RNG states,
sample cursor and exact request/source hashes in the dedicated checkpoint tree.
User STOP or deadline causes a safe save. Unknown exceptions are retained and
never retried automatically. Qualification-updated weights never initialize the
main run. Necessary tests must pass before training launch.

## Entry points and evidence

`protocol.json` / `model_routes.json` / split manifests are recorded before
scientific calls. `qualify.py` checks original hard token/latent/pixel parity,
rate gradients, detached control, all trainable gradients, frozen quantizer,
two updates, and new-checkpoint independent decoding in a separate process.
`train.py` is the sole main trainer. `owner.py` records PID/start_ticks, actual
exit, elapsed GPU-device and CPU-process time. No automatic old-job resumption.

Raw tensors, weights, full logs and corpus manifest stay on PFS. Git receives
source, scope, split identities, qualification summaries, resource receipts and
honest launch status. No new quality superiority claim is possible at launch.

## Actual qualified runtime

`model_v2.py` and `resize_gradient_v2.py` retain native forward interpolation and use a tested deterministic separable adjoint. Original failed v1 is retained. `qualify_v2.py` passed two real optimizer updates and separate-process RX. `train_v3.py` / `owner_v3.py` / `protocol_v3.json` are the sole live main route; v3 adds explicit checkpoint restoration and retains the original development baseline. First-tranche deadline is 2026-11-06 10:00 Asia/Shanghai, no extension. Main has actually updated and saved full state. Future restore must be separately admitted and preserve the original deadline unless the user authorizes more time.

## 实际终态：200步完整重建保护停止

八卡主训练执行200更新、1600个不同训练源曝光，全部损失/梯度记录有限，
全部实际前缀均在预算内。第100/200步固定32开发源完整PSNR分别21.824936/
21.880217dB，较原22.493114均超过0.15dB退化阈值；第200步LPIPS-VGG从
0.203780变为0.209945。连续两次触发，按协议安全保存、actual0、进程退出。
训练未达到5000上限，未到10点就提前停止；不是用户暂停旧模型的继续训练。
当前新旧模型与旧四臂都没有活动训练，STOP保留，未自动改权重、换目标或续跑。

完整200步检查点约5.29GB，包含优化器、八rank随机状态、游标、原开发基准及
保护计数。训练本身0.635174 GPU设备小时，连同资格合计见training_resource_final。
原来launch_snapshot是历史启动快照，当前状态以guard_stop与本节为准。

开发集完整/四容量全部逐源和均值见development_metrics/summary.csv，包含不利值。
这是32/8源的早期开发观察，无置信区间、不使用500测试源、不宣称路线失败或成功。
当前配置未守住完整重建目标；该下降不能单凭此次运行归因为码率项，仍需匹配
停止率梯度对照及损失分项分析。正式测试、匹配控制、继续训练均未执行。
