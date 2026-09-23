# Short-prefix study: implementation and launch qualification

Base commit: cf70b314d596fa09508496af66e0366509517952. The sole root is
/home/liulu/projects/VAR_COMM. The initial worktree was clean, GPU0 was idle,
and no previous training task was running. Original checkpoints/results remain
unchanged. This report records qualification and first-matrix execution, not a
new quality result or completion of the full plan.

## Registered scope

The user's complete design is saved in
experiments/var-short-prefix-hybrid-20260923/EXPERIMENT_PLAN.md. protocol.json
pre-registers five new N4084/E8168 hybrid arms; short modes6/7 use2816/2048
continuous uses respectively, with a new matched m8 control. True class at TX
and ideal shared nominalSNR are disclosed assumptions. Class and mode must pass
through the68-use protected header. Prefix-RX and VAR-RX share the same trainable
network structure and received class information; only VAR-RX runs frozen VAR.
All image failures are retained; latent auxiliary loss is masked on header failure.

The first stage uses fixed modes known as receiver configuration, and rejects
unexpected decoded modes. This is not a free sender-mode oracle. An eventual
content policy must allow the three received modes and dispatch by actual header,
including wrong accepted boundaries/labels; fixed-mode outputs cannot simply be
spliced into a policy while ignoring those failures.

## Actual preflight evidence

100 fixed calibration sources x fiveSNRs x three seeds x three modes =4500
real PHY transmissions. No development or new-test sources were used. Actual
VAE/VAR weights were used to measure decoded-prefix and completion errors.

At1dB, token error rate including erasures is7.0073% (m6),8.7935% (m7), and
13.8366% (m8). Header acceptance was100% in this finite sample; body CRC acceptance
was4.6667%,0%,0%. At4dB m6/m7 had no token errors in this sample; m8 had0.015686%
and98% body CRC acceptance. These finite observations do not guarantee reliability.
The default allocations are retained. CRC failure does not mean zero useful
information; no backup-protection training point has been launched.

The full root suite passes90 CPU tests. Sixteen new regressions verify actual noiseless FEC, legacy mode6 rejection,
exact budgets, actual decoded-mode dispatch, single noise, official ten-scale
prefix accumulation, matching parameters and populated-optimizer isolation.
Real GPU probes for all five new arms verify frozen-Decoder gradients, waveform
energy, independent parameters/buffers/Adam state, and bitwise next-update resume.
The probe updates were discarded. Twenty real-image cached-training versus online
TX/channel/RX comparisons passed at2e-5 tolerance; repeated online outputs were
identical. Detailed receipts and source bindings are published in
results/short_prefix_20260923_preflight/; they are engineering/PHY evidence, not
trained-model quality rankings.

## Next dependency stages

The first-matrix controller runs full new protocol caches and20k training milestones
with full calibration selection. Pure4084 preserves its actual10k checkpoint and
Adam/order/RNG identity before continuation. Additional10k opportunity is assessed
from the registered calibration trend, never development performance.

Selected-model development/timing, N3060 confirmation, two additional training
seeds for final candidates/direct controls, content-selection diagnostics (and a
predictor only if justified), and500 new communication-test sources remain pending.
The new test is opened only after model/protocol/policy freeze and a verifiable
exclusion manifest covering all historical communication populations.
