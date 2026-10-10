# N2048 finite calibration and confirmation execution

This document describes executable new-stage consumers. It is not a completion
claim. The original scientific files, original policies and original ledgers are
unchanged. The remote owner alone starts the explicitly registered stages.

## Scope and costs

The actual constructor-admitted RAW catalogue has 521 unique wires, 131 whole
actions, all 9 full-budget whole actions, and 3 complete m10 actions. All actions
pay the 68-symbol header and fit the 1980-symbol body in N2048. PARTIAL includes
every whole action; it is never forced to have a positive K.

| Stage | Registered logical conditions | Maximum new physical packet callbacks |
|---|---:|---:|
| RAW pilot: first100 original calibration sources, 521 actions, 4/10/19 dB, seed4101 | 156300 | 312600 |
| RAW full1000: per-SNR union of whole top5, partial top5, all9 full-budget whole and all3 m10, three seeds | at most198000; exact union registered after pilot | at most396000 |
| Entropy pilot: one N1024-calibration-selected family, first100, three SNRs, seed4101 | at most14400; exact count follows actual constructor admission | at most28800 |
| Entropy full1000: per-SNR pilot top3, three seeds | 27000 | 54000 |
| Frozen new100 confirmation: 3 methods, three SNRs, seeds9201/9202/9203 | 2700 | 5400 |

These are logical conditions and physical packet caps, not GPU render counts.
The 521 RAW wires have only **48 distinct clean (m,K) states**. If all received
tokens were correct, pilot rendering would have at most4800 source/state pairs
before any historical cache reuse, a96.93% reduction from156300 logical rows.
This is a conditional clean-channel scenario, not an observed cache-hit rate.
Actual erroneous hard tokens can create many additional states.

Existing whole-prefix caches are protocol-compatible with nine m1..9 shapes
covering128/521 wires, or38400 pilot logical conditions. This24.57% membership
opportunity is not a promise of cache hits: same-source actual token values and
all model/evaluator/numerical identities are checked for each hit. Original RAW
and completed T2 and T1 calibration caches are preferred. The T1 cache supplies
actual independent entropy-decoded states, including potential clean m9 states,
with existing DINOv2-L, LPIPS and PSNR measurements.

The new CPU receptions are completed before `plan-gpu` or `plan-render` seals
the actual known-state/cache inventory and finite new neural-call bounds. Two
GPU processes then consume disjoint sources. Both must reproduce original
float32 RGB exactly in overlapping qualification windows before production.
Calibration also checks all original metric differences against the frozen
tolerances. Confirmation qualification invokes zero metric calls.

## Historical N2048 evidence

The old M1 and M9 calibration source checkpoints contain DINO-S14
(`dinov2_vits14_pretrain.pth`), not DINOv2-L. They do not contain saved actual
token states or calibration RGB archives. Their channel protocol also differs.
Their old frame counts must not be relabelled as new T6 receptions or DINO-L
calibration. No completed M1 calibration closure was supplied in the audited
metadata; M9 does have its own historical complete closure.

H/full1000_render has an actual1000-source/42000-frame completion and RGB
archives. Its renderer source hash, frozen visual identity and numerical flags
match the current renderer. Its first-source sample has9 unique images and9
actual received-token hashes; only whole m7/m8/m9 intersect this T6 finite
catalogue. The six partial K shapes in that sample are not T6 legal actions.
H provides PSNR/MSE only, so any H RGB reuse would still require current LPIPS
and DINO-L scoring. The current implementation prefers the already scored T1
states instead of scanning all H archives speculatively. H images are never
borrowed by source ID or by nominal m alone.

`t6_cache_audit.py` reproduces the metadata-only48-state calculation and writes
the exact input SHA list and conditional clean-state table. It opens no source
images and performs zero PHY/model calls.

## Executable order

Use explicit new output directories under
`outputs/WCL-EVIDENCE-CLOSURE-20261009/` and fixed deadlines. Keep CPU PHY
processes at the original2threads and1interop; keep the original GPU numerical
flags at6threads and2interop. The remote owner schedules exclusive GPU windows.

1. Complete independent `t6_qualify_raw_phy.py prepare/run` and
   `t6_qualify_entropy_phy.py prepare/run`. RAW qualification covers each actual
   wire layout and every paid header. Entropy admission records all48 candidate
   resource queries; unsupported Sionna single-codeblock layouts are explicitly
   rejected before any quality access, with no truncated k and no substitute
   approximate code. Every legal entropy MCS exposes m4..10 to the receiver.
2. RAW: `t6_calibration_plan.py prepare-raw` freezes the exact pilot/full scope;
   `t6_raw_calibrate.py prepare-execution --plan-request ... --t2-completion ...
   --t1-completion ... --gpu-workers 2` adds current executable graph and strict
   historical cache bindings. The T1 argument is optional but recommended.
3. Each RAW or entropy calibration stage: parent `ledger-init`, then CPU worker
   indices0..7, wait for real exits, then its `plan-gpu`. Its independent two-GPU
   owner executes the exact-state plan. `close --owner-receipt ...` requires the
   owner's real `wait()`/zero-exit receipt. Run the separately registered full
   stage only after the complete pilot closure. Entropy uses
   `t6_entropy_calibrate.py` and `t6_entropy_gpu_parallel.py`.
4. `t6_freeze_policies.py --raw-completion ... --entropy-completion ...
   --entropy-family-selection ... --out ...` emits all9 winners only from
   complete original1000-source, three-noise calibration. It opens no
   confirmation data. This freeze is an input to the new100 source owner.
5. The separately owned new100 source preparation verifies the fixed IDs and
   audited content hashes before Encoder access. Any duplicate stops the stage;
   no replacement image is selected. It prepares original pixels/tokens and
   only the chosen entropy prefixes, independently decoded from their streams.
6. `t6_confirmation_render.py prepare --source-completion ...
   --calibration-freeze ... --environment-request ... --out ...
   --deadline-unix ... --gpu-workers 2`, then `ledger-init`.
   Start CPU workers separately with `--branch raw` and `--branch entropy`.
   The two branch processes preserve their original module bindings and share
   one SQLite budget5400, paired full2048 noise and fixed counters. Equal whole
   and partial physical configurations reuse the same actual frame receipt.
7. After both CPU branches finish, `plan-render` seals exact known states and
   the upper bounds for any genuinely unknown received entropy stream.
   `t6_confirmation_gpu_parallel.py owner --request ... --session ...` holds
   the common GPU lock, qualifies two disjoint source workers, waits for both,
   and writes the receipt. `t6_confirmation_render.py close --request ...
   --owner-receipt ...` emits the2700-row renderer completion consumed directly
   by `t6_score.py`, then the separately registered statistic consumer.

## Recovery, fairness and export contract

Actual received headers select the complete public receive profile, including
wrong accepted headers. RAW keeps actual hard tokens even when body CRC fails.
Entropy only sends CRC/parser-accepted bits into the independent source codec;
only source syntax failures become gray. Runtime/model/asset errors stop work.
TX truth enters only offline token-error diagnostics after the image is fixed.

The parent initializes the packet WAL once before parallel children. Subsequent
WAL constructors serialize, while real decodes remain parallel. Each actual
callback is reserved before execution; COMPLETE events reuse exact results and
unresolved RESERVED events stop rather than repeat. Source decoding, rendering
and reference-feature preparation also use durable reservations. Saved source
reference features restore the original tensors without recomputing models.

All images are exact float32 CHW RGB256. `per_frame.csv` preserves method and
point IDs, source/seed/SNR, actual receive/failure status, header/body/parser and
source-canonical outcomes, target and actual m/K, modulation and actual k/n,
paid header/body/padding, actual energy/rho, entropy lengths/fallback attempts,
and offline token errors. Each image reference carries `image_path`,
`image_key`, `image_slot`, file SHA and exact RGB SHA. Completion directly seals
every referenced image and the CSV. All2700 rows, failures and zero differences
are retained. Confirmation performs zero metric calls; the original four-metric
SuiteBackend consumer prepares all100 new references and never borrows the old
500 reference cache.
