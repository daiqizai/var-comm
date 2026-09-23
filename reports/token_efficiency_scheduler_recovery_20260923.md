# Scheduler recovery after predecessor job-control state changed

Observed: after the monitoring SSH connection closed, the original stopped
short-prefix controller resumed and attempted stage6 (m6 training). That child
exited at `require_available` before model loading because P2048 owned GPU0.
The original controller then exited. There is no m6 latest/checkpoint and no
m6 optimizer update. The exact delivered signal sequence was not logged;
we do not present a particular SIGHUP/SIGCONT sequence as proven.

A SIGSTOP state tied to a live predecessor was therefore an insufficient durable
priority gate. Leaving the supplemental coordinator unchanged would also make it
fail its predecessor-state check before P3060. The original failure log/status,
source binding, completed20k cache identity and process snapshot are preserved in
outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/recovery/controller_retirement_1790177146/.

The supplemental controller received SIGTERM and forwarded the existing graceful
pause request. P2048 saved step5967 with full model, optimizer, data order and RNG
state; checkpoint SHA256:
`7f5e657de2a2d03ecf01dfd1178c23ba83962594270a32ebf79208a0b8401258`.
Both controller and trainer exited normally from the safe-pause path. No active
training code, old controller, original queue identity or historical result was
rewritten. The prior failure remains a failure record.

The additive coordinator_v2 verifies an explicit retired-predecessor receipt,
checks original source/cache/failure evidence and current root processes, holds
the original controller lock, and preserves completed source/qualification stage
receipts. It resumes the same budget trainer and its hash-verified checkpoint.
It never signals a recycled PID. The live-process SIGSTOP assumption is removed
from the successor's gate. Original queue code remains unchanged for later C.

Four CPU lifecycle regressions cover retirement, recycled-PID distinction, live
work rejection, altered evidence rejection and duplicate-controller lock denial.
Existing actual-device gradient/optimizer/resume qualification is rerun by the
unchanged trainer before restoration; restored real training must advance beyond
5967 before recovery is reported complete. B1/B2 are still incomplete. C is not
restarted until A/B delivery and an explicit durable release decision.

## Actual recovery acceptance

The successor was launched with `start_new_session=True`. Its session ID equals
its verified controller PID, and the training child belongs to that session.
P2048 restored the recorded step5967 safe checkpoint and advanced beyond6000;
the latest checkpoint and selected step5000 hashes, active source bindings and
predecessor lock were verified. The actual-device populated-optimizer isolation
and bitwise-resume probe passed again; discarded probes do not train the model.
See results/token_channel_efficiency_20260923/scheduler_recovery/resume_acceptance.json.

Recovery implementation commit `a4bb6e8172ca774ea1edafe7d80c5d8ffca80f97`
was normally pushed and independently fetched from origin. All135 CPU tests,
repository/release validation and git fsck passed in that checkout without object
alternates. This verifies scheduling recovery, not B1/B2 quality completion.
