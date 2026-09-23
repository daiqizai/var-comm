# Short-prefix hybrid study (authorized 2026-09-23)

The original EXPERIMENT_PLAN.md is the user-authorized design, not a completion
receipt. protocol.json fixes budgets, data/noise, main1/4/7dB region, training,
selection and later-stage gates before new development access.

The new protocol uses mode bits0/1/2 for m6/7/8. The old m7/8/9 functions and
archives are unchanged. In the first fixed-codec matrix the singleton mode is
pre-shared: another CRC-valid mode is an illegal field and a scored erasure.
In the later adaptive family the actual decoded mode will choose the registered
body/continuous boundary and receiver. RX never receives the sender's true mode.
No pure4084 switching is permitted without a charged mode-signalling redesign.

Five new hybrid arms use the same sender form and same class/status/spatial
receiver architecture. V/P start with identical communication weights, independent
optimizers and paired samples/noise. Pretrained VAR capacity/cost is additional.
The archived full/light arms and original A/B are unchanged.

Run from the project root:

```bash
bash experiments/var-short-prefix-hybrid-20260923/scripts/run_first_matrix.sh
```

This durable single-GPU queue repeats real-weight qualification, builds complete
new-m actual-RX caches, trains m6 V/P, m7 V/P, m8 V to20k, and resumes the verified
pure parent from10k to20k. Every500 updates save weights, per-arm Adam, order,
noise/RNG and identities. Every2500 updates use all1k calibration sources, all
fiveSNRs and three seeds for selection. Cached training and online output have
been compared with real weights after discarded probe updates. They use the
same model TX/noise/RX functions; online cost additionally includes encoding,
PHY and actual VAR completion. No synthetic quality numbers are published.

Queue source identity is frozen before launch. A second controller is rejected
by a process lock. SIGTERM requests a paired-update checkpoint; other GPU users
or sustained thermal slowdown stop safely without changing hardware settings.
After a failure inspect the recorded log; do not edit a running dependency or
remove identities to force resume. Cache shards without a receipt are incomplete
and can be regenerated; committed hashes/scopes must match exactly.

Runtime status: outputs/SHORT-PREFIX-20260923/status.json. Training status and
selected records are in training/m{6,7,8}_seed2026092304 and pure_seed2026092304.
20k is a milestone: extension is decided from complete calibration only. The
queue intentionally does not automatically open new test data or infer completion
of the later N3060, seed-repeat and conditional content-policy stages.

Preflight summaries: results/short_prefix_20260923_preflight/.
Weights, pixels and tensor caches stay local under ignored outputs/.
