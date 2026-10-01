# N1024 extreme bandwidth study

The user explicitly authorized revised R1 stage two on 2026-10-01. This directory trains one fresh P1024 and delivers N1024 digital and receiver-plugin results, a paired comparison with frozen N512, and compatible completed resource references. The pipeline stops after checked publication.

## Registered scope

- Fresh P1024 uses the actual P512 recipe and initialization seed2026093001. N1024 has eight communication channels and actual per-frame E2048; only the budget-dependent TX/RX shape changes.
- First milestone20000; complete1000-source calibration every2500. The unchanged adjacent-interval gate may extend to30000/40000. A still-improving model at40000 is budget_truncated, not converged.
- Paid header68 and body956. QPSK m4/m5/m6 and16QAM m4/m5/m6/m7 are freshly calibrated on1000 sources. C/U are separate conditions;16QAM records original actual frame energy.
- A1/A2/VAR policies are freshly calibrated on200 sources using the original grid, true BYPASS and common-weight ablation. Development has100 fixed sources and three registered noise repeats.
- All primary outputs use the same frozen Dc. D0 uses the identical received digital latent. H_D and H_R stay separate; all failures and fidelity costs are reported.
- N512 remains frozen at its selected40000-step checkpoint with budget_truncated=true. No N256/N128, generation training, new loss, or historical queue is launched.

See [authorized scope](EXECUTION_PLAN.md), [protocol](protocol.json), and [training protocol](training_protocol.json).

## Execution

Run CPU checks and discarded-update real-GPU qualification, register source, then detach the supervisor:

```bash
python3 experiments/extreme-bandwidth-20261001-N1024/qualification.py
python3 experiments/extreme-bandwidth-20261001-N1024/publish.py --register-only
python3 -u experiments/extreme-bandwidth-20261001-N1024/supervisor.py
```

The supervisor enforces source identities and records durable stage status. It retries only safe resource pauses, preserves unknown failures, and never launches another experiment after publication.

Weights and caches remain in `outputs/EXTREME-BW-20261001-R1-N1024`. Lightweight results go to `results/extreme_bandwidth_20261001_R1_N1024`; the report goes to `reports/extreme_bandwidth_probe_20261001_R1_N1024.md`. Large tables use exact byte-preserving parts below the repository size limit. Historical points are reused only after checking source, preprocessing, decoder and noise identities.
