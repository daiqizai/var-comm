# Thermal pause and unchanged-state resume

At the 15:57 UTC monitor, the P2048 process and selected7500 checkpoint were
valid, but GPU0 showed software thermal slowdown. Subsequent10-second samples
were83C/software Active,84C/software Active,86C, then86C/software Active.
The existing Safety helper watches only hardware thermal slowdown and >=86C;
it can miss sustained software throttling below86C. This is a monitoring gap,
not evidence of NaN, weight corruption or invalid quality measurements.

The verified coordinator received SIGTERM and its trainer saved step8740 at a
safe update boundary. Actual checkpoint SHA:
`c0bbc466dd6386ecf26ab3b3c5aa1fbb3907d9788d18b651e4b253092e2f19ac`.
Both processes exited and their prior receipts/logs remain unchanged. The actual
observations and checkpoint identity are in
results/token_channel_efficiency_20260923/scheduler_recovery/thermal_pause.json.

An additive external thermal_guard wraps the unchanged coordinator_v2. It checks
both thermal flags and temperature every10 seconds, requests graceful checkpoint
pause after3 consecutive hot samples, and requires6 consecutive cool samples
(<=75C, no thermal flags) before resuming. It verifies PID/start_ticks/cmdline
before signals and verifies saved checkpoint hashes. Unknown failures stop the
guard without automatic retry. Cooldown can reduce throughput, but does not
change training samples, RNG, optimizer state, model, loss or numerical protocol.
No power, clock, fan or other shared hardware setting is changed. This guard
covers the current budget queue; subsequent evaluation/C controllers must adopt
equivalent monitoring before launch. It is not whole-study acceptance.

Synthetic CPU tests cover software-only throttling, malformed/unknown sensor
values, consecutive-sample pause/cooldown, recycled/wrong PIDs, safe-checkpoint
hash/reason validation and surviving-child rejection. Actual resume acceptance
must separately record advancing steps, process/session identities and GPU state.
