# Proposed RX import continuation — not authorized or dispatched

The original qualification TX completed48 subframes once. Original RX completed
six header and six body calls, then the first ARPC source syntax import failed:
the upstream `utils` package was absent from `sys.path`. Original q0000/q0001
results are persisted; q0002's four successful decoder-call results are lost.
No model inference occurred. All original bytes, ledgers and actual exit1 remain.

This independent entry resolves and qualifies the unchanged wire parser before
any packet reservation. It reuses both persisted pairs and their actual RX bytes.
It would process only the remaining22 observations, including q0002. That repeats
two header plus two body decoder calls, explicitly counted, without another
transmission or source traversal. It additionally persists the physical result
before source parsing. Original registered code is not edited.

The proposed qualification decoder bound is100 instead of96, combined bound7300
instead of7296. Physical subframes remain3648; logical objects, models, datasets,
MCS, source representations, scoring and bootstrap scope are unchanged. No new
authorization JSON has been issued. The entry requires an exact hash-bound user
approval descriptor and cannot execute under the old96-call qualification cap.

Synthetic import/container checks prove engineering readiness only. A successful
new real RX, independent model roundtrips and visual/resource review are still
required before any formal N2048 frames. No automatic retry or successor runs.
