# History and reuse

Raw R1: 200 calibration sources, 32,400 physical frames, 43,200 logical
reconstructions; J_old=.75 matched+.25 TX-overestimated-by-3-dB. The two-group
minus strong-single result was −0.009211461585946379, NO_GO_STOP. Raw R2 did not
run and remains unlaunched.

LE0/LE1: 24 independent arithmetic round-trips, 1,200 clean states for 50 sources,
750 exact reused and 450 newly generated. Two-stream overhead was median 30 bits,
range 28–32 (one additional L13+CRC16 and actual termination difference). No new
wireless image frames were part of that study.

LE2: 18,688 real packets and 16 two-group proxy nominees. Best two-group proxy J
was 0.6919812165440816, versus 0.7021032469582041 for strong single; difference
−0.010122030414122518. NO_PROMISING_MEASURED_CANDIDATE was a bounded proxy result,
not an exhaustive search or a completed image-channel grid. LE3 remains unrun.

The new mechanism is same-boundary-scale C+B with A absent. It does not relabel
the older cross-scale experiments or alter their objectives/budgets. Reuse is
limited to frozen integer arithmetic, conditional probability/native generation
definitions, paid packet syntax and the actual LDPC/header backend. New block
layouts still require real qualification and new complete RX records.

The qualified cal1000 PRO6000 source manifest supplies preprocessed pixels,
tokens and 25 single-source streams. No retokenization or single-source entropy
traversal is planned. Only the new three-stream organization is encoded. The
seven moved weights retain their exact original content hashes and are accessed
through results/checkpoint_relocation_20261028_v1/model_routes.json.

Old image/metric caches are not yet admitted. A later explicit reuse inventory
must verify originating process closure, actual received content, full native
pixel hash, weight/numeric identity and source preprocessing. Continuous states
may match; a C+B hole never matches merely by count or old prefix-only hash.
No historical H800 arithmetic stream substitutes for the corrected PRO route.
