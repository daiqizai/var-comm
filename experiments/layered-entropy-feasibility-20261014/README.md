# LE0 / LE1: layered entropy source feasibility

Read `user_package/scope.json` and the complete user plan. They are a plan,
not a prior implementation or registration. Only LE0 and LE1 were authorized.
The actual result is `results/layered_entropy_feasibility_20261014_v1/REPORT_zh.md`.

`source_split_codec.SplitSourceCodec.encode_all(tokens)` performs one TX prior
traversal and generates 24 single streams, four independently finished base
streams and 68 genuinely encoded enhancement streams. Each enhancement has
fresh arithmetic state while retaining the causal VAR context after its base.
No full-stream slicing or length subtraction constructs an enhancement.

`decode_received(received_header_bits, base_information,
enhancement_information, public_catalogue)` receives only actual paid-profile
bits, two protected information vectors and shared frozen models. It recomputes
CDFs from received tokens. A rejected base yields gray; a rejected enhancement
retains its actual base and uses the unchanged greedy VAR completion. There is
no damaged arithmetic KEEP, TX CDF, source pixel, unsent-token or oracle input.
Source tokens/CDFs are compared only after receiver output is saved.

Every group pays a 13-bit length and 16-bit CRC, including an explicit empty
enhancement profile. The frame is 68 header plus 956 body complex symbols,
nominal Es=2. The null profile preserves failure denominators when base cannot
fit. The 73-entry source proof catalogue is explicitly source-only and not a
radio-qualified catalogue. Future physical profiles must pay actual j/m/K and
all group allocations/MCS in the public 12-bit profile directory.

Original `ep_phy.Backend.plan` remains unchanged with n=956*q. The independent
`group_layout_design.GroupLayoutPlanner` exposes constructor metadata only;
it has no packet encode/decode/channel API. `candidate_plan.py` used the actual
pinned Sionna constructors for legal group k/n/q and calculated the capacity
domain and precise prospective BLER gaps. No encoder/decoder forward was run.
Proportional equal-MCS allocations are available to U as well as EEP; eventual
coincident U/EEP choices are allowed. No inference from nominal rates to BLER.

Do not run completed entries again. The original asset label failure, GPU3
resource stop, and metric API failure are preserved. Explicit isolated recovery
entries reuse completed bytes, keep cumulative science counters and original
deadline, and disclose necessary model reloads. See RESOURCE_RESUME_V1.md and
the score-completion registration. All actual 50 TX / 24 RX / 450 missing clean
renders / 450 new DINO scores completed. The old 750 clean scores were reused.
Tokenization, wireless frames, packet trials and training updates are all zero.

First38 streams retain exact terminated bitvectors and complete lengths, but
their separate finish increments were not flushed before the resource stop.
These auxiliary CSV fields remain explicitly missing, without repeating TX.
The full actual B+E-single segmentation cost is exact for all3400 rows.

LE0 includes every actually evaluated old raw candidate at true7/10dB. Its
NO_GO is preserved. Clean quality and source packability do not establish
wireless improvement. LE2/LE3/LE4 require separate authorization and were not
run or scheduled. This stage stops after its verified publication and backup.
