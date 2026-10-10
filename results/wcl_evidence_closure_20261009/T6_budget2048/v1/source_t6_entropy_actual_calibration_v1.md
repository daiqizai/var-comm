# T6 N2048 entropy calibration execution contract

This is a separate N2048 experiment. It does not change any T1 result, receiver,
source stream, or frozen N1024 policy. Real execution is scheduled by the sole
experiment owner after the N1024 quality and online-cost conclusion.

## Source and receiver protocol

The one family is `EC_VAR_WHOLE`, selected once from N1024 full-calibration
results by the separately sealed family-selection rule. No confirmation image
or score enters family or policy selection. The null class, VAR/VAE/Dc weights,
24-bit CDFs, integer arithmetic codec, canonical termination, and 30-bit terminal
lookahead remain unchanged. The independent `SourceCodec10` extends the same
algorithm to all 680 tokens. Existing m4..9 streams must remain byte-identical;
m10 receives an independent receiver roundtrip. No raw substitution is allowed.

Each frame pays 68 header symbols and 1980 body symbols. The source length stays
13 bits, with CRC16 over length, source payload and known information padding.
The source capacity is `min(k - 29, 8191)`; no k truncation and no implicit L14.
Targets are m7/8/9/10, q2/4/6, rates 1/2, 2/3, 3/4, 5/6: 48 resource queries.
Actual pinned Sionna constructors determine which layouts are supported before
quality is read. Each admitted layout retains every target candidate and every
actual received-prefix header m4..10. Unsupported queries remain in the
admission CSV. They are not silently approximated.

The receiver parses the actual received header against the full admitted
catalogue. A wrong accepted header can therefore select a different actual m.
An exact source-cache hit requires received bit equality and the frozen source
model identity. A miss runs independent source decoding with the received m;
source truth never decides acceptance. Header, CRC, framing or canonical source
rejection yields fixed RGB 0.5. Missing reachable source assets stop execution.

## Finite selection and accounting

- Qualification: for L actually admitted layouts, 8L body cases plus 7L known
  headers plus one unknown header, exactly 15L + 1 packet-decoder calls.
- Pilot: all admitted candidates C, the first 100 original calibration sources,
  SNR 4/10/19, seed 4101; 300C logical frames, maximum 600C packet calls.
- Full: pilot top three per SNR by source-mean DINOv2-L, lexical candidate ties,
  all 1000 original calibration sources, seeds 4101/4102/4103; 27000 logical
  frames, maximum 54000 packet calls.
- Source fallback descends from target to m4 by actual coded length only.
  Candidates sharing the same actual transmitted profile/source/noise share
  one exact physical reception. Full reuses its own N2048 pilot only after
  payload, counter, transmitted frame and observation equality checks.
- Old N1024 receptions are never relabelled as N2048. Exact same-source received
  token state may reuse a sealed old image and score under the original model,
  numerical and scoring identities.
- `--t1-completion` additionally admits completed T1 calibration-only caches
  under the same source, renderer, scorer and numerical identities. Every reused
  image has an actual received-state proof. Source-reference metric features are
  persisted with SHA seals; restarting a source does not recompute them, and an
  unresolved reference-preparation reservation stops automatic recovery.
- After actual CPU receptions, a separate read-only pass freezes GPU counts.
  Known source-cache states and unknown received payloads are distinguished.
  Unknown payloads receive a bounded independent decode allowance. Each new
  source decode/render is reserved before execution; unresolved calls require
  explicit inspection. One or two GPU workers are preregistered. In the
  two-worker mode, each worker has disjoint source indices and must reproduce
  an original image exactly, with overlapping qualification compute intervals.

The N2048 raw and entropy methods use the same source/seed keyed standard normal
array and the same numerical noise scaling. Different codewords still produce
different received observations. Confirmation uses its separately registered
seeds 9201/9202/9203 and may only start after all policies are frozen.

## Commands

All paths are explicit owner-supplied values. Qualification and CPU commands use
`CUDA_VISIBLE_DEVICES=''`; source/render owners use the original admitted GPU
Python and `CUDA_VISIBLE_DEVICES=0`. The owner maintains shared visual locks,
original numerical flags, process waits and exit receipts.

```text
t6_qualify_entropy_phy.py prepare --root ROOT --entropy-family-selection FAMILY --out QUAL --deadline-unix DEADLINE
t6_qualify_entropy_phy.py run --request QUAL/request.json
t6_calibration_sources.py prepare --root ROOT --family-freeze FAMILY --source-completion T1_SOURCE1000 --environment-request T2_REQUEST --count 100 --out SOURCE100 --deadline-unix DEADLINE
t6_calibration_sources.py run --request SOURCE100/request.json
t6_entropy_calibrate.py prepare --source-completion SOURCE100/completion.json --entropy-family-selection FAMILY --phy-qualification QUAL/completion.json --environment-request T2_REQUEST --t2-completion T2_COMPLETE --t1-completion T1_FULLCAL_COMPLETE --phase pilot --packet-cap EXACT_PILOT_CAP --workers 8 --gpu-workers 2 --out PILOT --deadline-unix DEADLINE
t6_entropy_calibrate.py ledger-init --request PILOT/execution_request.json
t6_entropy_calibrate.py cpu-worker --request PILOT/execution_request.json --index WORKER_INDEX
t6_entropy_calibrate.py plan-gpu --request PILOT/execution_request.json
t6_entropy_gpu_parallel.py owner --request PILOT/execution_request.json --session gpu_v1
t6_entropy_calibrate.py close --request PILOT/execution_request.json --owner-receipt PILOT/gpu_cohorts/gpu_v1/completion.json
```

`EXACT_PILOT_CAP = 600 * len(qualification_request['admitted_candidates'])`.
The CPU owner starts each registered index exactly once and waits for all exits.
For full calibration, extend sources to 1000 with `--prior-completion` pointing
to the completed first100 source stage, then prepare `--phase full
--pilot-completion PILOT/completion.json --packet-cap 54000` in a fresh output.
Only exact source identities, same qualification and same scoring environment
can be reused. Every completed stage seals actual artifacts and ledger totals.

Local synthetic verification runs no actual images, models or channel decoder.
It covers paid uint13 capacity, wrong received m10 dispatch, canonical failures,
complete candidate admission, missing-asset stop, physical-prefix reuse, actual
SQLite reservation counts and CPU/GPU checkpoint resume.
