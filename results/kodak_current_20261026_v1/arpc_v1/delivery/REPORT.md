# Kodak ARPC lane: source, physical link and independent RX

## Completed scope

The current frozen ARPC adaptation was evaluated on the existing 24 Kodak
256×256 center crops. Source indices 0–23 and the original source pixels are
unchanged. The parent population and scope SHA are pinned in registration.json.
This lane contains no new training, calibration, quality scoring or bootstrap.

The budget is N=1024 complex symbols (68 header + 956 body), AWGN, Es=2.
SNRs are 4, 10 and 19 dB with original stable noise seeds 2001/2002/2003.
The frozen target policies are respectively e7_q2_r2-3, e7_q4_r2-3 and
e7_q6_r5-6. Actual terminated payload lengths, paid caption/container fields,
length/CRC/alignment and actual received information determine fallback.
Official ARPC weights, native coding mathematics, τ=0.5, cfg=3 and generation
seed 0 are unchanged. No transmitted truth was supplied to independent RX.

## Actual computation and denominators

| Quantity | Actual count |
|---|---:|
| Caption inference | 24 |
| Tokenizer inference | 24 |
| TX probability traversals | 24 |
| Independently terminated e1–e7 source streams | 168 |
| TX clean endpoint renders | 0 |
| Logical physical-link events | 216 |
| Header decodes / body decodes | 216 / 216 |
| Distinct actual RX payloads | 72 |
| Independent prefix decodes / native renders | 72 / 72 |
| Additional logical placements using identical RX outputs | 144 |
| Unresolved native calls / failed native calls | 0 / 0 |
| Repeated successful scientific calls | 0 |

Source native ledger: 74/74 complete, including two model loads. RX native
ledger: 145/145 complete, including one model load. All 216 logical frames
remain in the denominator and have PAYLOAD_PARSED status; no gray output is
used in this realized grid. A fixed gray artifact is included only as the
original finalizer's defined fallback material and carries no logical weight.

| SNR (dB) | Logical frames | Actual RX endpoint counts |
|---|---:|---|
| 4 | 72 | e3:18, e4:54 |
| 10 | 72 | e5:72 |
| 19 | 72 | e6:72 |

The 216-row frame_manifest.csv records native float paths, NPZ hashes, raw
float array hashes, exact noise identifiers and endpoints. Its array convention
is `sha256(array.tobytes(C)); float32 CHW [3,256,256]`, key `float_output`.
No old PSNR/LPIPS values were available for these new ARPC outputs; their fields
are blank and are left to the separately controlled unified metrics lane.

## Actual execution and retained limitations

Native source completed with exit 0 in 87.433910 s; native RX completed with
exit 0 in 106.719163 s. Final CPU delivery completed with exit 0 in 1.373398 s.
The per-stage supervisor and child PID/start identities were checked on their
actual execution hosts. cross_host_closure_v2.json and actual_closure.json
record this evidence; elapsed times are execution receipts, not timing results.

1. The first source admission rejected GPU utilization 36% against the original
   25% guard before native launch or scientific calls. Its failure is preserved.
   An independently registered shared-inference continuation raised only that
   admission limit to 80%, retaining 64 GiB free admission, allocator and running
   free-memory limits, models and mathematics. It performed the first source
   invocation and the sole RX invocation; no model work was replayed.
2. PHY transmission exited 0. All 216 header/body decode reservations and all
   packet files were complete, but original PHY RX exited 1 at its final
   denied-attempt assertion. The denied-path list was in memory and is lost;
   it is not reported as empty. The previously reviewed saved-packet finalizer
   was reused through exact pure-helper ASTs. It verified all saved packets and
   actual payloads and performed a post-exit audit with zero repeated decodes.
   The frozen audit hook raises before an outside-allowlist open returns data.
   This establishes the enforced I/O boundary, not that no attempts occurred.
3. Native RX's preserved completion records 100 denied opens, all of the exact
   frozen `arpc_v1/native.py` program-source path. No source-image or TX-truth
   path occurs in that retained list. The original hook AST is unchanged and
   immediately raises after recording each denied path; it returned no file
   contents. The original attempt-free gate did not pass. The exact list and
   original outputs were not modified. A finite CPU-only finalization accepts
   only this observed 100-entry code-path list and records this limitation.
   Warning/linecache inspection is a possible cause, not an observed cause.
4. The first delivery CPU guard exited 1 before output creation because it tried
   to inspect a MAIN-host PID from PRO. That failed receipt is preserved. The
   independent CPU continuation consumes a fresh, SHA-bound MAIN-host gone
   proof only for the exact PHY identities; PRO identities are checked locally.
   No scientific work was repeated.

CPU engineering qualification: original population adapter 10 tests, shared
resource continuation 4, saved-packet finalization 7, exact native-code-denial
finalization 5, exact cross-host closure 4; all completed with exit 0. A prior
saved-packet test harness import failure is retained separately and did not
execute scientific work. Qualification is not a quality experiment.

## Handoff

See frame_manifest.csv, completion.json, actual_closure.json,
receiver_boundary_review.json and crosshost_closure_review.json. Full logs,
model outputs and native arrays remain at the registered PFS runtime roots.
The parent task handles four-metric evaluation, manuscript use and Git
publication. This ARPC source/link/RX lane does not claim the full study or its
quality statistics are delivered.
