# N2048 equal-protection full-representation validation

The user's 2026-10-19 “启动吧” authorizes execution of the delivered feasibility
plan at main 5f9baf7c. The historical plan remains unchanged. This new protocol
does not restart or replace the completed N1024 comparison.

Two N1024 subframes each pay 68 header symbols and 956 payload symbols, using
the unchanged single956 LDPC backend at k4780/n5736/q6, rate5/6, nominal Es2.
Each body pays length13, CRC16 and a96-bit reassembly record. Payload capacity
is1162bytes/9296bits; the finished original source object is fragmented after
source coding. The original termination occurs once; both transport lengths,
both outer CRCs and both whole-object CRC32 copies are paid. No shorter-source
fallback or partial-fragment source decoding is allowed. Invalid pairs are gray.

NeST entropy-coded m10 K0 and ARPC native256 e7 share the original external100,
4/10/19dB, three seeds6201/6202/6203 and the identical2048-symbol standard noise.
Subframes use distinct contiguous slices and public scrambler counters. Actual
frame energy is recorded; no per-frame normalization is used. This common-MCS
comparison is not a per-method optimized policy search.

First6 calibration IDs, both codecs and clean/corrupt cases:24pairs/48subframes.
Only12 clean independent neural source decodes/renders are scheduled. Corrupt
tests change one fragment but retain the original whole-object CRC, demonstrating
that the receiver does not use TX truth to detect the error. CPU-only synthetic
edge tests add no physical budget. Actual native outputs and post-exit truth
audit plus root visual/resource review must pass before the formal grid.

Formal cap1800objects/3600subframes; total3648subframes and7296header/body
decodes including qualification. Complete TX objects are reused after SHA checks;
TX probability traversals, tokenization and training remain zero. Unknown errors
are retained and not retried. Model workers only see received assembled bytes,
paid profile and public model settings; truth is read by a separate audit.

All scientific commands are once-only, with persistent call ledger, parent/child
identity, stdout/stderr and actual wait. Registration pins source bytes. New
dependent delivery/scoring entrypoints may be added separately, without editing
these running bindings. Seven original metric versions and a single10000-draw
source-paired bootstrap (seed2026101901) follow the closed formal observations.
No other methods, bandwidth sweep, training, power extension or large backup.
