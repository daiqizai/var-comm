# Same-scale selective recovery: P1 milestone and completed P2 packet probes

This is a partial study delivery. It does not establish wireless net benefit.
The authorized scope is P0–P3 only; P4/P5 and manuscript replacement are not started.
Scientific baseline: a7eb2f3cd1621f819845bdcc1e0de5bb242b793e.
Storage-only parent: e333620bba30b3ba1041dac354506273a9bcfc1d.
Scope SHA: 1e5ce2457ba25b77ad2fa87564fd3be128a3483dfd719812dbedd88f6cb362cf.

## Verified work

- Fixed content-independent 32-source sample from cal1000; original indices retained,
  no common500 or selected-development overlap.
- 32 source probability traversals, zero tokenizer calls. C/A/B are independently
  terminated arithmetic streams; actual lengths are in source_lengths.csv.
- 128 complete source roundtrips, 128 deletion-of-A proofs and 128 missing-C checks.
  B exactly recovers with C alone as context; no source/TX CDF is used in RX.
  Missing C invokes no prior generation and cannot recover B.
- 247 exact prior native RX images reused; 553 continuous and 128 non-contiguous C+B
  reconstructions newly computed. The normal union contains 928 states.
- 128 additional technical C+A+B renderer comparisons match the continuous-prefix
  native float pixels exactly. These comparisons are separate from normal states
  and are not wireless events. Total new decoder calls: 809.
- 960 native source-image pairs scored for DINO and all four reporting metrics:
  928 normal states plus 32 gray pairs. Fresh fixed-device reference features avoid
  historical hardware/numeric cache mixing. PSNR/LPIPS/DreamSim use the current
  frozen 2.11 route; CLIP-I uses the current 2.10 FP32 route. Weights, loaders and
  float preprocessing are unchanged; only the verified seven-file relocation applies.
- All source, codec, image and score owners/children actually exited 0 and are gone;
  ledgers have no unresolved calls. The original closed receipts are included.
- Actual pixel review used the first two hash-ordered sources. Added tokens sometimes
  alter appearance or texture unfavorably; all per-source signed changes are retained.

## P1 quality gate

Difference is C+B minus C. DINO is the fixed gate; positive distances are worse.
No new bootstrap, statistical-significance or independent-test claim is made.

| m | K | Mean delta DINO | Positive / negative sources |
|---|---|---|---|
|4|12|0.053907185967545956|23 / 9|
|4|25|0.030065944156376645|18 / 14|
|5|18|0.035832407884299755|20 / 12|
|5|36|0.06182961747981608|22 / 10|

All four predeclared configurations satisfy the positive-mean criterion.
p1_cb_minus_c_summary.csv contains all five metrics and sign counts;
p1_cb_minus_c_per_source.csv retains 128 source/configuration differences;
p1_clean_metrics.csv maps normal/gray/alias events to exact native pairs.
p1_gate.json joins source, pixel, quality, visual and protocol evidence.

## P2 and outstanding work

All 28 unique LDPC geometries passed their one real noiseless encode/decode roundtrip.
Four separate noiseless paid-header integration checks cover ordinary, NULL, a
different legal actual profile, and an illegal profile. These engineering checks
are counted separately from the noisy probes.

Four P2 CPU roles were uniquely launched for 10,752 noisy body packets and 384
headers at 0/2/4 dB. All four have now exited 0 with complete ledgers and gone processes;
p2_probe_closure.json records closure, with no accepted-but-wrong packets. RX observations/decisions precede TX comparison, and accepted
but wrong outcomes are retained. Packet success plus clean quality is a nomination
proxy, not a complete wireless image result.

Pending: all 220 proxy results and prescribed nominations,
P3 complete frames/source decoding, identical-event ALL/SEQ/SEL replay, all
denominators/costs/failures, SEL−SEQ mechanism difference, and comparison to the
best of at most four actually measured single-block candidates. No P3 frame is
claimed. Clean C+B quality gains do not establish that splitting pays for overhead.

Existing var-comm continuation now follows this scope only. Earlier raw R1, LE,
N512 and WCL results remain unchanged. No old stopped scientific task is resumed.

## Resources and provenance

Before P2 noisy probes, cumulative GPU owner wall time was 0.36073877511080354 device
hours; CPU qualification owner time was 0.010741265730725395 worker hours. Four P2
probes reserve at most 16 worker-hours. Add actual later waits and CPU audit durations
at final closure. Overall ceilings remain 32 GPU-device/32 CPU-worker hours, with
8 hours maximum per stage and tighter worker limits.

Arrays, source streams, logs and checkpoints stay on PFS outside Git. Light records
pin file identities; original registrations stay unchanged. validation/ contains
repository engineering checks, not additional image-quality experiments.
