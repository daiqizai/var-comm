# R0 coverage and supplementary-budget design

This is a metadata audit and design, not an execution authorization or shortlist.

| Family | Profiles | Endpoints | Positive-K endpoints | Unique body layouts | Known layout/SNR pairs | Missing pairs |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 689 | 44 | 36 | 687 | 896 | 1852 |
| S | 433 | 44 | 36 | 433 | 744 | 988 |
| U | 201 | 12 | 9 | 145 | 20 | 560 |
| EEP | 55 | 12 | 9 | 110 | 136 | 304 |
| anchors | 3 | 3 | 2 | 3 | 12 | 0 |

The union has 44 unique endpoints: 8 complete-scale and 36 partial-scale.
The archived clean table has complete scales m=4..8 and gray (under receiver
`gray`, separately from `VAR`); m=1..3 are absent. Positive-K values in that
archive use entropy ordering and cannot represent raster partial tokens.
Conditional on matching the old model/metric/source identities, completing all
current clean-table gaps on cal200 would require 39 * 200 = 7,800 clean VAR
reconstructions and DINO scores. A separately audited raster cache could reduce
this count. For the U/EEP endpoint set alone, 9 * 200 = 1,800 are missing.

Only U profiles 531 and 573 have both group BLER curves present. Their raw-clean
endpoints (m7,K25) and (m6,K48) would cost 400 cal200 clean reconstructions;
including the two positive-K fixed anchors raises this illustrative minimum to
800. This would only evaluate that restricted diagnostic, excluding 199 U
alternatives and every q6 single alternative from comparable screening. It is
not a fair or uniquely justified full-domain nomination.

The full union lacks 1,852 exact layout/SNR points across true SNR 4/7/10/13.
No interpolation, step approximation, or probabilities of zero or one were
inserted. At an illustrative 256 trials per point this would be 474,112 new body
packet trials, but no new sample count or packet budget is authorized here.

The old table describes random scrambled payloads and conditional group
independence, and explicitly leaves image-payload validation incomplete. Its
clean DROP mixture is only a pre-screen approximation; it also does not fully
describe KEEP reconstruction from CRC-rejected actual hard bits. The registered
actual R1 measurements remain necessary. All 689 candidate permissions remain
in the catalogue; no R1 winner or shortlist is selected by this audit.
