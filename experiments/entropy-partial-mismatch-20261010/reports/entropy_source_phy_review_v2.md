# EC partial engineering review v2: real-image link gate addendum

Status: read-only design review passed; no blocking inconsistency found in the added finite gate. This addendum preserves v1 and pins the newer ep_plan.py and PROTOCOL.md. No test, model, packet decoder, channel simulation, scientific ranking or real source draw was run for this addendum. The v1 local 17+4 engineering test record is historical evidence, not a test execution against this later plan. Complete calibration/render orchestration remains unimplemented.

## Added finite gate

The gate follows the real first32 source-codec gate and the new 457-call real PHY qualification. It uses the first four sources in the original calibration ordering, SNR 4/10/19 dB and noise seed 4101. Each SNR uses the already frozen original entropy-whole winner's target m and modulation/code rate, with K=0 and its three registered fractional K targets. This fixes 4 sources x 3 SNRs x 4 target actions = 48 logical image frames. Its separate bounds are at most 96 actual packet decoder calls, 48 independent RX VAR source recoveries and 48 reconstructions. The previous 32-source source-only and 457-call random-bit PHY gates are not counted as these real-image frames.

All actual streams, including length fallback, pass through the paid header/body. The receiver uses its received profile, including legal wrong-family/profile events, and reconstructs from received tokens. Stream/source identity and actual fallback, CRC/parser and output records remain in the gate. A rejected channel frame retains the fixed output; a model/software failure stops. Truth diagnostics cannot override receiver acceptance. No quality score or ranking is used to pick or replace the gate actions.

The four sources, candidate actions and seed belong to the following fixed pilot. Reuse is therefore possible only if the original source, exact source stream, actual waveform/noise/counter, public catalogue, decoded profile/body, frozen models/numerical execution and reconstruction event agree. The gate's actual calls remain recorded even when the pilot consumes their sealed results. Equal candidate IDs, source IDs or noise labels alone are insufficient. The v1 warning about CRC-valid unknown old-header outcomes continues to apply.

## Required executor acceptance record

The future gate owner must record target m/K and actual transmitted/received m/K separately, plus the number of accepted positive-K partial end-to-end reconstructions. A fixed gate can encounter channel rejections or fall back to whole endpoints. If all 48 frames reject or all actual accepted frames are whole, its records must be retained but must not be presented as successful qualification of a positive-K physical-to-VAR path. The protocol does not authorize adaptive source selection, extra SNRs or additional trials to obtain a success. This is an implementation acceptance boundary; the bounded 48-frame design itself has no blocker.

The gate's finite caps, deadline, durable reservations, exact-event reuse and real owner/child wait receipts still need implementation and binding before execution. No completion or operational readiness is implied by metadata preparation.
