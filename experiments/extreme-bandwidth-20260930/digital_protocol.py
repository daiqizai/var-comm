"""Registered N512 raw PHY: paid 68-symbol header, two-bit m4..m7 modes.

The terminated convolutional code and rate matcher are the existing project
implementation. Raw CRC failures retain their decoded hard candidate prefix.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from var_comm.scale_channel import (
    append_crc, bits_to_indices, convolutional_encode, crc_accepts, decode_map,
    encode_packet, indices_to_bits, rate_match_indices,
)

PROTOCOL = 'extreme-bandwidth-raw-N512-v1'
SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
N = 512
HEADER_USES = 68
MODE_MAPPING = {0: 4, 1: 5, 2: 6, 3: 7}
PHY_FAMILIES = ('QPSK', '16QAM')
PAM = np.array([-3., -1., 3., 1.]) / np.sqrt(5.)
LABELS = ((np.arange(4)[:, None] >> np.array([1, 0])) & 1).astype(np.uint8)


def raw_bits(m):
    if m not in MODE_MAPPING.values():
        raise ValueError('Registered source domain is m4..m7')
    return 12 * sum(p * p for p in SIZES[:m])


def action_record(m, phy):
    if phy not in PHY_FAMILIES:
        raise ValueError('Predeclared QPSK or 16QAM family required')
    body = N - HEADER_USES
    bits = raw_bits(m)
    slots = body * (2 if phy == 'QPSK' else 4)
    rate = (bits + 22) / slots
    legal = bits + 22 <= slots and rate <= .9
    return dict(action_id=f'{phy}/N{N}/m{m}', action_m=m, phy_family=phy,
        N=N, N_header=HEADER_USES, N_body=body, source_bits=bits,
        coded_bits=slots+136, body_coded_bits=slots, total_coded_bits=slots+136,
        mother_bits=2 * (bits + 22), header_mother_bits=68, effective_code_rate=rate,
        eligible=legal, exclusion_reason='' if legal else 'information_or_registered_rate_cap',
        body_crc_bits=16, body_tail_bits=6, header_source_bits=12,
        header_class_bits=10, header_mode_bits=2, header_coded_bits=136)


@dataclass(frozen=True)
class Action:
    m: int
    phy: str
    N: int = N

    def __post_init__(self):
        if self.N != N or not action_record(self.m, self.phy)['eligible']:
            raise ValueError('Unregistered or unencodable N512 action')

    @property
    def name(self):
        return action_record(self.m, self.phy)['action_id']


def legal_actions():
    return [Action(m, phy) for phy in PHY_FAMILIES for m in MODE_MAPPING.values()
            if action_record(m, phy)['eligible']]


def registration():
    return dict(protocol=PROTOCOL, N=N, N_header=HEADER_USES, N_body=N-HEADER_USES,
        header_mode_mapping={str(k): v for k, v in MODE_MAPPING.items()},
        header_class_source='existing application/ImageNet class_index; paid 10 bits',
        unconditional='same paid header; ignore decoded class for generation, class_emb index1000',
        phy_families='separately registered and known to transmitter/receiver; no MCS bit',
        raw_bits={str(m): raw_bits(m) for m in MODE_MAPPING.values()},
        actions=[action_record(m, phy) for phy in PHY_FAMILIES for m in MODE_MAPPING.values()],
        code='original terminated rate1/2 convolutional mother code, CRC16+6 tail, midpoint rate matcher',
        rate_cap=.9, header_failure='constant RGB .5; no prefix or generation',
        body_crc_failure='preserve unverified hard ML raw prefix, trusted_prefix_scales=0',
        completion='original ten-scale cumulative VAR, deterministic argmax, no CFG/sampling',
        energy=dict(QPSK='actual per-frame E=2N',
                    **{'16QAM':'original fixed PAM/sqrt5; actual frame E, no gain or normalization'}))


def waveform_sha(wave):
    return hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()


def _binary(bits):
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.isin(bits, (0, 1)).all():
        raise ValueError('Flat binary payload required')
    return bits.astype(np.uint8)


def modulate(bits, phy):
    """Exact existing QPSK or Gray 16QAM fixed constellation."""
    bits = _binary(bits)
    width = 2 if phy == 'QPSK' else 4 if phy == '16QAM' else 0
    if not width or len(bits) % width:
        raise ValueError('Complete registered constellation symbols required')
    if phy == 'QPSK':
        return (1. - 2. * bits.astype(float)).reshape(-1, 2)
    pairs = bits.reshape(-1, 2)
    return PAM[2 * pairs[:, 0] + pairs[:, 1]].reshape(-1, 2)


def soft_half_llr(y, snr, phy):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y).all() or not np.isfinite(snr):
        raise ValueError('Finite received complex coordinates and nominal SNR required')
    gamma = 10. ** (float(snr) / 10)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Finite positive noise precision required')
    if phy == 'QPSK':
        return (y * gamma).ravel()
    if phy != '16QAM':
        raise ValueError('PHY family')
    logp = -(y.ravel()[:, None] - PAM[None, :]) ** 2 * gamma / 2
    result = np.stack([.5 * (np.logaddexp.reduce(logp[:, LABELS[:, bit] == 0], axis=1)
                         - np.logaddexp.reduce(logp[:, LABELS[:, bit] == 1], axis=1))
                       for bit in range(2)], axis=1).ravel()
    if not np.isfinite(result).all():
        raise FloatingPointError('Soft demodulation')
    return result


def packet(bits, uses, phy):
    bits = _binary(bits)
    info = np.concatenate((append_crc(bits), np.zeros(6, dtype=np.uint8)))
    slots = uses * (2 if phy == 'QPSK' else 4)
    if len(info) > slots:
        raise ValueError('Information plus CRC/tail exceeds transmitted slots')
    mother = convolutional_encode(info)
    return modulate(mother[rate_match_indices(len(mother), slots)], phy)


def decode_packet(y, length, snr, phy):
    info = int(length) + 22
    slots = len(y) * (2 if phy == 'QPSK' else 4)
    if length < 1 or info > slots:
        raise ValueError('Invalid received payload length')
    mapping = rate_match_indices(2 * info, slots)
    evidence = np.bincount(mapping, weights=soft_half_llr(y, snr, phy), minlength=2 * info)
    decoded, score = decode_map(evidence)
    return decoded[:length], bool(crc_accepts(decoded[:-6])), float(score)


def transmit(scales, label, action):
    """Transmitter truth is permitted here; never passed to receive()."""
    if not isinstance(action, Action) or not isinstance(label, (int, np.integer)) or not 0 <= label < 1000:
        raise ValueError('Registered action and paid class source required')
    if len(scales) < action.m:
        raise ValueError('Source prefix missing scales')
    values = []
    for i, scale in enumerate(scales[:action.m]):
        scale = np.asarray(scale)
        if scale.shape != (SIZES[i] ** 2,) or not np.issubdtype(scale.dtype, np.integer) or np.any(scale < 0) or np.any(scale >= 4096):
            raise ValueError('Original 12-bit codebook indices required')
        values.append(scale)
    payload = indices_to_bits(np.concatenate(values))
    header = np.concatenate((indices_to_bits([label], 10), indices_to_bits([action.m - 4], 2)))
    wave = np.concatenate((encode_packet(header, HEADER_USES)['symbols'],
                           packet(payload, N - HEADER_USES, action.phy)))
    if wave.shape != (N, 2) or not np.isfinite(wave).all():
        raise ValueError('Complete finite N512 waveform required')
    energy = float(np.square(wave, dtype=np.float64).sum())
    if action.phy == 'QPSK' and energy != 2 * N:
        raise ValueError('QPSK per-frame energy mismatch')
    ledger = dict(**action_record(action.m, action.phy), protocol=PROTOCOL, E=energy,
        energy=energy, modulation=action.phy,
        energy_constraint='per_frame_2N' if action.phy == 'QPSK' else 'fixed_constellation_average_2_per_symbol')
    return wave, ledger


def apply_channel(wave, snr, source_id, seed, phy):
    from var_comm.study import seeded_noise
    wave = np.asarray(wave)
    if wave.shape != (N, 2) or phy not in PHY_FAMILIES or not np.isfinite(wave).all() or not np.isfinite(snr):
        raise ValueError('Complete registered waveform and finite nominal SNR required')
    noise = seeded_noise(f'{PROTOCOL}/{phy}/N{N}|{source_id}', int(seed), wave.shape)
    return wave + noise * 10. ** (-float(snr) / 20)


def receive(y, snr, phy):
    """Only y, nominal SNR and predeclared PHY; m comes from received bits."""
    y = np.asarray(y, dtype=float)
    if y.shape != (N, 2) or phy not in PHY_FAMILIES or not np.isfinite(y).all() or not np.isfinite(snr):
        raise ValueError('Complete received N512 waveform required')
    header, crc, hscore = decode_packet(y[:HEADER_USES], 12, snr, 'QPSK')
    label = int(bits_to_indices(header[:10], 10)[0])
    code = int(bits_to_indices(header[10:12], 2)[0])
    mode = MODE_MAPPING[code]
    legal = label < 1000 and action_record(mode, phy)['eligible']
    accepted = bool(crc and legal)
    result = dict(header_ok=accepted, header_crc_ok=crc, header_fields_legal=bool(legal),
        header_score=hscore, decoded_label=None, decoded_mode=None, body_crc_ok=False,
        source_complete=False, trusted_prefix_scales=0, hard_candidate_prefix_scales=0,
        raw_candidate_used_after_crc_failure=False, source_error='' if accepted else 'header_failure_or_illegal_received_action',
        prefix=[], source_bits=None)
    if not accepted:
        return result
    bits, body_crc, score = decode_packet(y[HEADER_USES:], raw_bits(mode), snr, phy)
    tokens = bits_to_indices(bits)
    lengths = [p * p for p in SIZES[:mode]]
    prefix = list(np.split(tokens, np.cumsum(lengths)[:-1]))
    result.update(decoded_label=label, decoded_mode=mode, body_crc_ok=body_crc,
        body_score=score, source_complete=True, source_bits=bits, prefix=prefix,
        trusted_prefix_scales=mode if body_crc else 0, hard_candidate_prefix_scales=mode,
        raw_candidate_used_after_crc_failure=not body_crc)
    return result
