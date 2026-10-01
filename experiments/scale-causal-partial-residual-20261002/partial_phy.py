"""Paid raw-prefix/partial-token PHY; no model or source truth enters receive.

Header: m-4 (3 bits), q (7 bits), order (2 bits), CRC16 and six tail
bits, rate-matched to 68 QPSK uses. q=0 is one whole-prefix packet.
q>0 adds a separately protected partial packet. Oracle alone pays an L-bit
bitmap inside its partial CRC domain. All allocated symbols are transmitted.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from var_comm.scale_channel import (
    append_crc, bits_to_indices, convolutional_encode, crc_accepts, decode_map,
    encode_packet, indices_to_bits, rate_match_indices,
)

PROTOCOL = 'raw-partial-token-20261002-v1'
SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
HEADER_USES = 68
ORDERS = ('raster', 'random', 'entropy', 'oracle')
ORDER_CODES = dict(enumerate(ORDERS))
PHY_FAMILIES = ('QPSK', '16QAM')
PAM = np.array([-3., -1., 3., 1.]) / np.sqrt(5.)
LABELS = ((np.arange(4)[:, None] >> np.array([1, 0])) & 1).astype(np.uint8)


def _int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def raw_bits(m):
    m = _int(m, 'm')
    if not 4 <= m <= 8:
        raise ValueError('Whole-prefix domain is m4..m8')
    return 12 * sum(p*p for p in SIZES[:m])


def next_length(m):
    m = _int(m, 'm')
    if not 4 <= m <= 7:
        raise ValueError('Partial-prefix domain is m4..m7')
    return SIZES[m]**2


def _allocation(N, phy, m, q, order):
    N, m, q = _int(N, 'N'), _int(m, 'm'), _int(q, 'q')
    if N not in (512, 1024) or phy not in PHY_FAMILIES:
        raise ValueError('Registered budget and PHY required')
    if order not in (*ORDERS, 'whole') or q < 0:
        raise ValueError('Registered order and nonnegative q required')
    if q == 0:
        order = 'whole'
    elif order == 'whole' or not 4 <= m <= 7 or not 1 <= q < next_length(m):
        raise ValueError('Partial q must lie in 1..L-1 at m4..m7')
    payload = raw_bits(m)
    body, width = N-HEADER_USES, 2 if phy == 'QPSK' else 4
    prefix_info = payload + 22
    mask = next_length(m) if q and order == 'oracle' else 0
    partial_payload = 12*q + mask
    partial_info = partial_payload+22 if q else 0
    # Integer ceilings enforce the original rate <= .9 exactly. Allocation
    # uses round-half-up of the information-length ratio, then legal clamping.
    minimum_prefix = (10*prefix_info+9*width-1)//(9*width)
    minimum_partial = (10*partial_info+9*width-1)//(9*width) if q else 0
    eligible = minimum_prefix+minimum_partial <= body
    if q and eligible:
        denominator = prefix_info+partial_info
        proposed = (2*body*prefix_info+denominator)//(2*denominator)
        prefix_uses = min(max(proposed, minimum_prefix), body-minimum_partial)
    else:
        prefix_uses = body if not q else minimum_prefix
    partial_uses = body-prefix_uses
    return dict(protocol=PROTOCOL, N=N, phy_family=phy, m=m, q=q, order=order,
        action_m=m, action_q=q, N_header=HEADER_USES, N_body=body,
        N_prefix=prefix_uses, N_partial=partial_uses,
        next_scale_tokens=SIZES[m]**2 if m < 10 else 0,
        prefix_source_bits=payload, partial_token_bits=12*q, mask_bits=mask,
        partial_source_bits=partial_payload, token_payload_bits=payload+12*q,
        source_bits=payload+partial_payload,
        prefix_information_bits=prefix_info, partial_information_bits=partial_info,
        prefix_mother_bits=2*prefix_info, partial_mother_bits=2*partial_info,
        header_source_bits=12, header_mode_bits=3, header_q_bits=7,
        header_order_bits=2, header_class_bits=0, header_crc_bits=16,
        header_tail_bits=6, header_mother_bits=68, header_coded_bits=136,
        body_crc_bits=16*(1+bool(q)), body_tail_bits=6*(1+bool(q)),
        prefix_coded_bits=width*prefix_uses, partial_coded_bits=width*partial_uses,
        body_coded_bits=width*body, coded_bits=136+width*body,
        total_coded_bits=136+width*body,
        prefix_effective_rate=prefix_info/(width*prefix_uses),
        partial_effective_rate=partial_info/(width*partial_uses) if q and partial_uses > 0 else None,
        eligible=eligible, exclusion_reason='' if eligible else 'segment_information_or_rate_cap',
        modulation=phy, energy_constraint='per_frame_2N' if phy=='QPSK' else 'fixed_constellation_actual_frame_energy')


@dataclass(frozen=True)
class Action:
    N: int
    phy: str
    m: int
    q: int
    order: str

    def __post_init__(self):
        rec = _allocation(self.N, self.phy, self.m, self.q, self.order)
        if not rec['eligible']:
            raise ValueError('Action exceeds a segment information/rate limit')
        if self.q == 0:
            object.__setattr__(self, 'order', 'whole')

    @property
    def name(self):
        return f'{self.phy}/N{self.N}/m{self.m}/q{self.q}/{self.order}'

    def to_dict(self):
        return dict(N=int(self.N), phy=self.phy, m=int(self.m), q=int(self.q), order=self.order)


def action_record(action):
    if not isinstance(action, Action):
        raise ValueError('Action object required')
    return dict(action_id=action.name, **_allocation(**action.to_dict()))


def max_legal_q(N, phy, m, order):
    if order not in ORDERS:
        raise ValueError('Partial order required')
    for q in range(next_length(m)-1, 0, -1):
        if _allocation(N, phy, m, q, order)['eligible']:
            return q
    return 0


def action_grid(N, phy, orders=ORDERS, include_whole=True):
    """Frozen finite grid: whole m4..m8; partial quarters and max legal q."""
    result = {}
    if include_whole:
        for m in range(4, 9):
            if _allocation(N, phy, m, 0, 'whole')['eligible']:
                action = Action(N, phy, m, 0, 'whole'); result[action.name] = action
    for order in orders:
        if order not in ORDERS:
            raise ValueError('Unregistered partial order')
        for m in range(4, 8):
            length = next_length(m); maximum = max_legal_q(N, phy, m, order)
            for q in sorted({length//4, length//2, 3*length//4, maximum}):
                if q > 0 and q <= maximum:
                    action = Action(N, phy, m, q, order); result[action.name] = action
    return list(result.values())


def waveform_sha(wave):
    return hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()


def _binary(bits):
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.isin(bits, (0, 1)).all():
        raise ValueError('Flat binary payload required')
    return bits.astype(np.uint8)


def modulate(bits, phy):
    bits = _binary(bits)
    width = 2 if phy == 'QPSK' else 4 if phy == '16QAM' else 0
    if not width or len(bits) % width:
        raise ValueError('Complete registered constellation symbols required')
    if phy == 'QPSK':
        return (1.-2.*bits.astype(float)).reshape(-1, 2)
    pairs = bits.reshape(-1, 2)
    return PAM[2*pairs[:, 0]+pairs[:, 1]].reshape(-1, 2)


def soft_half_llr(y, snr, phy):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y).all() or not np.isfinite(snr):
        raise ValueError('Finite received coordinates and nominal SNR required')
    gamma = 10.**(float(snr)/10)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Finite positive noise precision required')
    if phy == 'QPSK':
        return (y*gamma).ravel()
    if phy != '16QAM':
        raise ValueError('PHY family')
    logp = -(y.ravel()[:, None]-PAM[None, :])**2*gamma/2
    return np.stack([.5*(np.logaddexp.reduce(logp[:, LABELS[:, bit]==0], axis=1)
                         -np.logaddexp.reduce(logp[:, LABELS[:, bit]==1], axis=1))
                     for bit in range(2)], axis=1).ravel()


def packet(bits, uses, phy):
    bits = _binary(bits); info = np.concatenate((append_crc(bits), np.zeros(6, dtype=np.uint8)))
    slots = uses*(2 if phy=='QPSK' else 4)
    if uses <= 0 or 10*len(info) > 9*slots:
        raise ValueError('Packet information/rate cap exceeded')
    mother = convolutional_encode(info)
    return modulate(mother[rate_match_indices(len(mother), slots)], phy)


def decode_packet(y, length, snr, phy):
    info = int(length)+22; width = 2 if phy=='QPSK' else 4
    slots = len(y)*width
    if length < 1 or 10*info > 9*slots:
        raise ValueError('Invalid received payload length or segment rate')
    mapping = rate_match_indices(2*info, slots)
    evidence = np.bincount(mapping, weights=soft_half_llr(y, snr, phy), minlength=2*info)
    decoded, score = decode_map(evidence)
    return decoded[:length], bool(crc_accepts(decoded[:-6])), float(score)


def _scales(scales, count):
    if len(scales) < count:
        raise ValueError('Source prefix missing scales')
    result = []
    for k, scale in enumerate(scales[:count]):
        scale = np.asarray(scale)
        if scale.shape != (SIZES[k]**2,) or not np.issubdtype(scale.dtype, np.integer) or np.any(scale<0) or np.any(scale>=4096):
            raise ValueError('Original 12-bit codebook tokens required')
        result.append(scale.astype(np.int64))
    return result


def transmit(scales, action, positions=None):
    """TX source truth is allowed here. Entropy/oracle positions are TX inputs."""
    ledger = action_record(action); scales = _scales(scales, action.m+bool(action.q))
    code = 0 if not action.q else ORDERS.index(action.order)
    header = np.concatenate((indices_to_bits([action.m-4], 3),
                             indices_to_bits([action.q], 7), indices_to_bits([code], 2)))
    prefix = indices_to_bits(np.concatenate(scales[:action.m]))
    blocks = [encode_packet(header, HEADER_USES)['symbols'],
              packet(prefix, ledger['N_prefix'], action.phy)]
    if action.q:
        if positions is None and action.order == 'raster':
            positions = np.arange(action.q)
        positions = np.asarray(positions)
        length = next_length(action.m)
        if positions.shape != (action.q,) or not np.issubdtype(positions.dtype, np.integer) or len(set(positions.tolist())) != action.q or np.any(positions<0) or np.any(positions>=length):
            raise ValueError('Unique actual selected positions required')
        if action.order == 'oracle':
            positions = np.sort(positions); mask = np.zeros(length, dtype=np.uint8); mask[positions] = 1
            payload = np.concatenate((mask, indices_to_bits(scales[action.m][positions])))
        else:
            payload = indices_to_bits(scales[action.m][positions])
        blocks.append(packet(payload, ledger['N_partial'], action.phy))
    elif positions is not None and np.asarray(positions).size:
        raise ValueError('Whole prefix cannot carry partial positions')
    wave = np.concatenate(blocks)
    if wave.shape != (action.N, 2) or not np.isfinite(wave).all():
        raise ValueError('Exact finite full-budget waveform required')
    energy = float(np.square(wave, dtype=np.float64).sum())
    if action.phy == 'QPSK' and energy != 2*action.N:
        raise ValueError('QPSK per-frame energy mismatch')
    return wave, dict(**ledger, E=energy, energy=energy)


def apply_channel(wave, snr, noise):
    """The runner supplies one paired standard-normal realization, with no gain."""
    wave, noise = np.asarray(wave), np.asarray(noise)
    if wave.ndim != 2 or wave.shape[1] != 2 or noise.shape != wave.shape or not np.isfinite(wave).all() or not np.isfinite(noise).all() or not np.isfinite(snr):
        raise ValueError('Finite matching full-budget waveform/noise required')
    return wave+noise*10.**(-float(snr)/20)


def receive(y, snr, N, phy):
    """No true action, source, class, position, feedback or clean tokens."""
    N = _int(N, 'N'); y = np.asarray(y, dtype=np.float64)
    if N not in (512, 1024) or phy not in PHY_FAMILIES or y.shape != (N, 2) or not np.isfinite(y).all() or not np.isfinite(snr):
        raise ValueError('Complete received waveform and declared budget/PHY required')
    bits, crc, score = decode_packet(y[:HEADER_USES], 12, snr, 'QPSK')
    m = 4+int(bits_to_indices(bits[:3], 3)[0]); q = int(bits_to_indices(bits[3:10], 7)[0])
    code = int(bits_to_indices(bits[10:12], 2)[0]); order = ORDER_CODES[code] if q else 'whole'
    action = None
    try:
        action = Action(N, phy, m, q, order)
        legal = bool(q or code == 0)
    except ValueError:
        legal = False
    accepted = bool(crc and legal)
    event = dict(header_ok=accepted, header_crc_ok=crc, header_fields_legal=legal,
        header_score=float(score), action=action if accepted else None,
        decoded_m=m if accepted else None, decoded_q=q if accepted else None,
        decoded_order=order if accepted else None, prefix=[], prefix_crc_ok=False,
        partial_values=np.empty(0, dtype=np.int64), partial_positions=None,
        partial_crc_ok=None, partial_fields_legal=False, partial_usable=False,
        body_crc_ok=False, trusted_prefix_scales=0, hard_candidate_prefix_scales=0,
        raw_candidate_used_after_crc_failure=False, source_complete=False,
        partial_discard_reason='header_failure' if not accepted else '',
        source_error='header_failure_or_illegal_received_action' if not accepted else '')
    if not accepted:
        return event
    rec = action_record(action); boundary = HEADER_USES+rec['N_prefix']
    payload, prefix_crc, prefix_score = decode_packet(y[HEADER_USES:boundary], rec['prefix_source_bits'], snr, phy)
    tokens = bits_to_indices(payload); lengths = [p*p for p in SIZES[:m]]
    event.update(prefix=list(np.split(tokens, np.cumsum(lengths)[:-1])),
        prefix_crc_ok=prefix_crc, prefix_score=float(prefix_score),
        trusted_prefix_scales=m if prefix_crc else 0, hard_candidate_prefix_scales=m,
        raw_candidate_used_after_crc_failure=not prefix_crc, source_complete=True)
    if q:
        part, partial_crc, partial_score = decode_packet(y[boundary:], rec['partial_source_bits'], snr, phy)
        positions = np.flatnonzero(part[:rec['mask_bits']]) if rec['mask_bits'] else None
        values = bits_to_indices(part[rec['mask_bits']:])
        fields_legal = len(values)==q and (positions is None or len(positions)==q)
        usable = bool(prefix_crc and partial_crc and fields_legal)
        reason = 'prefix_crc_failure' if not prefix_crc else 'partial_crc_failure' if not partial_crc else 'partial_fields_illegal' if not fields_legal else ''
        event.update(partial_values=values, partial_positions=positions, partial_crc_ok=partial_crc,
            partial_score=float(partial_score), partial_fields_legal=fields_legal,
            partial_usable=usable, partial_discard_reason=reason,
            body_crc_ok=bool(prefix_crc and partial_crc and fields_legal))
    else:
        event.update(body_crc_ok=prefix_crc, partial_fields_legal=True,
            partial_discard_reason='not_transmitted')
    if not prefix_crc:
        event['source_error'] = 'unverified_hard_prefix; partial_discarded'
    return event


def event_fields(event):
    """JSON/CSV-safe physical decisions; hard candidates remain in the event."""
    names = ('header_ok', 'header_crc_ok', 'header_fields_legal', 'decoded_m', 'decoded_q',
        'decoded_order', 'prefix_crc_ok', 'partial_crc_ok', 'partial_fields_legal',
        'partial_usable', 'body_crc_ok', 'trusted_prefix_scales',
        'hard_candidate_prefix_scales', 'raw_candidate_used_after_crc_failure',
        'source_complete', 'partial_discard_reason', 'source_error')
    return {k:event[k] for k in names}


def registration():
    return dict(protocol=PROTOCOL, budgets=[512, 1024], phy_families=list(PHY_FAMILIES),
        header_uses=68, header_fields=dict(m=3, q=7, order=2), class_bits=0,
        whole_modes=[4, 5, 6, 7, 8], partial_prefix_modes=[4, 5, 6, 7],
        partial_q='1..L-1; floor25/50/75percent and maximum legal, deduplicated',
        q0='whole prefix, one body CRC16+tail6, canonical order code0',
        allocation='information-length ratio, round-half-up, clamp both segment rates<=.9',
        oracle='L-bit raster bitmap inside partial CRC; token values in ascending bitmap positions',
        header_failure='RGB .5; no latent',
        prefix_failure='retain actual hard prefix, discard partial for every order; no feedback/fallback sorting',
        partial_failure='discard all partial tokens; complete from actual hard prefix',
        generation='frozen unconditional class1000, deterministic argmax, one prediction per scale',
        QPSK_energy='per-frame 2N', QAM_energy='original fixed PAM/sqrt5, actual frame energy')
