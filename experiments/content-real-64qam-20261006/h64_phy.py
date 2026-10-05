"""Exact H wire framing, fixed-energy Gray QAM and metered actual packet RX.

Truth comparisons are performed by the caller after receiving these outcomes.
CRC acceptance and source parser acceptance remain separate observations.
"""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from h64_catalog import PROTOCOL, SIZES, digest, require, token_count

HEADER_PHY_KEY = 'H-profile12-crc16-tail6-conv-ratematch136-qpsk-Es2'


def binary(bits):
    a = np.asarray(bits)
    require(a.ndim == 1 and np.isin(a, (0, 1)).all(), 'A binary vector is required')
    return a.astype(np.uint8, copy=True)


def integer_bits(value, width):
    require(type(value) is int and 0 <= value < (1 << width), 'Integer does not fit field')
    return ((value >> np.arange(width-1, -1, -1)) & 1).astype(np.uint8)


def bits_integer(bits):
    a = binary(bits)
    return sum(int(x) << (len(a)-1-i) for i, x in enumerate(a))


def crc16(bits):
    v = 65535
    for bit in binary(bits):
        feedback = (v >> 15) ^ int(bit)
        v = ((v << 1) ^ (0x1021 if feedback else 0)) & 65535
    return v


def append_crc(bits):
    a = binary(bits)
    return np.concatenate((a, integer_bits(crc16(a), 16)))


def crc_ok(bits):
    a = binary(bits)
    return bool(len(a) >= 16 and crc16(a[:-16]) == bits_integer(a[-16:]))


def pack_body(payload, profile):
    a = binary(payload)
    capacity = int(profile['k'])-29
    require(0 < len(a) <= min(capacity, 8191), 'Source payload does not fit paid L/k')
    if profile['mode'] == 'raw':
        require(len(a) == 12*token_count(profile['m'], profile['K']), 'Raw payload length differs from public state')
    else:
        require(profile['mode'] == 'arithmetic' and len(a) >= 2, 'Invalid arithmetic mode/length')
    info = np.concatenate((integer_bits(len(a), 13), a, np.zeros(capacity-len(a), dtype=np.uint8)))
    return append_crc(info)


def parse_body(decoded, profile):
    """Physical CRC first, then length/padding. Arithmetic canonical check is separate."""
    a = binary(decoded)
    require(len(a) == profile['k'], 'Actual decoder output has wrong k')
    if not crc_ok(a):
        return dict(crc_accepted=False, parser_accepted=False, status='CRC_REJECT', payload=None)
    L = bits_integer(a[:13])
    capacity = profile['k']-29
    invalid = None
    if not 0 < L <= capacity:
        invalid = 'L_OUTSIDE_PROFILE_CAPACITY'
    elif profile['mode'] == 'raw' and L != 12*token_count(profile['m'], profile['K']):
        invalid = 'RAW_LENGTH_DIFFERS_FROM_PUBLIC_STATE'
    elif profile['mode'] == 'arithmetic' and L < 2:
        invalid = 'ARITHMETIC_LENGTH_TOO_SHORT'
    elif profile['mode'] not in ('raw', 'arithmetic'):
        invalid = 'UNKNOWN_MODE'
    elif np.any(a[13+L:-16]):
        invalid = 'NONZERO_KNOWN_PADDING'
    if invalid:
        return dict(crc_accepted=True, parser_accepted=False, status='DECODE_INVALID',
                    invalid_reason=invalid, payload=None, declared_length=L)
    return dict(crc_accepted=True, parser_accepted=True, status='PAYLOAD_PARSED',
                declared_length=L, payload=a[13:13+L].tolist())


def raw_payload(scales, m, K=0):
    n = token_count(m, K)
    require(n > 0 and len(scales) >= m+(K > 0), 'Missing raw source scales')
    values = []
    for i in range(m+(K > 0)):
        a = np.asarray(scales[i])
        require(a.shape == (SIZES[i]**2,) and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 4096)), 'Invalid raw scale tokens')
        values.extend(a[:K] if i == m else a)
    require(len(values) == n, 'Raw raster token count mismatch')
    return np.concatenate([integer_bits(int(x), 12) for x in values])


def raw_tokens(payload, profile):
    require(profile['mode'] == 'raw', 'Not raw mode')
    a = binary(payload)
    require(len(a) == 12*token_count(profile['m'], profile['K']), 'Invalid raw bit count')
    return np.asarray([bits_integer(x) for x in a.reshape(-1, 12)], dtype=np.int64)


def pam(q):
    require(q in (4, 6), 'H supports 16QAM/64QAM')
    L = 1 << (q//2)
    indices = np.arange(L, dtype=np.int64)
    decoded = indices.copy()
    shift = 1
    while shift < L:
        decoded ^= decoded >> shift
        shift *= 2
    return (2*decoded-(L-1)).astype(np.float64)/math.sqrt((L*L-1)/3)


def modulate(bits, q):
    a = np.asarray(bits)
    require(a.ndim == 2 and a.shape[1] % q == 0 and np.isin(a, (0, 1)).all(), 'Aligned binary codeword batch required')
    width = q//2
    pairs = a.reshape(len(a), -1, width).astype(np.int64)
    labels = np.sum(pairs*(1 << np.arange(width-1, -1, -1)), axis=-1)
    return pam(q)[labels].reshape(len(a), -1, 2).astype(np.float32)


def logsumexp(x, axis):
    maximum = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(maximum+np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True)), axis=axis)


def demap(received, snr_db, q):
    y = np.asarray(received, dtype=np.float64)
    require(y.ndim == 3 and y.shape[-1] == 2 and np.isfinite(y).all()
            and math.isfinite(float(snr_db)), 'Finite received real I/Q batch required')
    gamma = 10**(float(snr_db)/10)
    lp = -.5*gamma*(y.reshape(len(y), -1, 1)-pam(q))**2
    labels = np.arange(1 << (q//2))
    values = []
    for shift in range(q//2-1, -1, -1):
        one = ((labels >> shift) & 1).astype(bool)
        values.append(logsumexp(lp[:, :, one], -1)-logsumexp(lp[:, :, ~one], -1))
    # Sionna expects log P(bit=1)/P(bit=0).
    return np.stack(values, -1).reshape(len(y), -1).astype(np.float32)


def scramble_mask(n, frame_counter, session='body', group=0):
    require(type(frame_counter) is int and frame_counter >= 0 and type(group) is int and group >= 0,
            'Public nonnegative frame/group counters required')
    seed = int.from_bytes(hashlib.sha256(json.dumps([PROTOCOL, 'scramble', session, frame_counter, group],
                                                  sort_keys=True).encode()).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(seed)).integers(0, 2, int(n), dtype=np.uint8)


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def body_phy_key(profile, backend):
    return digest(checked_layout(backend,profile))


def checked_layout(backend, profile):
    layout=backend.plan(profile['k'],profile['n'],profile['q'])
    if profile.get('layout_id') is not None:
        require(layout.get('layout_id')==profile['layout_id'], 'Public profile and actual LDPC layout differ')
    return layout


def transmit_body(backend, payload, profile, frame_counter, session='body'):
    info = pack_body(payload, profile)
    checked_layout(backend,profile)
    code = as_numpy(backend.encode(info[None], profile['n'], profile['q']))
    require(code.shape == (1, profile['n']) and np.isin(code, (0, 1)).all(), 'Encoder output invalid')
    scrambled = (code.astype(np.uint8)+scramble_mask(profile['n'], frame_counter, session)) % 2
    wave = modulate(scrambled, profile['q'])[0]
    return wave, dict(k=len(info), n=profile['n'], q=profile['q'], L=len(payload),
        source_rate=len(payload)/profile['n'], input_rate=len(info)/profile['n'],
        source_padding_bits=profile['k']-29-len(payload), actual_energy=float(np.square(wave.astype(np.float64)).sum()),
        body_symbols=len(wave), total_symbols=68+len(wave), info_sha256=array_sha(info))


def receive_body(backend, received, profile, snr_db, frame_counter, ledger, phase, event_id, session='body'):
    """One packet charge; receiver parameters come from the parsed public header."""
    y = np.asarray(received, dtype=np.float32)
    require(y.shape == (profile['n']//profile['q'], 2), 'Wrong paid body shape')
    layout = checked_layout(backend,profile)
    key = digest(layout)
    ledger.register_configuration(key, layout)
    request = dict(profile_key=profile['profile_key'], received_sha256=array_sha(y), snr_db=float(snr_db),
                   public_frame_counter=frame_counter, session=session)
    def decode():
        logits = demap(y[None], snr_db, profile['q'])
        logits *= 1-2*scramble_mask(profile['n'], frame_counter, session).astype(np.float32)
        result = as_numpy(backend.decode(logits, profile['k'], profile['n'], profile['q']))
        require(result.shape == (1, profile['k']) and np.isin(result, (0, 1)).all(), 'Decoder output invalid')
        decoded = result[0].astype(np.uint8)
        return dict(**parse_body(decoded, profile), decoded_bits=decoded.tolist(),
                    phy_key=key, profile_key=profile['profile_key'])
    return ledger.decode_once(phase, event_id, 'body', key, request, decode)


def receive_header(header, received, snr_db, codebook, ledger, phase, event_id):
    y = np.asarray(received, dtype=np.float64)
    require(y.shape == (68, 2) and np.isfinite(y).all(), 'Wrong paid header shape')
    identity = dict(phy_key=HEADER_PHY_KEY, symbols=68, source_bits=12, crc_bits=16, tail_bits=6,
                    coded_bits=136, modulation='QPSK', Es=2)
    ledger.register_configuration(HEADER_PHY_KEY, identity)
    request = dict(received_sha256=array_sha(y), snr_db=float(snr_db), codebook_sha256=digest(codebook))
    return ledger.decode_once(phase, event_id, 'header', HEADER_PHY_KEY, request,
                             lambda: header.receive(y, snr_db, codebook))


def receive_frame(backend, header, received, snr_db, frame_counter, codebook, ledger, phase, event_id, session='body'):
    """No TX profile argument: a wrong accepted header selects its actual RX layout."""
    y = np.asarray(received)
    require(y.shape == (1024, 2) and np.isfinite(y).all(), 'Exactly 1024 finite paid symbols required')
    outcome = receive_header(header, y[:68], snr_db, codebook, ledger, phase, event_id+':header')
    if not outcome['header_ok']:
        return dict(header=outcome, body=None, status='HEADER_REJECT')
    pid = outcome['profile_id']
    p = codebook[str(pid)] if str(pid) in codebook else codebook[pid]
    require(p['n']//p['q'] == 956, 'Public codebook has a different H body length')
    body = receive_body(backend, y[68:], p, snr_db, frame_counter, ledger, phase, event_id+':body', session)
    return dict(header=outcome, body=body, status=body['status'])


def exact_uncoded_ser(snr_db, q):
    """Uniform square M-QAM coherent ML SER at Es/N0. Not an LDPC BLER formula."""
    require(q in (4, 6), '16/64 QAM only')
    M = 1 << q
    Q = .5*math.erfc(math.sqrt(3*10**(float(snr_db)/10)/(M-1))/math.sqrt(2))
    return 1-(1-2*(1-1/math.sqrt(M))*Q)**2


def exact_ser_from_decision_regions(snr_db, q):
    """Independent integration over every actual PAM decision region."""
    levels = np.sort(pam(q))
    boundaries = np.concatenate(([-np.inf], (levels[:-1]+levels[1:])/2, [np.inf]))
    sigma = 10**(-float(snr_db)/20)
    cdf = lambda x: .5*math.erfc(-x/math.sqrt(2))
    p_axis = np.mean([cdf((boundaries[i+1]-x)/sigma)-cdf((boundaries[i]-x)/sigma)
                      for i, x in enumerate(levels)])
    return float(1-p_axis*p_axis)
