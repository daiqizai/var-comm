"""Frozen paid side-information protocol for the two Swin total budgets."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
import sys
import numpy as np

RATES = {1024: 6, 2048: 13}
WIDTH, TOKENS = 320, 256
CAL_SNRS = (1, 4, 7, 10, 13)
CAL_SEEDS = (4101,)
TRAIN_SNRS = tuple(range(1, 14))
_PHY = None


def layout(N):
    if type(N) is not int or N not in RATES:
        raise ValueError('Only registered total N1024/N2048 are supported')
    channels = RATES[N]
    rank_bits = (math.comb(WIDTH, channels) - 1).bit_length()
    data = TOKENS * channels // 2
    return dict(N=N, E=2*N, channels=channels, data_N=data, header_N=N-data,
        power_bits=32, mask_rank_bits=rank_bits, payload_bits=32+rank_bits,
        crc_bits=16, tail_bits=6, information_bits=32+rank_bits+22,
        mother_coded_bits=2*(32+rank_bits+22), transmitted_header_bits=2*(N-data),
        no_information_padding_N=0)


def configure_phy(root):
    global _PHY
    path = Path(root).resolve()/'src'
    sys.path.insert(0, str(path))
    from var_comm import scale_channel
    if Path(scale_channel.__file__).resolve() != path/'var_comm/scale_channel.py':
        raise RuntimeError('Unexpected convolutional PHY implementation')
    _PHY = scale_channel
    return Path(scale_channel.__file__).resolve()


def _phy():
    if _PHY is None:
        raise RuntimeError('Explicitly configure the hash-bound project PHY first')
    return _PHY


def rank_subset(indices, channels):
    values = [int(i) for i in indices]
    if len(values) != channels or values != sorted(set(values)) or any(i < 0 or i >= WIDTH for i in values):
        raise ValueError('Expected sorted distinct active channels')
    return sum(math.comb(i, j+1) for j, i in enumerate(values))


def unrank_subset(rank, channels):
    if not 0 <= rank < math.comb(WIDTH, channels):
        raise ValueError('Illegal received mask rank')
    selected, upper = [], WIDTH-1
    for ordinal in range(channels, 0, -1):
        low, high = ordinal-1, upper
        while low < high:
            middle = (low+high+1)//2
            if math.comb(middle, ordinal) <= rank:
                low = middle
            else:
                high = middle-1
        selected.append(low)
        rank -= math.comb(low, ordinal)
        upper = low-1
    return tuple(sorted(selected))


def _bits(value, width):
    if not 0 <= value < 1 << width:
        raise ValueError('Header integer field overflow')
    return np.asarray([(value >> shift) & 1 for shift in range(width-1, -1, -1)], dtype=np.uint8)


def _integer(bits):
    value = 0
    for bit in bits:
        value = 2*value+int(bit)
    return value


@dataclass(frozen=True)
class RxContext:
    N: int
    channels: int
    power: float | None
    indices: tuple[int, ...] | None
    accepted: bool
    crc_accepted: bool
    fields_legal: bool


def encode_header(power, indices, N):
    spec = layout(N)
    power = float(np.float32(power))
    if not math.isfinite(power) or power <= 0:
        raise ValueError('Cannot transmit invalid normalization power')
    value = int.from_bytes(struct.pack('>f', power), 'big')
    payload = np.concatenate((_bits(value, 32), _bits(rank_subset(indices, spec['channels']), spec['mask_rank_bits'])))
    packet = _phy().encode_packet(payload, spec['header_N'])
    if np.asarray(packet['symbols']).shape != (spec['header_N'], 2):
        raise RuntimeError('Header did not consume its registered total budget')
    return np.asarray(packet['symbols'], dtype=np.float64)


def decode_header(observed, snr, N):
    spec = layout(N)
    if np.asarray(observed).shape != (spec['header_N'], 2):
        raise ValueError('Received header length differs')
    mapping = _phy().rate_match_indices(spec['mother_coded_bits'], spec['transmitted_header_bits'])
    evidence = _phy().channel_evidence(observed, mapping, spec['information_bits'], float(snr))
    decoded, _ = _phy().decode_map(evidence)
    crc = bool(_phy().crc_accepts(decoded[:-6]))
    payload = decoded[:spec['payload_bits']]
    power = struct.unpack('>f', _integer(payload[:32]).to_bytes(4, 'big'))[0]
    legal = math.isfinite(power) and power > 0
    try:
        indices = unrank_subset(_integer(payload[32:]), spec['channels'])
    except ValueError:
        indices, legal = None, False
    accepted = crc and legal
    # Erase both fields on failure; no downstream access to rejected candidates.
    return RxContext(N, spec['channels'], power if accepted else None,
        indices if accepted else None, accepted, crc, legal)


def standard_noise(source_id, seed, N, snr):
    key = f'SWIN-EXTERNAL-20261004|{source_id}|{int(seed)}|{int(N)}|{float(snr):g}'
    seed_value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    return np.random.default_rng(seed_value).standard_normal((int(N), 2)).astype(np.float64)


def transmit_frame(data_iq, power, indices, N):
    spec = layout(N)
    data = np.asarray(data_iq, dtype=np.float64)
    if data.shape != (spec['data_N'], 2) or not np.isfinite(data).all():
        raise ValueError('Transmitted Swin data differs from the registered shape')
    signal = np.concatenate((data, encode_header(power, indices, N)))
    if abs(float(np.square(signal).sum())-2*N) > .02:
        raise RuntimeError('Actual Swin frame violates E=2N')
    return signal


def observe_frame(signal, standard, snr, N):
    if np.asarray(signal).shape != (N, 2) or np.asarray(standard).shape != (N, 2):
        raise ValueError('Actual frame/noise size differs')
    received = np.asarray(signal, dtype=np.float64)+np.asarray(standard, dtype=np.float64)/10**(float(snr)/20)
    data_N = layout(N)['data_N']
    return received[:data_N], decode_header(received[data_N:], snr, N)


LEARNING_RATES = (1e-4, 3e-5, 1e-5)


def plateau_decision(history, step, maximum=240000, learning_rate=1e-4, last_lr_step=0):
    """Inspect all rate/SNR cells; reduce LR twice before stopping on a plateau."""
    if step < 20000 or step % 20000 or maximum < step:
        raise ValueError('Stopping decisions require a registered 20k milestone')
    bystep = {item['step']: item for item in history}
    needed = (step-20000, step-10000, step)
    if not all(s in bystep for s in needed):
        raise RuntimeError('Two full 10k calibration intervals are required')
    if learning_rate not in LEARNING_RATES or not 0 <= last_lr_step <= step:
        raise ValueError('Invalid registered optimizer stage')
    changes = {}
    metrics = [('mean_mse_by_N', str(N)) for N in RATES]
    metrics += [('mean_mse_by_cell', f'N{N}_snr{snr}') for N in RATES for snr in CAL_SNRS]
    for field, key in metrics:
        values = [float(bystep[s][field][key]) for s in needed]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise RuntimeError('Invalid complete calibration MSE')
        changes[field+'/'+key] = [(values[i]-values[i+1])/values[i] for i in range(2)]
    plateau = all(v < .002 for values in changes.values() for v in values)
    capped = step >= maximum
    eligible = step >= 40000 and step-last_lr_step >= 20000
    final_lr = learning_rate == LEARNING_RATES[-1]
    stop_plateau = plateau and eligible and final_lr
    lower = plateau and eligible and not final_lr and not capped
    next_lr = LEARNING_RATES[LEARNING_RATES.index(learning_rate)+1] if lower else learning_rate
    extend = not stop_plateau and not capped
    return dict(step=step, relative_mse_improvements=changes, threshold=.002,
        plateau_rule_met=plateau, final_lr_plateau_met=stop_plateau, extend=extend,
        earliest_decision_step=40000, last_lr_step=last_lr_step, current_learning_rate=learning_rate,
        lower_learning_rate=lower, next_learning_rate=next_lr,
        next_limit=min(step+20000, maximum) if extend else step,
        budget_truncated=capped and not stop_plateau,
        claim='predeclared_final_lr_calibration_plateau' if stop_plateau else 'budget_truncated' if capped else 'lower_learning_rate' if lower else 'continue_training',
        scientific_convergence_proven=False)

