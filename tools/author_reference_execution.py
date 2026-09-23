"""Paid author TX/channel/RX shared by parity checks and CPU-endpoint timing.

No target, TX power or learned rate mask is provided to the formal receiver.
The native reference is a separate diagnostic, never the delivered reconstruction.
"""
import hashlib
import math
import time
import numpy as np

SNRS = (1., 4., 7., 13., 19.)
METHODS = tuple([('swin', r) for r in (32, 64, 96)] + [('adjscc', r) for r in (2, 4, 6)])

def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def frame_counter(index, snr, seed, calibration=False):
    return 2026091600 + int(index)*32 + SNRS.index(float(snr))*3 + int(seed) - (4101 if calibration else 2001)

def resources(family, rate):
    if (family, rate) not in METHODS:
        raise ValueError('unregistered author operating point')
    data = 128*rate if family == 'swin' else 2048*rate
    metadata = 2*(32 + (math.comb(320, rate)-1).bit_length() + 16 + 6) if family == 'swin' else 0
    return data, metadata

def receive(model, family, rate, observed, snr, counter, decode):
    """Only observed waveform plus pre-shared SNR/rate/frame counter cross RX."""
    data_n, metadata_n = resources(family, rate)
    if observed.shape != (data_n+metadata_n, 2) or not np.isfinite(observed).all():
        raise ValueError('invalid received waveform')
    if family == 'swin':
        decoded = decode(observed[data_n:], snr, 320, rate)
        image = model.receive(observed[:data_n], snr, decoded) if decoded['usable'] else np.full((3,256,256), .5, dtype=np.float32)
    else:
        decoded = {'usable': True, 'crc_accepted': None}
        image = model.base_receive(observed, snr, counter)
    return image, decoded

def execute(model, family, rate, pixels, source_id, snr, seed, counter, *, sync=None, noise_fn=None, encode=None, decode=None):
    if sync is None:
        import torch
        sync = torch.cuda.synchronize
    if noise_fn is None:
        from var_comm.study import seeded_noise
        noise_fn = seeded_noise
    if encode is None or decode is None:
        from external_positioning.radio import encode_metadata, decode_metadata
        encode, decode = encode_metadata, decode_metadata
    if pixels.shape != (3,256,256) or pixels.dtype != np.uint8:
        raise ValueError('registered CPU uint8 RGB endpoint required')
    data_n, metadata_n = resources(family, rate)
    sync(); start = time.perf_counter()
    signal, context = model.transmit(pixels, snr, rate if family == 'swin' else counter)
    packet = encode(context['power'], context['indices'], 320) if family == 'swin' else None
    waveform = np.concatenate((signal, packet['symbols'])) if packet else signal
    sync(); tx_ms = 1000*(time.perf_counter()-start)
    energy = float(np.square(waveform).sum())
    if signal.shape != (data_n,2) or waveform.shape != (data_n+metadata_n,2) or not np.isfinite(energy) or abs(energy-2*len(waveform)) > .01:
        raise RuntimeError('author waveform resource/energy contract')
    # Exactly one full-waveform noise realization, wholly outside RX timing.
    noise = noise_fn(source_id, seed, waveform.shape)
    observed = waveform + noise / np.sqrt(10**(snr/10))
    sync(); start = time.perf_counter()
    image, decoded = receive(model, family, rate, observed, snr, counter, decode)
    sync(); rx_ms = 1000*(time.perf_counter()-start)
    if image.shape != pixels.shape or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise RuntimeError('illegal CPU RGB output')
    info = {'N':len(waveform), 'E':energy, 'data_N':data_n, 'metadata_N':metadata_n,
            'tx_ms':tx_ms, 'rx_ms':rx_ms, 'total_ms':tx_ms+rx_ms,
            'timing_endpoints':'CPU uint8 RGB -> CPU float64 I/Q; CPU noisy I/Q -> CPU float32 RGB; channel outside RX',
            'data_transmitted_sha256':array_sha(signal), 'data_observed_sha256':array_sha(observed[:data_n]),
            'standard_data_noise_sha256':array_sha(noise[:data_n]), 'image_sha256':array_sha(image),
            'metadata_usable':bool(decoded['usable']), 'metadata_crc_accepted':decoded['crc_accepted'],
            'metadata_exact':bool(np.array_equal(decoded['payload'],packet['payload'])) if packet else None,
            'fallback':'invalid_metadata_gray' if not decoded['usable'] else ''}
    return image, info, noise[:data_n]
