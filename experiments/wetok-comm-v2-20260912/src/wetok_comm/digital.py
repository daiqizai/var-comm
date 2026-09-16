"""Paid fixed-length WeTok 8PSK plus soft-bit convolutional-code reference."""

import ctypes
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .common import PROJECT, WORKSPACE, sha256, write_json

sys.path.insert(0, str(WORKSPACE / 'var-next-scale-comm/src'))
from var_comm.scale_channel import append_crc, convolutional_encode, crc_accepts, rate_match_indices

_LIBRARY = None


def library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = WORKSPACE / 'var-next-scale-comm/src/var_comm/token_trellis.cpp'
    digest = sha256(source)
    root = PROJECT / 'outputs/WETOK-COMM-V2-NATIVE-BUILD' / digest[:16]
    root.mkdir(parents=True, exist_ok=True)
    binary = root / 'token_trellis.so'
    if not binary.exists():
        temporary = root / 'token_trellis.pending.so'
        command = ['g++', '-O3', '-std=c++17', '-fPIC', '-shared', str(source), '-o', str(temporary)]
        subprocess.run(command, capture_output=True, text=True, check=True)
        temporary.replace(binary)
        write_json(root / 'build.json', {'source_sha256': digest, 'binary_sha256': sha256(binary), 'command': command,
                   'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0]})
    descriptor = json.loads((root / 'build.json').read_text())
    if descriptor['source_sha256'] != digest or descriptor['binary_sha256'] != sha256(binary):
        raise RuntimeError('native FEC implementation changed')
    _LIBRARY = ctypes.CDLL(str(binary))
    floating = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integral = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    _LIBRARY.decode_token_map.argtypes = [ctypes.c_int] * 4 + [floating, floating, ctypes.c_int, integral, floating]
    _LIBRARY.decode_token_map.restype = ctypes.c_int
    return _LIBRARY


class WeTokDigital:
    def __init__(self):
        self.payload_bits, self.crc_bits, self.tail_bits = 8192, 16, 6
        self.information_bits = self.payload_bits + self.crc_bits + self.tail_bits
        self.mapping = rate_match_indices(2 * self.information_bits, 3060 * 3)
        phases = np.arange(8)
        gray = phases ^ (phases >> 1)
        self.constellation = np.empty((8, 2), dtype=np.float64)
        self.constellation[gray] = np.sqrt(2) * np.stack((np.cos(2 * np.pi * phases / 8), np.sin(2 * np.pi * phases / 8)), axis=1)
        self.bit_table = ((np.arange(8)[:, None] >> np.array([2, 1, 0])) & 1).astype(np.uint8)

    def transmit(self, native_indices):
        codes = np.asarray(native_indices)
        if codes.shape != (16, 16, 4) or codes.dtype != np.uint8:
            raise ValueError('digital TX requires the complete native 4x8-bit group grid')
        source = np.unpackbits(codes.reshape(-1), bitorder='little')
        protected = append_crc(source)
        information = np.concatenate((protected, np.zeros(6, dtype=np.uint8)))
        coded = convolutional_encode(information)[self.mapping]
        symbols = coded.reshape(3060, 3) @ np.array([4, 2, 1], dtype=np.int64)
        return self.constellation[symbols]

    def receive(self, observations, snr_db):
        received = np.asarray(observations, dtype=np.float64)
        if received.shape != (3060, 2) or not np.isfinite(received).all():
            raise ValueError('digital RX receives exactly the paid waveform')
        gamma = 10 ** (float(snr_db) / 10)
        score = -.5 * gamma * np.square(received[:, None] - self.constellation[None]).sum(-1)
        llrs = np.stack([np.logaddexp.reduce(score[:, self.bit_table[:, bit] == 0], axis=1)
                         - np.logaddexp.reduce(score[:, self.bit_table[:, bit] == 1], axis=1) for bit in range(3)], axis=1)
        evidence = np.ascontiguousarray(np.bincount(self.mapping, weights=.5 * llrs.reshape(-1), minlength=2 * self.information_bits), dtype=np.float64)
        prior = np.zeros(4096, dtype=np.float64)
        decoded = np.empty(self.information_bits, dtype=np.uint8)
        objective = np.zeros(1, dtype=np.float64)
        result = library().decode_token_map(6, 0, self.information_bits, 0, evidence, prior, 0, decoded, objective)
        if result:
            raise RuntimeError(f'FEC bit-metric decoder failed: {result}')
        accepted = crc_accepts(decoded[:-6])
        native = np.packbits(decoded[:self.payload_bits], bitorder='little').reshape(16, 16, 4)
        return {'indices': native, 'crc_accepted': accepted, 'decoded_bits': decoded[:self.payload_bits],
                'candidate_retained_for_image': True}

    def ledger(self):
        return {'raw_source_bits': self.payload_bits, 'CRC_bits': self.crc_bits, 'tail_bits': self.tail_bits,
                'mother_coded_bits': 2 * self.information_bits, 'transmitted_coded_bits': len(self.mapping),
                'modulation': 'Gray_8PSK_constant_symbol_energy_2', 'complex_uses': 3060, 'header_uses': 0,
                'total_energy': 6120, 'decoder': 'soft_bit_metric_Viterbi_not_claimed_optimal_joint_symbol_ML'}
