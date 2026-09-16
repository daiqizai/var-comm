"""CPU-only real 8PSK/FEC transmission of native WeTok payloads."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np

from wetok_comm.common import PROJECT, WORKSPACE, artifact_hashes, now, output_path, settings, sha256, snapshot, verify_sources, write_json
from wetok_comm.digital import WeTokDigital
from wetok_comm.evaluation import raw_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/study.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config = settings(arguments.config)
    if not arguments.execute:
        print('PLAN ONLY: native 8192 bits + CRC/tail + paid 8PSK/FEC; no GPU model')
        return
    output = (arguments.output_dir or PROJECT / 'outputs/WETOK-DIGITAL-8PSK-20260912').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('digital reference outputs must stay in VAR_COMM')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), arguments.config, EXPERIMENT / 'src/wetok_comm/digital.py',
        EXPERIMENT / 'src/wetok_comm/evaluation.py', WORKSPACE / 'var-next-scale-comm/src/var_comm/scale_channel.py',
        WORKSPACE / 'var-next-scale-comm/src/var_comm/token_trellis.cpp'])
    cache = output_path(config, 'cache')
    receipt = json.loads((cache / 'completion.json').read_text())
    for name in ('development_codes.npy', 'development_ids.json'):
        if sha256(cache / name) != receipt['output_hashes'][name]:
            raise RuntimeError('native source cache changed')
    codes = np.load(cache / 'development_codes.npy', mmap_mode='r')[:, 0]
    identifiers = json.loads((cache / 'development_ids.json').read_text())
    modem = WeTokDigital()
    candidates = np.empty((100, 7, 3, 16, 16, 4), dtype=np.uint8)
    accepted = np.zeros((100, 7, 3), dtype=np.bool_)
    rows = []
    for index, identifier in enumerate(identifiers):
        source = np.array(codes[index], copy=True)
        tick = time.perf_counter()
        transmitted = modem.transmit(source)
        tx_seconds = time.perf_counter() - tick
        if np.max(np.abs(np.square(transmitted).sum(-1) - 2)) > 1e-12:
            raise RuntimeError('digital symbols exceed fixed per-symbol energy')
        truth = np.unpackbits(source.reshape(-1), bitorder='little')
        for snr_index, snr in enumerate(config['evaluation']['snrs_db']):
            for seed_index, seed in enumerate(config['evaluation']['noise_seeds']):
                noise = raw_noise(identifier, seed)
                received = transmitted + noise / np.sqrt(10 ** (snr / 10))
                tick = time.perf_counter()
                decoded = modem.receive(received, snr)
                rx_seconds = time.perf_counter() - tick
                candidates[index, snr_index, seed_index] = decoded['indices']
                accepted[index, snr_index, seed_index] = decoded['crc_accepted']
                rows.append({'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed,
                             'raw_source_bits': 8192, 'coded_bits': 9180, 'complex_uses': 3060, 'total_energy': 6120.,
                             'crc_accepted': decoded['crc_accepted'], 'bit_error_rate': float(np.mean(decoded['decoded_bits'] != truth)),
                             'candidate_used_for_image': True, 'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest(),
                             'received_sha256': hashlib.sha256(received.tobytes()).hexdigest(),
                             'tx_seconds': tx_seconds, 'FEC_receiver_seconds': rx_seconds})
        write_json(output / 'status.json', {'status': 'CPU_PHY_REPLAY', 'source_images_completed': index + 1, 'local_time': now()})
    np.savez(output / 'candidates.npz', indices=candidates, crc_accepted=accepted)
    with (output / 'per_frame.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verify_sources(sources)
    write_json(output / 'completion.json', {'status': 'DIGITAL_CANDIDATES_COMPLETE', 'completed_local': now(),
               'source_hashes': sources, 'ledger': modem.ledger(), 'rows': len(rows), 'optimizer_updates': 0,
               'ground_truth_used_for_decoding': False, 'output_hashes': artifact_hashes(output)})
    print('DIGITAL_CANDIDATES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
