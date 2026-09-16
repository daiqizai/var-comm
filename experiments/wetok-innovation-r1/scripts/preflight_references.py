"""Read-only CPU checks of pinned parent waveforms and historical strong reference images."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src')]

import numpy as np
import torch

from innovation_comm.evaluation import References, load_evaluation, reference_names, validate_population
from wetok_comm.common import now, sha256, write_json
from wetok_comm.evaluation import raw_noise
from wetok_comm.training import read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: 441 archived reference images and 21 parent signals, CPU only, no new-model inference')
        return
    output = EXPERIMENT / 'docs/reference_preflight.json'
    if output.exists():
        raise FileExistsError('preserve prior reference preflight evidence')
    torch.set_num_threads(2)
    evaluation, config, base, parent_milestone = load_evaluation()
    references = References(evaluation)
    images, codes, identifiers = read_population(base, 'development')
    validate_population(identifiers)
    checked, signal_checks, maximum_error, maximum_power_error = 0, 0, 0., 0.
    for index in (0, 37, 99):
        for snr in base['evaluation']['snrs_db']:
            signal = references.parent_signal(index, snr)
            if signal.dtype != torch.float32 or tuple(signal.shape) != (1, 3060, 2) or not bool(torch.isfinite(signal).all()):
                raise RuntimeError('parent waveform has an invalid shape or dtype')
            power_error = abs(float(signal.square().sum(-1).mean()) - 2)
            maximum_power_error = max(maximum_power_error, power_error)
            if power_error > 1e-5:
                raise RuntimeError('pinned parent waveform violates the energy ledger')
            signal_hash = hashlib.sha256(signal.numpy().tobytes()).hexdigest()
            signal_checks += 1
            for seed in base['evaluation']['noise_seeds']:
                old = references.rows[index, snr, seed, 'geometry204x30__single_pass']
                if signal_hash != old['transmitted_sha256']:
                    raise RuntimeError('parent waveform is not the one used by its frozen image result')
                noise_hash = hashlib.sha256(raw_noise(identifiers[index], seed).tobytes()).hexdigest()
                for name in reference_names():
                    image, row, archive = references.image(index, identifiers[index], snr, seed, name, noise_hash)
                    mse = np.mean((image.numpy().astype(np.float64) - images[index].numpy().astype(np.float64)) ** 2)
                    error = abs(float(-10 * np.log10(max(float(mse), 1e-12))) - float(row['psnr_db']))
                    maximum_error = max(maximum_error, error)
                    if error > evaluation['reference_metric_tolerances']['psnr_db']:
                        raise RuntimeError('reference/source pixels do not reproduce archived PSNR')
                    checked += 1
    report = {'status': 'INNOVATION_REFERENCE_CPU_PREFLIGHT_PASS', 'completed_local': now(), 'checked_rows': checked,
        'checked_parent_signals': signal_checks, 'maximum_PSNR_error_dB': maximum_error, 'maximum_power_error': maximum_power_error,
        'implementation_sha256': sha256(EXPERIMENT / 'src/innovation_comm/evaluation.py'),
        'reference_receipt_sha256': evaluation['reference_receipt_sha256'], 'new_model_outputs_evaluated': 0,
        'GPU_used': False, 'scope': 'Archived references only; new receiver evaluation remains pending.'}
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
