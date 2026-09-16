"""Read-only CPU integrity check of old geometry and strong reference images."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import torch

from wetok_comm.common import now, sha256, write_json
from wetok_comm.evaluation import raw_noise
from wetok_comm.geometry_evaluation import GeometryReferences, load_geometry_evaluation
from wetok_comm.interface_evaluation import SYSTEM_REFERENCES
from wetok_comm.training import read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: 441 archived reference/source/noise checks on CPU; no new receiver output')
        return
    output = EXPERIMENT / 'docs/geometry_reference_preflight.json'
    if output.exists():
        raise FileExistsError('preserve prior preflight evidence')
    torch.set_num_threads(2)
    evaluation, config, base, qualification = load_geometry_evaluation()
    references = GeometryReferences(config)
    images, codes, identifiers = read_population(base, 'development')
    checked, maximum_error = 0, 0.
    for index in (0, 37, 99):
        for snr in base['evaluation']['snrs_db']:
            for seed in base['evaluation']['noise_seeds']:
                noise_hash = hashlib.sha256(raw_noise(identifiers[index], seed).tobytes()).hexdigest()
                examples = [references.control_image(index, identifiers[index], snr, seed, variant, noise_hash)
                            for variant in config['variants']]
                examples.extend(references.image(index, identifiers[index], snr, seed, name, noise_hash)[:2] for name in SYSTEM_REFERENCES)
                for image, row in examples:
                    mse = (image - images[index]).square().mean()
                    error = abs(float(-10 * mse.clamp_min(1e-12).log10()) - float(row['psnr_db']))
                    maximum_error = max(maximum_error, error)
                    if error > evaluation['reference_metric_tolerances']['psnr_db']:
                        raise RuntimeError('geometry reference/source pixels do not reproduce archived PSNR')
                    checked += 1
    report = {'status': 'GEOMETRY_REFERENCE_CPU_PREFLIGHT_PASS', 'completed_local': now(), 'checked_rows': checked,
        'maximum_PSNR_error_dB': maximum_error, 'implementation_sha256': sha256(EXPERIMENT / 'src/wetok_comm/geometry_evaluation.py'),
        'reference_receipt_sha256': config['control_evaluation_receipt_sha256'], 'new_model_outputs_evaluated': 0,
        'GPU_used': False, 'scope': 'Archived references only; final matched new/old geometry GPU remeasurement remains pending.'}
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
