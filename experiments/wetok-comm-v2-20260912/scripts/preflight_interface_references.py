"""CPU-only validation of frozen baseline image pointers and source/noise accounting."""

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
from wetok_comm.interface_evaluation import FrozenImageReferences, load_evaluation_config, reference_names
from wetok_comm.training import read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=EXPERIMENT / 'docs/interface_reference_preflight.json')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, study, base, parent = load_evaluation_config()
    if not arguments.execute:
        print('PLAN ONLY: 441 frozen reference rows on CPU; no new model, update, GPU or test population')
        return
    if arguments.output.exists():
        raise FileExistsError('preserve prior preflight evidence and choose a new output path')
    torch.set_num_threads(2)
    references = FrozenImageReferences(evaluation, study)
    images, codes, identifiers = read_population(base, 'development')
    checked, maximum_error = 0, 0.
    for index in (0, 37, 99):
        for snr in base['evaluation']['snrs_db']:
            for seed in base['evaluation']['noise_seeds']:
                noise_hash = hashlib.sha256(raw_noise(identifiers[index], seed).tobytes()).hexdigest()
                for name in reference_names(study):
                    image, row, archive = references.image(index, identifiers[index], snr, seed, name, noise_hash)
                    mse = (image - images[index]).square().mean()
                    psnr = float(-10 * mse.clamp_min(1e-12).log10())
                    error = abs(psnr - float(row['psnr_db']))
                    maximum_error = max(maximum_error, error)
                    if error > evaluation['reference_metric_tolerances']['psnr_db']:
                        raise RuntimeError('CPU source/image reference does not reproduce the known PSNR')
                    checked += 1
    report = {'status': 'INTERFACE_FROZEN_REFERENCE_CPU_PREFLIGHT_PASS', 'completed_local': now(),
        'checked_rows': checked, 'source_images': [0, 37, 99], 'maximum_PSNR_error_dB': maximum_error,
        'reference_receipt_sha256': evaluation['reference_receipt_sha256'],
        'implementation_sha256': sha256(EXPERIMENT / 'src/wetok_comm/interface_evaluation.py'),
        'new_receiver_outputs_evaluated': 0, 'GPU_used': False, 'optimizer_updates': 0,
        'limitation': 'Existing development reference integrity only; LPIPS/SSIM/DINO will be recomputed in the final GPU evaluation.'}
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
