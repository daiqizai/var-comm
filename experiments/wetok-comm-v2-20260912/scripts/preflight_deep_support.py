"""CPU checks of all frozen support references and four actual neural replays."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import torch

from wetok_comm.common import now, sha256, write_json
from wetok_comm.deep_support import DeepSupportReferences, FrozenDeepSupport, add_actual_noise, legacy_model_hash, load_deep_support
from wetok_comm.evaluation import raw_noise
from wetok_comm.interface_study import load_interface_study
from wetok_comm.training import read_population


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: 600 existing reference checks plus four CPU neural replays, no updates or GPU')
        return
    output = EXPERIMENT / 'docs/deep_support_preflight.json'
    if output.exists():
        raise FileExistsError('do not replace a recorded support preflight')
    torch.set_num_threads(2)
    config = load_deep_support()
    study, base, parent = load_interface_study()
    model = FrozenDeepSupport(config, 'cpu')
    references = DeepSupportReferences(config, base)
    images, codes, identifiers = read_population(base, 'development')
    checked, maximum_psnr_error, maximum_pixel_error, maximum_power_error = 0, 0., 0., 0.
    neural_replays = []
    for index, identifier in enumerate(identifiers):
        for snr in (5., 6.):
            for seed in base['evaluation']['noise_seeds']:
                reference, row = references.image(index, identifier, snr, seed)
                mse = (reference - images[index]).square().mean()
                difference = abs(float(-10 * mse.clamp_min(1e-12).log10()) - float(row['psnr_db']))
                maximum_psnr_error = max(maximum_psnr_error, difference)
                if difference > config['metric_replay_tolerances']['psnr_db']:
                    raise RuntimeError('support source/reference PSNR does not reproduce')
                checked += 1
                if index in (0, 37) and seed == base['evaluation']['noise_seeds'][0]:
                    signal = model.transmit(images[index:index + 1], snr)
                    noise = torch.tensor(raw_noise(identifier, seed)[None], dtype=torch.float32)
                    prediction = model.receive(add_actual_noise(signal, noise, snr), snr)[0]
                    error = float((prediction - reference).abs().max())
                    maximum_pixel_error = max(maximum_pixel_error, error)
                    maximum_power_error = max(maximum_power_error, float((signal.square().sum(-1).mean(-1) - 2).abs().max()))
                    if error > config['pixel_replay_max_error']:
                        raise RuntimeError('CPU support replay differs from the old actual waveform protocol')
                    neural_replays.append({'image_index': index, 'actual_snr': snr,
                        'NN_condition': config['actual_to_condition_snr'][snr], 'seed': seed, 'maximum_pixel_error': error})
    if legacy_model_hash(model.model) != config['legacy_model_sha256']:
        raise RuntimeError('CPU preflight changed the frozen strong baseline')
    result = {'status': 'DEEP_SUPPORT_CPU_PREFLIGHT_PASS', 'completed_local': now(), 'frozen_reference_rows_checked': checked,
        'actual_CPU_neural_replays': neural_replays, 'maximum_reference_PSNR_error': maximum_psnr_error,
        'maximum_CPU_neural_pixel_error': maximum_pixel_error, 'maximum_power_error': maximum_power_error,
        'source_config_sha256': sha256(EXPERIMENT / 'configs/deep_support.yaml'),
        'source_implementation_sha256': sha256(EXPERIMENT / 'src/wetok_comm/deep_support.py'),
        'optimizer_updates': 0, 'GPU_used': False, 'new_interface_development_outputs_evaluated': 0,
        'scope': 'All reference files/PSNR checked; only four neural CPU replays. Full 600-row GPU replay remains pending.'}
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
