"""CPU-only real-parent initialization checks; does not activate training or profile the GPU."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

import torch
import yaml

from grid_controls.common import initial_system, load_parent, output_path, settings
from innovation_comm.common import new_system
from joint_sender.common import initial_system as original_joint_system
from wetok_comm.common import now, sha256, snapshot, write_json
from wetok_comm.native import indices_to_features
from wetok_comm.training import module_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, original, reference, base, parent_record = settings()
    if not arguments.execute:
        print('PLAN ONLY: CPU parent/parameter/waveform checks; no new GPU work, training or development inference')
        return
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cpu')
    parent = load_parent(reference, base, device)
    parent_before = module_sha256(parent)
    output = output_path(config, 'preparation')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/protocol.md',
        *sorted((EXPERIMENT / 'src/grid_controls').glob('*.py')), JOINT / 'src/joint_sender/runtime.py',
        INNOVATION / 'src/innovation_comm/model.py'])
    tolerance = yaml.safe_load((INNOVATION / 'configs/evaluation.yaml').read_text())['parent_signal_replay_tolerance']
    torch.manual_seed(reference['training']['new_parameter_seed'])
    truth = indices_to_features(torch.randint(256, (2, 16, 16, 4), dtype=torch.uint8))
    snrs = torch.tensor([1., 19.])
    noise = torch.randn(2, 3060, 2) * torch.pow(10., snrs / 10.).rsqrt()[:, None, None]
    rows = []
    with torch.no_grad():
        expected_signal = parent.transmit(truth, snrs)
        for variant in config['variants']:
            model = initial_system(parent, variant, reference, device).eval()
            signal = model.transmit(truth, snrs)
            waveform_error = float((signal - expected_signal).abs().max())
            if waveform_error > tolerance or float((signal.square().sum(-1).mean(-1) - 2).abs().max()) > 1e-5:
                raise RuntimeError('initial signal changed beyond the existing tolerance or violated energy')
            if variant == 'full_grid_state_history':
                baseline = original_joint_system(parent, 'multiscale_state_history', reference, device)
                if module_sha256(model) != module_sha256(baseline):
                    raise RuntimeError('ordinary grid control added or altered communication parameters')
                initial_receiver_error = None
            else:
                baseline = new_system(parent, variant, reference, device).eval()
                if module_sha256(model) != module_sha256(baseline):
                    raise RuntimeError('Joint innovation initialization differs from the frozen-sender control')
                received = expected_signal + noise
                expected = baseline.receive(received, snrs)['receiver_features']
                actual = model.receive(received, snrs)['receiver_features']
                initial_receiver_error = float((expected - actual).abs().max())
                if initial_receiver_error > tolerance:
                    raise RuntimeError('initial Joint innovation output changed beyond existing tolerance')
            rows.append({'variant': variant, 'model_state_sha256': module_sha256(model),
                'encoder_sha256': module_sha256(model.encoder), 'receiver_sha256': module_sha256(model.receiver),
                'communication_parameters': sum(parameter.numel() for parameter in model.parameters()),
                'optimized_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                'encoder_parameters': sum(parameter.numel() for parameter in model.encoder.parameters()),
                'initial_waveform_max_error': waveform_error, 'initial_receiver_max_error': initial_receiver_error,
                'ordinary_different_grid_not_claimed_identical_initial_outputs': variant == 'full_grid_state_history'})
    if module_sha256(parent) != parent_before:
        raise RuntimeError('CPU initialization modified the frozen parent')
    write_json(output / 'initialization.json', {'status': 'CPU_GRID_INITIALIZATION_PASS_NOT_ACTIVATED',
        'completed_local': now(), 'parent_checkpoint_sha256': reference['parent_model_sha256'],
        'parent_state_sha256': parent_before, 'variants': rows, 'existing_signal_tolerance': tolerance,
        'source_hashes': sources, 'GPU_used': False, 'optimizer_updates': 0, 'data_population_accessed': False,
        'research_goal_complete': False})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
