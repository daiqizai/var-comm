"""Verify actual initial weights, functions and reference history without GPU or optimizer updates."""

from pathlib import Path
import sys
import argparse
from unittest import mock

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path.insert(0, str(EXPERIMENT / 'src'))

import torch

from joint_sender.runtime import tensor_sha256
from vector_control.common import check_initial_models, initial_system, load_parent, output_path, qualified_reference, settings, validate_optimizer
from wetok_comm.common import artifact_hashes, configure_torch, now, snapshot, write_json
from wetok_comm.training import module_sha256, paired_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, r2, original, grid, reference, base, parent_record = settings()
    if not arguments.execute:
        print('PLAN ONLY: verify original initialization, complete10000 reference and equal initial functions; no GPU')
        return
    configure_torch()
    qualified = qualified_reference(config, r2, grid, scan_outputs=True)
    parent = load_parent(reference, base, 'cpu')
    model = initial_system(parent, reference, 'cpu').eval()
    control = check_initial_models(parent, model, reference, qualified, config)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    validate_optimizer(qualified['reference_optimizer'], parameters, config, base, 10000)
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], betas=tuple(base['training']['betas']), weight_decay=base['training']['weight_decay'])
    if optimizer.state:
        raise RuntimeError('new control must start from fresh Adam0 as the original residual did')
    generator = torch.Generator().manual_seed(913)
    source = torch.randint(2, (5, 32, 16, 16), generator=generator).float() * 2 - 1
    snrs = torch.tensor(base['channel']['snrs_db'])
    with torch.no_grad():
        signal = model.transmit(source, snrs)
        torch.testing.assert_close(signal, control.transmit(source, snrs), atol=0, rtol=0)
        received = signal + torch.randn((5, 3060, 2), generator=generator) * torch.pow(10., -snrs / 20)[:, None, None]
        with mock.patch.object(model.encoder, 'forward', wraps=model.encoder.forward) as new_calls:
            actual = model.receive(received, snrs)
        with mock.patch.object(control.encoder, 'forward', wraps=control.encoder.forward) as old_calls:
            expected = control.receive(received, snrs)
        if new_calls.call_count != 2 or old_calls.call_count != 2:
            raise RuntimeError('the vector control changed the number of RX re-encodings')
        for key in ('logits', 'native_fq', 'receiver_features'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        for key in ('states', 'predicted_symbols', 'source_hypotheses', 'feature_gates', 'residual_noise_ratios'):
            for current, old in zip(actual[key], expected[key]):
                torch.testing.assert_close(current, old, atol=0, rtol=0)
    count = 0
    for batch, row in zip(paired_batches(base, 20000, 7000, 17000), qualified['trace']):
        if (row['global_data_step'] != batch['step'] + 1 or row['batch_sha256'] != batch['fingerprint'] or
            row['paired_standard_noise_sha256'] != tensor_sha256(batch['noise'])):
            raise RuntimeError('new control data/noise history differs from residual')
        count += 1
    if count != 10000:
        raise RuntimeError('paired source trace is incomplete')
    output = output_path(config, 'preparation')
    output.mkdir(parents=True, exist_ok=False)
    hashes = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/protocol.md',
        *sorted((EXPERIMENT / 'src/vector_control').glob('*.py'))])
    write_json(output / 'initialization.json', {'status': 'R3_ORIGINAL_INITIALIZATION_AND_COMPLETE_HISTORY_PASS', 'local_time': now(),
        'reference_bindings': qualified['bindings'], 'initial_model_sha256': module_sha256(model),
        'initial_parent_sha256': module_sha256(parent), 'communication_parameters': sum(parameter.numel() for parameter in parameters),
        'initial_function_max_error': 0., 'reference_RX_E_calls': old_calls.call_count, 'new_RX_E_calls': new_calls.call_count,
        'paired_batches_verified': count, 'fresh_Adam_initial_states': len(optimizer.state), 'optimizer_updates': 0,
        'GPU_used': False, 'new_development_inference': False, 'source_hashes': hashes, 'research_goal_complete': False,
        'output_hashes': artifact_hashes(output)})
    print('PASS: actual original weights, zero-fusion functions, matched E calls and all10000 paired batches; no GPU or update')


if __name__ == '__main__':
    main()
