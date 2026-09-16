"""CPU-only diagnostics of the mapping normalization and receiver state variance."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np
import torch

from wetok_comm.common import PROJECT, configure_torch, now, output_path, settings, sha256, write_json
from wetok_comm.native import indices_to_features
from wetok_comm.training import new_network, paired_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: frozen checkpoint forward diagnostics on CPU, no visual model or updates')
        return
    configure_torch()
    torch.set_num_threads(2)
    config = settings()
    cache = output_path(config, 'cache')
    codes = np.load(cache / 'train_codes.npy', mmap_mode='r')
    batch = next(paired_batches(config, 20000, 5000, 5001))
    selected = np.array(codes[batch['indices'].numpy(), batch['flip'].long().numpy()], copy=True)
    source = indices_to_features(torch.from_numpy(selected))
    snrs = torch.full((4,), 19.)
    parent = output_path(config, 'training')
    saved = torch.load(parent / 'milestones/step_0005000_optimizer.pt', map_location='cpu', weights_only=True)
    records = []
    for step in (2000, 5000):
        for variant in config['arms']:
            network = new_network(config, variant, 'cpu').eval()
            weights = (torch.load(parent / variant / 'checkpoints/step_0002000.pt', map_location='cpu', weights_only=True)['model']
                       if step == 2000 else saved['models'][variant])
            network.load_state_dict(weights, strict=True)
            captured = {}
            variances = []
            def capture_lift(module, inputs, output):
                captured['lift'] = output.detach()
            def capture_residual(module, inputs, output):
                captured['residual'] = output.detach()
            def capture_context(module, inputs):
                values = inputs[0].detach()
                variances.append({'positions': values.shape[1], 'min_variance_across_32_channels': float(values.var(-1, unbiased=False).min()),
                                  'median_variance_across_32_channels': float(values.var(-1, unbiased=False).median())})
            hooks = [network.encoder.channel_lift.register_forward_hook(capture_lift),
                     network.encoder.output.register_forward_hook(capture_residual),
                     network.receiver.context_input[0].register_forward_pre_hook(capture_context)]
            with torch.no_grad():
                signal = network.transmit(source, snrs)
                result = network.receive(signal, snrs)
                unnormalized = torch.einsum('mn,bnc->bmc', network.encoder.spatial_mixing, captured['lift'] + captured['residual'])
                rms = unnormalized.square().mean((1, 2)).sqrt()
                records.append({'checkpoint_step': step, 'variant': variant, 'raw_TX_RMS': rms.tolist(),
                                'logit_absolute_mean': float(result['logits'].abs().mean()),
                                'logit_absolute_max': float(result['logits'].abs().max()),
                                'noiseless_bit_error_rate': float(result['native_fq'].ne(source).float().mean()),
                                'context_layernorm_input_variance': variances})
            for hook in hooks:
                hook.remove()
    output = PROJECT / 'outputs/WETOK-MAPPING-CONDITIONING-20260912'
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'CPU_MAPPING_CONDITIONING_DIAGNOSTIC', 'completed_local': now(),
              'fixed_training_batch_fingerprint': batch['fingerprint'], 'source_images': 4, 'channel': 'noiseless_nominal19',
              'optimizer_updates': 0, 'visual_models_loaded': False, 'records': records,
              'parent_checkpoint_sha256': sha256(parent / 'milestones/step_0005000_optimizer.pt'),
              'limitations': 'Small training-only diagnostic, not a quality result or sufficient population-wide explanation.'}
    write_json(output / 'summary.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
