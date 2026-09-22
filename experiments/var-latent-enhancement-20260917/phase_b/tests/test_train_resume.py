import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from latent_enhancement.runtime import ResourceBusy, write_json
from latent_enhancement_b import train as training_module


class CPUProxy:
    cuda = SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                           max_memory_allocated=lambda: 0, get_rng_state_all=lambda: [], set_rng_state_all=lambda states: None)

    def device(self, name):
        return torch.device('cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


class ToyPopulation:
    def __init__(self, name):
        self.name = name
        self.shape = (32, 16, 16)
        self.snrs = torch.tensor([1., 4., 7., 13., 19.])
        self.seeds = [10, 11] if name == 'train' else [10, 11, 12]

    def __len__(self):
        return 20000 if self.name == 'train' else 1000

    def batch(self, indices, snr_indices, noise_indices, enhancement_seeds, device):
        grid = torch.arange(32 * 16 * 16).reshape(1, 32, 16, 16)
        truth = torch.sin((grid + indices.reshape(-1, 1, 1, 1)) * 0.01)
        base = truth * 0.5
        status = torch.ones(len(indices), 3)
        status[:, 0] = (indices % 7 != 0).float()
        status[:, 1] = (indices % 3 == 0).float()
        status[:, 2] = 8 * status[:, 0]
        noise = torch.cos(torch.arange(2048)[None, :] + enhancement_seeds[:, None]).reshape(len(indices), 1024, 2)
        return {'F': truth, 'Fb_TX': base, 'Fb_RX': base + noise_indices[:, None, None, None] * 0.05,
                'snr_db': self.snrs[snr_indices], 'rx_status': status, 'standard_noise': noise, 'target': truth[:, :3].sigmoid()}


class ToyPerceptual(torch.nn.Module):
    def forward(self, predicted, target):
        return (predicted - target).square().mean((1, 2, 3), keepdim=True)


def toy_models(paths, device):
    torch.manual_seed(102)
    return torch.nn.Conv2d(32, 3, 1).requires_grad_(False), None


def toy_decoder(vae, device):
    torch.manual_seed(103)
    return torch.nn.Sequential(torch.nn.Conv2d(32, 3, 1), torch.nn.Sigmoid()).requires_grad_(False)


def fake_calibrate(arms, decoder, original_decoder, perceptual, population, scale, output, step, full, device, microbatch):
    metrics = {'utility': 0.1 - step * 0.001, 'psnr_db': 20 + step * 0.1, 'lpips': 0.2 - step * 0.001}
    result = {'elapsed_seconds': 0., 'summary': {name: {'all': dict(metrics), 'per_snr': {'0': dict(metrics)}} for name in arms}}
    write_json(output / 'calibration' / f"{'full' if full else 'subset'}_{step:07d}.json", result)
    return result


class ResumeIntegrationTests(unittest.TestCase):
    def test_real_paired_training_loop_exact_resume_on_cpu(self):
        torch.set_num_threads(2)
        config = {'stage_B': {'initialization_seed': 15, 'model_hidden_channels': 8, 'model_residual_blocks': 1,
            'learning_rate': 0.0002, 'weight_decay': 0.0001, 'data_seed': 20, 'channel_seed': 21, 'logical_batch_size': 16,
            'microbatch_size': 4, 'LPIPS_weight': 0.1, 'normalized_latent_weight': 0.01, 'gradient_clip_norm': 1,
            'minimum_updates': 2, 'safety_maximum_updates': 4, 'full_calibration_interval': 2,
            'subset_calibration_interval': 1, 'checkpoint_interval': 1},
            'stopping': {'full_calibration_plateau_checks': 3, 'relative_utility_improvement': 0.001,
                         'PSNR_improvement_db': 0.01, 'LPIPS_improvement': 0.0001}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_states = []
            for experiment in ('continuous', 'resumed'):
                output = root / experiment
                for relative in ('qualification/completion.json', 'rx_cache/completion.json', 'decoder_gate.json'):
                    write_json(output / relative, {'synthetic_CPU_test_only': True})
                with contextlib.ExitStack() as stack:
                    replacements = {'OUT_B': output, 'torch': CPUProxy(), 'configure': lambda: None,
                        'load_gate': lambda: {}, 'validate_gpu_qualification': lambda: None,
                        'settings': lambda: copy.deepcopy(config), 'stage_b_sources': lambda: {},
                        'MatchedPopulation': ToyPopulation, 'load_models': toy_models, 'model_paths': lambda: {},
                        'load_decoder': toy_decoder, 'scale_statistics': lambda device: torch.ones(32),
                        'perceptual_model': lambda device: ToyPerceptual(), 'calibrate': fake_calibrate}
                    for name, value in replacements.items():
                        stack.enter_context(patch.object(training_module, name, value))
                    if experiment == 'resumed':
                        def yield_after_first_paired_update():
                            path = output / 'training/status.json'
                            if path.exists():
                                status = json.loads(path.read_text())
                                if status.get('step_per_arm', status.get('step')) == 1:
                                    raise ResourceBusy('synthetic other GPU task')
                        with patch.object(training_module, 'require_available', yield_after_first_paired_update):
                            self.assertEqual(training_module.main(), 75)
                        latest = json.loads((output / 'training/latest.json').read_text())
                        self.assertEqual(latest['step'], 1)
                    with patch.object(training_module, 'require_available', lambda: None):
                        self.assertEqual(training_module.main(), 0)
                    completion = json.loads((output / 'training/completion.json').read_text())
                    self.assertEqual(completion['updates_per_arm'], 4)
                    self.assertFalse(completion['full_experiment_complete'])
                    latest = json.loads((output / 'training/latest.json').read_text())
                    final_states.append(torch.load(latest['path'], map_location='cpu', weights_only=True))
            self.assertEqual(final_states[0]['training_state']['data_trace_sha256'], final_states[1]['training_state']['data_trace_sha256'])
            for name, values in final_states[0]['arms'].items():
                torch.testing.assert_close(values, final_states[1]['arms'][name], rtol=0, atol=0)
            for name in final_states[0]['optimizers']:
                for parameter, state in final_states[0]['optimizers'][name]['state'].items():
                    for key, value in state.items():
                        other = final_states[1]['optimizers'][name]['state'][parameter][key]
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, other, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
