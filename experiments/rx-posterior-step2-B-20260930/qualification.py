"""Real-weight engineering qualification on discarded copies of the P_low arm.

The caller's model, optimizer and random streams are preserved. The only output
is an engineering receipt; no probe update belongs to the training history.
"""
from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/rx-posterior-step1-20260929'))
import run_preflight as environment

import torch

from latent_enhancement.runtime import digest, write_json
from latent_enhancement.training import PairedOrder
from short_prefix.train import update
from var_comm.next_scale_prior import state_sha256
from protocol import PARENT_STEP, TRAIN_SNRS

PARENT = ROOT / 'outputs/SHORT-PREFIX-20260923/training/pure_seed2026092304'
PARENT_SHA = 'cba6190cabff71d2e6906644025a4769a5c2c46ec492bc8fc8d275fa085778af'
NATIVE_ATOL = 2e-5
NATIVE_RTOL = 2e-5


def same_tree(a, b):
    """Exact comparison, including optimizer counters and random state tensors."""
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def global_rng():
    return dict(torch_rng=torch.get_rng_state().clone(),
                cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()])


def set_global_rng(record):
    torch.set_rng_state(record['torch_rng'])
    torch.cuda.set_rng_state_all(record['cuda_rng'])


def source_bindings():
    files = [*HERE.glob('*.py'), HERE / 'protocol.json', HERE / 'EXECUTION_PLAN.md']
    return {str(p): digest(p) for p in sorted(files) if p.is_file()}


def copied_optimizer(models, state):
    optimizer = torch.optim.AdamW(models['P_low'].parameters(), lr=2e-4, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(state))
    return {'P_low': optimizer}


def streams(parent):
    order = PairedOrder(parent['order']['count'], 0)
    order.load_state_dict(copy.deepcopy(parent['order']))
    channel = torch.Generator()
    channel.set_state(parent['rng'])
    return order, channel


def next_probe(population, order, channel, added_step, device):
    """Exercise the real draw sequence using calibration pixels only.

    Training-order indices are recorded, then mapped onto four fixed calibration
    sources. These repeats are engineering probes, not quality observations.
    """
    train_ids = order.next(16)
    ids = train_ids.remainder(4)
    snr_indices = torch.randint(len(TRAIN_SNRS), (16,), generator=channel)
    unused_training_noise_indices = torch.randint(2, (16,), generator=channel)
    noise_seed = 2026092302 + PARENT_STEP + added_step
    batch = population.batch(ids, snr_indices, unused_training_noise_indices,
                             [noise_seed] * 16, device)
    draw = dict(training_order_indices=train_ids.clone(), calibration_indices=ids.clone(),
                snr_indices=snr_indices.clone(), unused_ni=unused_training_noise_indices.clone(),
                noise_seed=noise_seed)
    return batch, draw


def snapshot(models, opts, order, channel):
    return dict(models=copy.deepcopy(models.state_dict()),
                optimizer=copy.deepcopy(opts['P_low'].state_dict()),
                order=copy.deepcopy(order.state_dict()), channel_rng=channel.get_state().clone(),
                **global_rng())


def energy(wave):
    if wave.shape != (16, 4084, 2) or wave.dtype != torch.float32 or not torch.isfinite(wave).all():
        raise AssertionError('Probe waveform shape/dtype/finite values differ')
    values = wave.double().square().sum((1, 2))
    torch.testing.assert_close(values, torch.full_like(values, 8168.), atol=.02, rtol=1e-5)
    return dict(min=float(values.min()), max=float(values.max()),
                maximum_abs_error=float((values - 8168.).abs().max()))


def qualify(models, opts, calibration_population, decoder, lp, scale, parent_payload, out_dir):
    """Qualify selected27500 continuation and restore all caller-owned state."""
    if set(models) != {'P_low'} or set(opts) != {'P_low'}:
        raise AssertionError('Qualification requires exactly the registered single arm')
    if calibration_population.role != 'calibration' or len(calibration_population) != 1000:
        raise AssertionError('Qualification must use the original calibration population')
    if calibration_population.snrs != TRAIN_SNRS:
        raise AssertionError('Qualification SNR mapping differs')
    if any(p.requires_grad or p.grad is not None for p in decoder.parameters()):
        raise AssertionError('Dc must be frozen before qualification')
    if any(p.requires_grad or p.grad is not None for p in lp.parameters()):
        raise AssertionError('LPIPS must be frozen before qualification')
    if decoder.training or lp.training:
        raise AssertionError('Frozen Dc/LPIPS must use evaluation mode')
    selected = json.loads((PARENT / 'selected_P4084.json').read_text())
    checkpoint = Path(selected['checkpoint'])
    if selected['step'] != PARENT_STEP or digest(checkpoint) != PARENT_SHA:
        raise AssertionError('Qualification parent is not the frozen selected27500 checkpoint')
    if parent_payload['state']['step'] != PARENT_STEP:
        raise AssertionError('Parent payload counter differs')
    parent_state = {k[len('P4084.'):]: v for k, v in parent_payload['models'].items()
                    if k.startswith('P4084.')}
    if not same_tree(models['P_low'].state_dict(), parent_state):
        raise AssertionError('Initial model differs from the selected parent')
    if not same_tree(opts['P_low'].state_dict(), parent_payload['optimizers']['P4084']):
        raise AssertionError('Initial optimizer does not inherit the real parent moments')

    device = next(models.parameters()).device
    before_rng = global_rng()
    before_models = copy.deepcopy(models.state_dict())
    before_opts = copy.deepcopy(opts['P_low'].state_dict())
    before_modes = {name: module.training for name, module in models.named_modules()}
    decoder_sha, lp_sha = state_sha256(decoder), state_sha256(lp)
    bindings = source_bindings()
    report = None
    try:
        trial = copy.deepcopy(models)
        trial_opts = copied_optimizer(trial, before_opts)
        order, channel = streams(parent_payload)
        trial.eval()
        ids = torch.arange(16).remainder(4)
        si = torch.arange(16).remainder(4)
        initial_batch = calibration_population.batch(ids, si, torch.zeros_like(ids), [4101] * 16, device)
        with torch.no_grad():
            initial = trial['P_low']
            native, wave = initial(initial_batch['F'], initial_batch['snr'], initial_batch['noise'])
            received = wave + initial_batch['noise'] * torch.pow(10., -initial_batch['snr'] / 20)[:, None, None]
            split = initial.receive(received, initial_batch['snr'])
            if not torch.equal(native, split):
                raise AssertionError('Native forward differs from explicit TX/AWGN/RX at batch16')
            single = torch.cat([initial(initial_batch['F'][i:i+1], initial_batch['snr'][i:i+1],
                                        initial_batch['noise'][i:i+1])[0] for i in range(16)])
            torch.testing.assert_close(native, single, atol=NATIVE_ATOL, rtol=NATIVE_RTOL)
            initial_energy = energy(wave)
            batch1_error = float((native - single).abs().max())

        set_global_rng(parent_payload)
        batch0, draw0 = next_probe(calibration_population, order, channel, 0, device)
        update(trial, trial_opts, batch0, decoder, lp, scale, 4)
        after_first = snapshot(trial, trial_opts, order, channel)
        buffer = io.BytesIO()
        torch.save(after_first, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=True)
        if not same_tree(saved, after_first):
            raise AssertionError('Serialized probe checkpoint differs')

        batch1, draw1 = next_probe(calibration_population, order, channel, 1, device)
        update(trial, trial_opts, batch1, decoder, lp, scale, 4)
        uninterrupted = snapshot(trial, trial_opts, order, channel)

        resumed = copy.deepcopy(models)
        resumed.load_state_dict(saved['models'], strict=True)
        resumed_opts = copied_optimizer(resumed, saved['optimizer'])
        resumed_order, resumed_channel = streams(parent_payload)
        resumed_order.load_state_dict(saved['order'])
        resumed_channel.set_state(saved['channel_rng'])
        set_global_rng(saved)
        batch1_resumed, draw1_resumed = next_probe(calibration_population, resumed_order, resumed_channel, 1, device)
        if not same_tree(draw1, draw1_resumed) or not same_tree(batch1, batch1_resumed):
            raise AssertionError('Resumed data/SNR/noise draw differs')
        update(resumed, resumed_opts, batch1_resumed, decoder, lp, scale, 4)
        continuation = snapshot(resumed, resumed_opts, resumed_order, resumed_channel)
        if not same_tree(uninterrupted, continuation):
            raise AssertionError('Populated-optimizer/order/channel/CPU/CUDA resume is not bitwise identical')

        gradients = {'TX': 0., 'RX': 0.}
        for name, parameter in trial['P_low'].named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError('Missing or nonfinite trainable gradient: ' + name)
            gradients['TX' if name.startswith('encode.') else 'RX'] += float(parameter.grad.abs().sum())
        if min(gradients.values()) <= 0:
            raise AssertionError('TX or RX received no loss gradient through frozen Dc/LPIPS')
        steps = {float(s['step']) for s in trial_opts['P_low'].state.values()}
        if steps != {float(PARENT_STEP + 2)}:
            raise AssertionError('Optimizer counters did not continue from selected27500')
        with torch.no_grad():
            _, wave_after = trial['P_low'](batch1['F'], batch1['snr'], batch1['noise'])
            updated_energy = energy(wave_after)
        if state_sha256(decoder) != decoder_sha or state_sha256(lp) != lp_sha:
            raise AssertionError('Frozen Dc/LPIPS state changed')
        if any(p.grad is not None for p in decoder.parameters()) or any(p.grad is not None for p in lp.parameters()):
            raise AssertionError('Frozen Dc/LPIPS acquired a parameter gradient')
        if digest(checkpoint) != PARENT_SHA or source_bindings() != bindings:
            raise AssertionError('Parent checkpoint or bound sources changed')
        report = dict(status='REAL_GPU_POPULATED_PARENT_BITWISE_RESUME_PASS', passed=True,
                      synthetic=False, probe_updates_discarded=True, training_updates=0,
                      probe_updates=2, probe_sources='first four calibration sources repeated to batch16',
                      parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=PARENT_SHA,
                      parent_registration_sha256=parent_payload['registration_sha256'],
                      source_bindings=bindings, native_batch16_explicit_chain_max_abs=0.,
                      native_batch16_vs_batch1_max_abs=batch1_error,
                      native_batch16_vs_batch1_atol=NATIVE_ATOL,
                      native_batch16_vs_batch1_rtol=NATIVE_RTOL,
                      initial_energy=initial_energy, updated_energy=updated_energy,
                      optimizer_initial_step=PARENT_STEP, optimizer_after_probe_step=PARENT_STEP+2,
                      optimizer_parameter_states=len(trial_opts['P_low'].state),
                      resumed_model_optimizer_order_channel_cpu_cuda_rng_exact=True,
                      frozen_decoder_state_sha256=decoder_sha, frozen_lpips_state_sha256=lp_sha,
                      frozen_parameter_gradients_absent=True, trainable_gradient_sums=gradients,
                      first_noise_seed=draw0['noise_seed'], second_noise_seed=draw1['noise_seed'])
    finally:
        set_global_rng(before_rng)
        if not same_tree(models.state_dict(), before_models) or not same_tree(opts['P_low'].state_dict(), before_opts):
            raise AssertionError('Qualification changed the actual training model/optimizer')
        if {name: module.training for name, module in models.named_modules()} != before_modes:
            raise AssertionError('Qualification changed the actual training modes')
        if not same_tree(global_rng(), before_rng):
            raise AssertionError('Qualification failed to restore the caller CPU/CUDA RNG')
    if report is None:
        raise AssertionError('Qualification did not finish')
    report['actual_training_model_optimizer_modes_rng_unchanged'] = True
    output = Path(out_dir) / 'qualification_resumed_parent.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps(dict(status=report['status'], training_updates=0, receipt=str(output))), flush=True)
    return report


def main():
    result = subprocess.run([sys.executable, str(HERE / 'train.py'), '--qualification-only'])
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
