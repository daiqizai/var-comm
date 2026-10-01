"""Discarded fresh-budget probes: physical interface, gradients and exact resume."""
import copy
import io
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/rx-posterior-step1-20260929'))
import run_preflight as environment
import torch
from latent_enhancement.runtime import digest, write_json
from latent_enhancement.training import PairedOrder
from short_prefix.train import update
from var_comm.next_scale_prior import state_sha256
_rules_path = HERE/'training_rules.py'
_rules_spec = importlib.util.spec_from_file_location('_extreme_bw_n1024_qualification_rules', _rules_path)
_rules = importlib.util.module_from_spec(_rules_spec)
_rules_spec.loader.exec_module(_rules)
N, ARM, SEED = _rules.N, _rules.ARM, _rules.SEED
TRAIN_SNRS, ORDER_SEED, CHANNEL_SEED = _rules.TRAIN_SNRS, _rules.ORDER_SEED, _rules.CHANNEL_SEED


def same_tree(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def global_rng():
    return dict(torch_rng=torch.get_rng_state().clone(), cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()])


def set_global_rng(record):
    torch.set_rng_state(record['torch_rng'])
    torch.cuda.set_rng_state_all(record['cuda_rng'])


def energy(wave, count, uses):
    if wave.shape != (count, uses, 2) or wave.dtype != torch.float32 or not torch.isfinite(wave).all():
        raise AssertionError('Finite FP32 physical waveform length changed')
    values = wave.double().square().sum((1, 2))
    torch.testing.assert_close(values, torch.full_like(values, 2*uses), atol=.02, rtol=1e-5)
    return dict(N=uses, expected_energy=2*uses, min=float(values.min()), max=float(values.max()), maximum_abs_error=float((values-2*uses).abs().max()))


def copied_optimizer(models, state):
    result = torch.optim.AdamW(models[ARM].parameters(), lr=2e-4, weight_decay=1e-4)
    result.load_state_dict(copy.deepcopy(state))
    return {ARM: result}


def streams():
    return PairedOrder(20000, ORDER_SEED), torch.Generator().manual_seed(CHANNEL_SEED)


def next_probe(pop, order, rng, step, device):
    train_ids = order.next(16)
    ids = train_ids.remainder(4)
    si = torch.randint(5, (16,), generator=rng)
    ni = torch.randint(2, (16,), generator=rng)
    seed = CHANNEL_SEED+step
    batch = pop.batch(ids, si, ni, [seed]*16, device)
    return batch, dict(order_indices=train_ids.clone(), calibration_indices=ids.clone(), snr_indices=si.clone(), unused_ni=ni.clone(), noise_seed=seed)


def snapshot(models, opts, order, rng):
    return dict(models=copy.deepcopy(models.state_dict()), optimizer=copy.deepcopy(opts[ARM].state_dict()),
                order=copy.deepcopy(order.state_dict()), channel_rng=rng.get_state().clone(), **global_rng())


def qualify(models, opts, cal, decoder, lp, scale, out_dir):
    key = '_extreme_bw_n1024_train'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE/'train.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    training = sys.modules[key]
    if Path(training.__file__).resolve() != (HERE/'train.py').resolve():
        raise RuntimeError('Training module identity collision')
    own_bindings = training.own_bindings
    REFERENCE_SHA = training.REFERENCE_SHA
    P512_REFERENCE_SHA = training.P512_REFERENCE_SHA
    P512_PROTOCOL_SHA = training.P512_PROTOCOL_SHA
    load_reference_for_qualification = training.load_reference_for_qualification
    BudgetContinuous = training.BudgetContinuous
    from token_efficiency.models import BudgetContinuous as OriginalBudgetContinuous
    if set(models) != {ARM} or set(opts) != {ARM} or opts[ARM].state:
        raise AssertionError('Qualification requires the one fresh empty-optimizer arm')
    if len(cal) != 1000 or cal.role != 'calibration' or cal.snrs != TRAIN_SNRS or cal.N != N:
        raise AssertionError('Qualification requires original calibration data at N1024')
    if models[ARM].uses != N or models[ARM].channels != 8:
        raise AssertionError('Qualification requires the actual eight-channel P1024 model')
    if any(p.requires_grad or p.grad is not None for m in (decoder, lp) for p in m.parameters()) or decoder.training or lp.training:
        raise AssertionError('Dc and LPIPS must be frozen in evaluation mode')
    device = next(models.parameters()).device
    before_rng, before_models = global_rng(), copy.deepcopy(models.state_dict())
    before_opts = copy.deepcopy(opts[ARM].state_dict())
    before_modes = {name: module.training for name, module in models.named_modules()}
    bindings = own_bindings()
    decoder_sha, lp_sha = state_sha256(decoder), state_sha256(lp)
    initial_sha = state_sha256(models[ARM])
    report = None
    try:
        # Verify the same seed constructs exactly the same model on a fresh process.
        torch.manual_seed(SEED)
        fresh = BudgetContinuous(scale.cpu(), N).to(device)
        if not same_tree(fresh.state_dict(), models[ARM].state_dict()):
            raise AssertionError('Fresh registered initialization is not reproducible')
        del fresh
        # Reuse one actual P2048 selected checkpoint for a small interface parity
        # check, with four calibration sources and one noise repeat only.
        reference, ref_meta = load_reference_for_qualification(scale, device)
        original = OriginalBudgetContinuous(scale.cpu(), 2048).to(device).eval().requires_grad_(False)
        original.load_state_dict(reference.state_dict(), strict=True)
        ref_pop = copy.copy(cal)
        ref_pop.N = 2048
        ids = torch.arange(4)
        ref_batch = ref_pop.batch(ids, torch.zeros_like(ids), torch.zeros_like(ids), [4101]*4, device)
        with torch.no_grad():
            ref_z, ref_wave = reference(ref_batch['F'], ref_batch['snr'], ref_batch['noise'])
            old_z, old_wave = original(ref_batch['F'], ref_batch['snr'], ref_batch['noise'])
            observed = ref_wave+ref_batch['noise']*torch.pow(10., -ref_batch['snr']/20)[:, None, None]
            split = reference.receive(observed, ref_batch['snr'])
            if not torch.equal(ref_z, old_z) or not torch.equal(ref_wave, old_wave) or not torch.equal(ref_z, split):
                raise AssertionError('Actual selected P2048 native/split/extended-interface parity failed')
            reference_energy = energy(ref_wave, 4, 2048)
        del original, reference, ref_batch, ref_wave, old_wave, ref_z, old_z, split, observed, ref_pop
        trial, trial_opts = copy.deepcopy(models), None
        trial_opts = copied_optimizer(trial, before_opts)
        order, rng = streams()
        trial.eval()
        ids = torch.arange(16).remainder(4)
        si = torch.arange(16).remainder(5)
        initial_batch = cal.batch(ids, si, torch.zeros_like(ids), [4101]*16, device)
        with torch.no_grad():
            native, wave = trial[ARM](initial_batch['F'], initial_batch['snr'], initial_batch['noise'])
            observed = wave+initial_batch['noise']*torch.pow(10., -initial_batch['snr']/20)[:, None, None]
            split = trial[ARM].receive(observed, initial_batch['snr'])
            if not torch.equal(native, split):
                raise AssertionError('P1024 native/split physical chain differs')
            single = torch.cat([trial[ARM](initial_batch['F'][i:i+1], initial_batch['snr'][i:i+1], initial_batch['noise'][i:i+1])[0] for i in range(16)])
            torch.testing.assert_close(native, single, atol=2e-5, rtol=2e-5)
            batch_error = float((native-single).abs().max())
            initial_energy = energy(wave, 16, N)
        set_global_rng(before_rng)
        batch0, draw0 = next_probe(cal, order, rng, 0, device)
        update(trial, trial_opts, batch0, decoder, lp, scale, 4)
        saved = snapshot(trial, trial_opts, order, rng)
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location='cpu', weights_only=True)
        if not same_tree(saved, restored):
            raise AssertionError('Serialized probe checkpoint differs')
        batch1, draw1 = next_probe(cal, order, rng, 1, device)
        update(trial, trial_opts, batch1, decoder, lp, scale, 4)
        uninterrupted = snapshot(trial, trial_opts, order, rng)
        resumed = copy.deepcopy(models)
        resumed.load_state_dict(restored['models'], strict=True)
        resumed_opts = copied_optimizer(resumed, restored['optimizer'])
        resumed_order, resumed_rng = streams()
        resumed_order.load_state_dict(restored['order'])
        resumed_rng.set_state(restored['channel_rng'])
        set_global_rng(restored)
        batch1_resumed, draw1_resumed = next_probe(cal, resumed_order, resumed_rng, 1, device)
        if not same_tree(draw1, draw1_resumed) or not same_tree(batch1, batch1_resumed):
            raise AssertionError('Resumed data/noise draw changed')
        update(resumed, resumed_opts, batch1_resumed, decoder, lp, scale, 4)
        if not same_tree(uninterrupted, snapshot(resumed, resumed_opts, resumed_order, resumed_rng)):
            raise AssertionError('Model/optimizer/order/channel/CPU/CUDA resume not bitwise identical')
        gradients = {'TX': 0., 'RX': 0.}
        for name, parameter in trial[ARM].named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError('Missing/nonfinite communication gradient '+name)
            gradients['TX' if name.startswith('encode.') else 'RX'] += float(parameter.grad.abs().sum())
        if min(gradients.values()) <= 0:
            raise AssertionError('No TX/RX gradient through frozen Dc/LPIPS')
        if {float(v['step']) for v in trial_opts[ARM].state.values()} != {2.}:
            raise AssertionError('Fresh optimizer counters did not start at zero')
        with torch.no_grad():
            _, updated_wave = trial[ARM](batch1['F'], batch1['snr'], batch1['noise'])
            updated_energy = energy(updated_wave, 16, N)
        if state_sha256(decoder) != decoder_sha or state_sha256(lp) != lp_sha or any(p.grad is not None for m in (decoder, lp) for p in m.parameters()):
            raise AssertionError('Frozen decoder/perceptual state or gradients changed')
        if digest(ref_meta['checkpoint']) != ref_meta['selected']['checkpoint_sha256'] or own_bindings() != bindings:
            raise AssertionError('Reference checkpoint or frozen sources changed')
        report = dict(status='FRESH_P1024_REAL_GPU_EXACT_RESUME_PASS', passed=True, synthetic=False,
                      training_updates=0, probe_updates=2, probe_updates_discarded=True,
                      probe_sources='first four calibration sources repeated to batch16; no development',
                      initial_model_state_sha256=initial_sha, reference_registration_sha256=REFERENCE_SHA,
                      matched_recipe_registration_sha256=P512_REFERENCE_SHA,
                      matched_recipe_protocol_sha256=P512_PROTOCOL_SHA,
                      P512_weights_or_optimizer_loaded=False,
                      reference_selected=ref_meta['selected'], reference_P2048_native_old_split_max_abs=0.,
                      reference_energy=reference_energy, source_bindings=bindings,
                      N=N, E=2*N, training_seed=SEED, communication_channels=8, native_explicit_chain_max_abs=0.,
                      native_batch16_vs_batch1_max_abs=batch_error, native_atol=2e-5, native_rtol=2e-5,
                      initial_energy=initial_energy, updated_energy=updated_energy,
                      optimizer_initial_step=0, optimizer_after_probe_step=2,
                      resumed_model_optimizer_order_channel_cpu_cuda_rng_exact=True,
                      frozen_decoder_state_sha256=decoder_sha, frozen_lpips_state_sha256=lp_sha,
                      frozen_parameter_gradients_absent=True, trainable_gradient_sums=gradients,
                      first_noise_seed=draw0['noise_seed'], second_noise_seed=draw1['noise_seed'])
    finally:
        set_global_rng(before_rng)
        if not same_tree(models.state_dict(), before_models) or not same_tree(opts[ARM].state_dict(), before_opts):
            raise AssertionError('Qualification changed actual model/optimizer')
        if {name: module.training for name, module in models.named_modules()} != before_modes or not same_tree(global_rng(), before_rng):
            raise AssertionError('Qualification changed actual modes/random streams')
    if report is None:
        raise AssertionError('Qualification did not complete')
    report['actual_training_model_optimizer_modes_rng_unchanged'] = True
    path = Path(out_dir)/'qualification_fresh_budget.json'
    write_json(path, report)
    print('qualification passed; discarded updates only', path, flush=True)
    return report


if __name__ == '__main__':
    result = subprocess.run([sys.executable, str(HERE/'train.py'), '--qualification-only'])
    raise SystemExit(result.returncode)
