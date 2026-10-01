"""One fresh P1024 using the completed P512 recipe and calibration-only stopping."""
import argparse
import copy
import csv
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
# VAR adds its own source directory to sys.path during model loading. Pin our
# entry point and load every sibling by absolute path so its train.py cannot win.
sys.modules['_extreme_bw_n1024_train'] = sys.modules[__name__]


def local_module(name):
    key = '_extreme_bw_n1024_'+name
    path = HERE/(name+'.py')
    if key in sys.modules:
        module = sys.modules[key]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError('Local module identity collision '+key)
        return module
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


os.environ['VAR_COMM_DECODER_GATE'] = str(ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v2_repaired_20260921/decoder_gate.json')
sys.path.insert(0, str(ROOT/'experiments/rx-posterior-step1-20260929'))
import run_preflight as environment
import numpy as np
import torch
from latent_enhancement.runtime import write_json, save_torch, digest, verify_snapshot, ResourceBusy, require_available, perceptual_model, model_paths
from latent_enhancement.training import PairedOrder
from latent_enhancement_b.common import load_decoder, scale_statistics
from var_comm.next_scale_prior import load_models, state_sha256
from short_prefix.common import Safety, identity_files, register, configure_runtime
from short_prefix.data import Population
from short_prefix.train import losses, update
BudgetContinuous = local_module('budget_model').BudgetContinuous
_rules = local_module('training_rules')
N, ARM, SEED = _rules.N, _rules.ARM, _rules.SEED
ORDER_SEED, CHANNEL_SEED = _rules.ORDER_SEED, _rules.CHANNEL_SEED
TRAIN_SNRS, CAL_SEEDS = _rules.TRAIN_SNRS, _rules.CAL_SEEDS
INTERVAL, MILESTONE = _rules.INTERVAL, _rules.MILESTONE
extension_decision, select_checkpoint = _rules.extension_decision, _rules.select_checkpoint
verify_matched_recipe = _rules.verify_matched_recipe

OUT = ROOT/'outputs/EXTREME-BW-20261001-R1-N1024'
TRAIN = OUT/f'training/p1024_{SEED}'
REFERENCE = ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/training/P2048_seed2026092304'
REFERENCE_SHA = 'f653a6ae8d75398ccfd02465d92a9cce218a8928a1c3548a39db7524c81f5f09'
P512_REFERENCE = ROOT/'outputs/EXTREME-BW-20260930-R1/training/p512_2026093001/registration.json'
P512_REFERENCE_SHA = '7790418a13e6a4142b6e56bf1bae9de0076194f9dc0f0d6bc2158fc48f3d8be9'
P512_PROTOCOL = ROOT/'experiments/extreme-bandwidth-20260930/training_protocol.json'
P512_PROTOCOL_SHA = 'd817968617890d8c8663bd2e038022aa3a0aeb4c6e3daf2e97fa1386833a682e'
DECODER_SHA = 'bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1'
SOURCE_NAMES = ['train.py', 'budget_model.py', 'qualification.py', 'training_rules.py', 'training_protocol.json', 'EXECUTION_PLAN.md']


def read(path):
    return json.loads(Path(path).read_text())


def own_bindings():
    return {str(HERE/name): digest(HERE/name) for name in SOURCE_NAMES}


def verify_reference():
    if digest(REFERENCE/'registration.json') != REFERENCE_SHA:
        raise RuntimeError('Audited P2048 reference registration changed')
    rec = read(REFERENCE/'registration.json')
    verify_snapshot(rec['bindings'])
    required = dict(batch_size=16, microbatch_size=4, learning_rate=2e-4, weight_decay=1e-4, gradient_clip=1, full_calibration_interval=2500)
    if any(rec['protocol'][key] != value for key, value in required.items()):
        raise RuntimeError('P2048 recipe changed')
    if rec['parent'] is not None or rec['initialization_seed'] != 2026092304 or rec['decoder_sha256'] != DECODER_SHA:
        raise RuntimeError('P2048 fresh initialization or decoder identity changed')
    if rec['order_seed'] != ORDER_SEED or rec['channel_seed'] != CHANNEL_SEED:
        raise RuntimeError('P2048 order/channel initialization changed')
    return rec


def verify_matched_reference(cfg, original):
    """Bind the actual P512 recipe and data without loading its trained weights."""
    verify_snapshot({str(P512_REFERENCE): P512_REFERENCE_SHA,
                     str(P512_PROTOCOL): P512_PROTOCOL_SHA})
    matched = read(P512_REFERENCE)
    verify_snapshot(matched['bindings'])
    if matched['protocol'] != read(P512_PROTOCOL):
        raise RuntimeError('Actual P512 registration and source protocol differ')
    verify_matched_recipe(cfg, matched['protocol'])
    paths = dict(matched_recipe_registration=str(P512_REFERENCE.relative_to(ROOT)),
                 matched_recipe_registration_sha256=P512_REFERENCE_SHA,
                 matched_recipe_protocol=str(P512_PROTOCOL.relative_to(ROOT)),
                 matched_recipe_protocol_sha256=P512_PROTOCOL_SHA)
    if any(cfg[key] != value for key, value in paths.items()):
        raise RuntimeError('Matched P512 source or recipe linkage changed')
    if (matched['N'], matched['initialization_seed'], matched['parent']) != (512, SEED, None):
        raise RuntimeError('Expected the actual fresh P512 initialization recipe')
    if matched['development_read'] or matched['decoder_state_sha256'] != DECODER_SHA:
        raise RuntimeError('Matched P512 training boundary or decoder differs')
    if matched['optimizer_initialization'] != 'FRESH_EMPTY_STEP0':
        raise RuntimeError('Matched P512 optimizer was not fresh')
    if (matched['order_seed'], matched['channel_seed']) != (ORDER_SEED, CHANNEL_SEED):
        raise RuntimeError('Matched P512 order/channel streams differ')
    for role in ('train', 'calibration'):
        if matched['latent_cache_bindings'][role] != original[role+'_files'] or matched['source_image_bindings'][role] != original[role+'_images']:
            raise RuntimeError('Actual P512 and original data identities differ: '+role)
        if matched['train_ids' if role == 'train' else 'calibration_ids'] != [r['image_id'] for r in original[role+'_images']]:
            raise RuntimeError('Actual P512 source order differs: '+role)
    return matched


def verify_population(population, reference):
    if population.role not in ('train', 'calibration'):
        raise RuntimeError('Only training/calibration allowed here')
    key = 'train' if population.role == 'train' else 'calibration'
    expected = 20000 if key == 'train' else 1000
    if len(population) != expected or population.snrs != TRAIN_SNRS:
        raise RuntimeError('Population size or SNR mapping changed')
    if population.bindings != reference[key+'_files'] or population.image_bindings != reference[key+'_images']:
        raise RuntimeError('Population differs from original P2048 data/ordering')
    if key == 'calibration' and population.seeds != CAL_SEEDS:
        raise RuntimeError('Original calibration noise repeats changed')


def load_reference_for_qualification(scale, device):
    """Only a small engineering parity probe; this does not reevaluate an old grid."""
    reg = verify_reference()
    selected = read(REFERENCE/'selected_P2048.json')
    if selected != read(REFERENCE/'finalization.json')['selected']:
        raise RuntimeError('Reference selected/finalization mismatch')
    cp = Path(selected['checkpoint'])
    if digest(cp) != selected['checkpoint_sha256'] or selected['registration_sha256'] != REFERENCE_SHA:
        raise RuntimeError('Reference checkpoint changed')
    payload = torch.load(cp, map_location='cpu', weights_only=True)
    if payload['registration_sha256'] != REFERENCE_SHA or payload['state']['step'] != selected['step']:
        raise RuntimeError('Reference checkpoint state mismatch')
    model = BudgetContinuous(scale.cpu(), 2048)
    model.load_state_dict({k[len('P2048.'):]: v for k, v in payload['models'].items() if k.startswith('P2048.')}, strict=True)
    meta = dict(method='P2048', arm='P2048', training_seed=2026092304, N=2048,
                kind='continuous', selected=selected, registration=reg,
                decoder_sha256=reg['decoder_sha256'], checkpoint=str(cp),
                selected_file=str(REFERENCE/'selected_P2048.json'), selected_sha256=digest(REFERENCE/'selected_P2048.json'))
    return model.to(device).eval().requires_grad_(False), meta


def load_for_evaluation(selected_path, scale, device):
    """Load only a completed calibration-selected P1024 and expose its exact identity."""
    selected_path = Path(selected_path)
    folder = selected_path.parent
    rec, reg, done = read(selected_path), read(folder/'registration.json'), read(folder/'completion.json')
    verify_snapshot(reg['bindings'])
    if rec != done['selected'][ARM] or rec['arm_key'] != ARM or rec['N'] != N:
        raise RuntimeError('Selected P1024/completion mismatch')
    regsha = digest(folder/'registration.json')
    if rec['registration_sha256'] != regsha or done['registration_sha256'] != regsha:
        raise RuntimeError('P1024 registration mismatch')
    if not done['state']['finished'] or done['development_read'] is not False:
        raise RuntimeError('P1024 calibration-only selection incomplete')
    best = select_checkpoint(done['state']['history'])
    if best['step'] != rec['step'] or best['utility'] != rec['utility']:
        raise RuntimeError('P1024 not selected by registered utility')
    verify_snapshot({best['calibration_csv']: best['calibration_sha256'], rec['checkpoint']: rec['checkpoint_sha256']})
    payload = torch.load(rec['checkpoint'], map_location='cpu', weights_only=True)
    if payload['registration_sha256'] != regsha or payload['state']['step'] != rec['step']:
        raise RuntimeError('P1024 checkpoint identity mismatch')
    model = BudgetContinuous(scale.cpu(), N)
    model.load_state_dict({k[len(ARM)+1:]: v for k, v in payload['models'].items() if k.startswith(ARM+'.')}, strict=True)
    if not torch.equal(model.scale.flatten().cpu(), scale.flatten().cpu()):
        raise RuntimeError('Evaluation s_F differs from the registered training scale')
    meta = dict(method=ARM, arm=ARM, training_seed=SEED, N=N, kind='continuous',
                training=str(folder), selected=rec, registration=reg,
                registration_sha256=regsha, selected_file=str(selected_path), selected_sha256=digest(selected_path),
                checkpoint=rec['checkpoint'], decoder_sha256=reg['decoder_state_sha256'],
                training_completed_step=done['state']['step'], completed_added_updates=done['state']['step'],
                total_updates=rec['step'], parent_updates=0, training_snrs_db=TRAIN_SNRS,
                E=2*N, communication_channels=model.channels,
                matched_recipe_registration_sha256=reg['reference_recipe_sha256'])
    return model.to(device).eval().requires_grad_(False), meta


def calibration(models, pop, decoder, lp, scale, device, step, safety):
    path = TRAIN/'calibration'/f'full_{step:05d}.csv'
    receipt = path.with_suffix('.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    modelsha = state_sha256(models[ARM])
    if receipt.exists():
        rec = read(receipt)
        if digest(path) != rec['sha256'] or rec['step'] != step or rec['rows'] != 15000 or rec['model_state_sha256'] != modelsha:
            raise RuntimeError('Completed full calibration identity mismatch')
        return dict(step=step, utility=rec['utility'], calibration_csv=str(path), calibration_sha256=rec['sha256'])
    rows, started = [], time.time()
    model = models[ARM].eval()
    with torch.no_grad():
        for start in range(0, len(pop), 16):
            if safety.check():
                raise ResourceBusy('Sustained thermal condition during full calibration')
            ids = torch.arange(start, min(start+16, len(pop)))
            for si, snr in enumerate(TRAIN_SNRS):
                for ni, seed in enumerate(CAL_SEEDS):
                    batch = pop.batch(ids, torch.full_like(ids, si), torch.full_like(ids, ni), [seed]*len(ids), device)
                    utility, mse, percept, aux, _ = losses(model, batch, decoder, lp, scale)
                    for j, index in enumerate(ids):
                        rows.append(dict(method=ARM, source_index=int(index), image_id=pop.ids[int(index)], snr_db=snr,
                                         seed=seed, mse=float(mse[j]), lpips_alex=float(percept[j]), normalized_latent=float(aux[j]), utility=float(utility[j])))
            write_json(TRAIN/'status.json', dict(status='FULL_CALIBRATION', pid=os.getpid(), step=step,
                                               sources=min(start+16, len(pop)), total_sources=len(pop), time=time.time()))
    if len(rows) != 15000 or len({(r['image_id'], r['snr_db'], r['seed']) for r in rows}) != 15000:
        raise RuntimeError('Full1000x5x3 calibration incomplete')
    utility = float(np.mean([r['utility'] for r in rows]))
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    write_json(receipt, dict(step=step, utility=utility, rows=15000, sha256=digest(path), seconds=time.time()-started,
                             source_count=1000, snrs_db=TRAIN_SNRS, seeds=CAL_SEEDS, development_read=False, model_state_sha256=modelsha))
    print('full calibration', step, utility, 'seconds', time.time()-started, flush=True)
    return dict(step=step, utility=utility, calibration_csv=str(path), calibration_sha256=digest(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification-only', action='store_true')
    args = parser.parse_args()
    TRAIN.mkdir(parents=True, exist_ok=True)
    if (TRAIN/'completion.json').exists() and not args.qualification_only:
        load_for_evaluation(TRAIN/'selected_P1024.json', scale_statistics(), torch.device('cpu'))
        print('P1024 already complete; validated existing selection', flush=True)
        return
    configure_runtime()
    require_available()
    device = torch.device('cuda:0')
    cfg, reference = read(HERE/'training_protocol.json'), verify_reference()
    matched_reference = verify_matched_reference(cfg, reference)
    if cfg['N'] != N or cfg['training_seed'] != SEED or cfg['training_snrs_db'] != TRAIN_SNRS:
        raise RuntimeError('Training protocol constants changed')
    scale = scale_statistics(device)
    vae, var = load_models(model_paths(), device)
    encoder_sha = state_sha256(vae)
    del var
    decoder = load_decoder(vae, device)
    del vae
    if state_sha256(decoder) != DECODER_SHA:
        raise RuntimeError('Explicit repaired20260921 Dc differs')
    lp = perceptual_model(device)
    cal = Population('calibration', N=N)
    verify_population(cal, reference)
    torch.manual_seed(SEED)
    models = torch.nn.ModuleDict({ARM: BudgetContinuous(scale.cpu(), N)}).to(device)
    opts = {ARM: torch.optim.AdamW(models[ARM].parameters(), lr=2e-4, weight_decay=1e-4)}
    if sum(p.numel() for p in models.parameters()) != cfg['architecture']['parameter_count']:
        raise RuntimeError('Registered architecture parameter count differs')
    initial_model_sha = state_sha256(models[ARM])
    if args.qualification_only:
        local_module('qualification').qualify(models, opts, cal, decoder, lp, scale, TRAIN)
        return
    qualified = read(TRAIN/'qualification_fresh_budget.json')
    if not qualified['passed'] or qualified['training_updates'] != 0 or not qualified['actual_training_model_optimizer_modes_rng_unchanged']:
        raise RuntimeError('Fresh budget qualification absent or changed real training')
    if qualified['initial_model_state_sha256'] != initial_model_sha or qualified['reference_registration_sha256'] != REFERENCE_SHA:
        raise RuntimeError('Fresh initialization/qualification identity mismatch')
    if qualified['matched_recipe_registration_sha256'] != P512_REFERENCE_SHA or qualified['matched_recipe_protocol_sha256'] != P512_PROTOCOL_SHA:
        raise RuntimeError('Qualification matched P512 recipe linkage changed')
    if qualified['N'] != N or qualified['communication_channels'] != 8:
        raise RuntimeError('Qualification physical budget changed')
    if qualified['frozen_decoder_state_sha256'] != state_sha256(decoder) or qualified['frozen_lpips_state_sha256'] != state_sha256(lp):
        raise RuntimeError('Qualification frozen loss/decoder identity changed')
    verify_snapshot(qualified['source_bindings'])
    train = Population('train', N=N)
    verify_population(train, reference)
    if set(train.ids)&set(cal.ids):
        raise RuntimeError('Training/calibration overlap')
    verify_snapshot({**train.bindings, **cal.bindings})
    bindings = identity_files([*[HERE/name for name in SOURCE_NAMES], REFERENCE/'registration.json', P512_REFERENCE, P512_PROTOCOL, TRAIN/'qualification_fresh_budget.json',
                              ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/train.py',
                              ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/data.py',
                              ROOT/'experiments/var-latent-enhancement-20260917/research/src/latent_research/models.py'])
    for path, expected in matched_reference['bindings'].items():
        if path in bindings and bindings[path] != expected:
            raise RuntimeError('Matched P512 executed dependency changed: '+path)
        bindings[path] = expected
    registration = dict(protocol=cfg, bindings=bindings, reference_recipe_path=str(P512_REFERENCE),
                        reference_recipe_sha256=P512_REFERENCE_SHA,
                        matched_recipe_protocol_path=str(P512_PROTOCOL), matched_recipe_protocol_sha256=P512_PROTOCOL_SHA,
                        original_recipe_path=str(REFERENCE/'registration.json'), original_recipe_sha256=REFERENCE_SHA,
                        matched_recipe_source_bindings=matched_reference['bindings'], parent=None, initialization_seed=SEED,
                        initial_model_state_sha256=initial_model_sha, N=N, parameter_count=cfg['architecture']['parameter_count'],
                        decoder_state_sha256=state_sha256(decoder), encoder_state_sha256=encoder_sha, lpips_state_sha256=state_sha256(lp),
                        latent_cache_bindings={'train': train.bindings, 'calibration': cal.bindings},
                        source_image_bindings={'train': train.image_bindings, 'calibration': cal.image_bindings},
                        train_ids=train.ids, calibration_ids=cal.ids, order_seed=ORDER_SEED, channel_seed=CHANNEL_SEED,
                        noise_namespace=f'VAR-CONTINUOUS-{N}', optimizer_initialization='FRESH_EMPTY_STEP0',
                        order_rng_initialization='FRESH_ORIGINAL_P2048_ORDER_CHANNEL_SEEDS', development_read=False)
    register(TRAIN/'registration.json', registration)
    regsha = digest(TRAIN/'registration.json')
    order, rng = PairedOrder(len(train), ORDER_SEED), torch.Generator().manual_seed(CHANNEL_SEED)
    state = dict(step=0, parent_step=0, limit=MILESTONE, last_full=-1, history=[], decisions=[], training_seconds=0., calibration_seconds=0., finished=False)
    if (TRAIN/'latest.json').exists():
        latest = read(TRAIN/'latest.json')
        if latest['reason'] == 'failure':
            raise RuntimeError('Failure checkpoint requires diagnosis; automatic resume prohibited')
        if digest(latest['path']) != latest['sha256']:
            raise RuntimeError('Resume checkpoint SHA changed')
        payload = torch.load(latest['path'], map_location='cpu', weights_only=True)
        if payload['registration_sha256'] != regsha:
            raise RuntimeError('Resume registration changed')
        models.load_state_dict(payload['models'], strict=True)
        opts[ARM].load_state_dict(copy.deepcopy(payload['optimizers'][ARM]))
        order.load_state_dict(payload['order'])
        rng.set_state(payload['rng'])
        state = payload['state']
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
        if state['step'] != latest['step'] or state['step'] > state['limit'] or state['limit'] not in (20000, 30000, 40000):
            raise RuntimeError('Resume progress outside registered budget')
    stop = [False]
    def halt(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    safety = Safety()

    def checkpoint(reason):
        folder = TRAIN/'checkpoints'
        folder.mkdir(exist_ok=True)
        path = folder/f"step_{state['step']:05d}_{reason}_{time.time_ns()}.pt"
        save_torch(path, dict(models=models.state_dict(), optimizers={n: o.state_dict() for n, o in opts.items()},
                             order=order.state_dict(), rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all(), state=copy.deepcopy(state), registration_sha256=regsha))
        rec = dict(path=str(path), sha256=digest(path), step=state['step'], reason=reason)
        write_json(TRAIN/'latest.json', rec)
        return rec

    def full():
        before, started = checkpoint('before_full_calibration'), time.time()
        row = calibration(models, cal, decoder, lp, scale, device, state['step'], safety)
        state['calibration_seconds'] += time.time()-started
        row.update(checkpoint=before['path'], checkpoint_sha256=before['sha256'])
        if any(r['step'] == state['step'] for r in state['history']):
            raise RuntimeError('Duplicate complete calibration')
        state['history'].append(row)
        state['last_full'] = state['step']
        if state['step'] in (20000, 30000, 40000):
            decision = extension_decision(state['history'], state['step'])
            state['decisions'].append(decision)
            register(TRAIN/f"extension_{state['step']:05d}.json", decision)
            state['limit'], state['finished'] = decision['next_limit'], not decision['extend']
        checkpoint('full_calibration')
        write_json(TRAIN/'calibration_history.json', state['history'])

    try:
        if state['last_full'] != state['step'] and state['step']%INTERVAL == 0:
            full()
        while not state['finished']:
            if stop[0]:
                raise ResourceBusy('Requested safe pause before next update')
            if state['step'] >= state['limit']:
                raise RuntimeError('Training boundary reached without a complete stopping decision')
            ids = order.next(16)
            si = torch.randint(5, (len(ids),), generator=rng)
            ni = torch.randint(len(train.seeds), (len(ids),), generator=rng)
            batch = train.batch(ids, si, ni, [CHANNEL_SEED+state['step']]*len(ids), device)
            started = time.time()
            update(models, opts, batch, decoder, lp, scale, 4)
            state['training_seconds'] += time.time()-started
            state['step'] += 1
            if stop[0] or (state['step']%10 == 0 and safety.check()):
                raise ResourceBusy('Requested pause or sustained thermal condition')
            if state['step']%100 == 0:
                write_json(TRAIN/'status.json', dict(status='TRAINING', pid=os.getpid(), state=state, hardware=safety.last, time=time.time()))
                print('P1024 updates', state['step'], 'limit', state['limit'], 'training seconds', state['training_seconds'], flush=True)
            if state['step']%INTERVAL == 0:
                full()
            elif state['step']%500 == 0:
                checkpoint('regular')
        verify_snapshot(bindings)
        if state_sha256(decoder) != registration['decoder_state_sha256'] or state_sha256(lp) != registration['lpips_state_sha256']:
            raise RuntimeError('Frozen loss/decoder changed')
        if any(p.grad is not None for p in decoder.parameters()) or any(p.grad is not None for p in lp.parameters()):
            raise RuntimeError('Frozen parameters acquired gradients')
        best = select_checkpoint(state['history'])
        selected = dict(step=best['step'], utility=best['utility'], checkpoint=best['checkpoint'], checkpoint_sha256=best['checkpoint_sha256'],
                        arm_key=ARM, N=N, training_seed=SEED, registration_sha256=regsha, total_updates=best['step'], parent_updates=0,
                        calibration_csv=best['calibration_csv'], calibration_sha256=best['calibration_sha256'])
        register(TRAIN/'selected_P1024.json', selected)
        write_json(TRAIN/'completion.json', dict(status='CALIBRATION_ONLY_SELECTION_COMPLETE', state=state, selected={ARM: selected},
                                                registration_sha256=regsha, development_read=False, synthetic=False, convergence_claimed=False,
                                                budget_truncated=state['decisions'][-1]['budget_truncated']))
        write_json(TRAIN/'status.json', dict(status='TRAINING_COMPLETE', state=state, time=time.time()))
    except ResourceBusy as exc:
        checkpoint('safe_pause')
        write_json(TRAIN/'status.json', dict(status='PAUSED_SAFE_CHECKPOINT', reason=str(exc), state=state, time=time.time()))
        raise SystemExit(75)
    except Exception:
        checkpoint('failure')
        write_json(TRAIN/f'failure_{time.time_ns()}.json', dict(traceback=traceback.format_exc(), state=state, time=time.time()))
        raise


if __name__ == '__main__':
    try:
        main()
    except ResourceBusy as exc:
        write_json(TRAIN/'status.json', dict(status='WAITING_FOR_SAFE_RESOURCE_BEFORE_LOAD', reason=str(exc), time=time.time()))
        raise SystemExit(75)
