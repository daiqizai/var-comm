"""One newly trained Swin SA+RA, complete source identities and resumable state."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

# Must precede CUDA initialization for deterministic matrix operations.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from swin_data import RGBPopulation, sha, read
from swin_model import build_official
from swin_protocol import RATES, CAL_SNRS, CAL_SEEDS, TRAIN_SNRS, configure_phy, layout, standard_noise, plateau_decision
from swin_replay import physical_batch
from swin_state import SourceOrder, identity, write, register, restore_training_log

HERE = Path(__file__).resolve().parent


class PauseRequested(Exception):
    pass


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def model_hash(model):
    result = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        result.update(key.encode()+b'\0'+str(value.dtype).encode()+b'\0'+str(value.shape).encode()+b'\0')
        result.update(value.tobytes())
    return result.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def exact_tree(a, b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(exact_tree(a[k], b[k]) for k in a)
    if type(a) is not type(b): return False
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(exact_tree(x, y) for x, y in zip(a, b))
    return a == b


def rng_state():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state_all(state['cuda'])


def configure_runtime():
    if not torch.__version__.startswith('1.12.1'):
        raise RuntimeError('Swin official SA/RA uses the registered isolated torch1.12.1 runtime')
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def available_gpu():
    raw = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    entries = [x.strip() for x in raw.splitlines() if x.strip()]
    if any(not x.isdecimal() for x in entries): raise RuntimeError('Unreadable GPU ownership')
    if set(map(int, entries)) - {os.getpid()}:
        raise PauseRequested('Another GPU owner appeared; checkpoint and yield')
    temperature = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    if not temperature.isdecimal(): raise RuntimeError('Unreadable GPU temperature')
    if int(temperature) >= 86: raise PauseRequested('GPU thermal guard requested a safe pause')


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected: raise RuntimeError('Bound Swin input/source changed: '+path)


def validate_config(config):
    required = dict(initialization='random', model='SwinJSCC_w/_SAandRA', model_size='base',
        image_size=256, total_N=list(RATES), channels=list(RATES.values()),
        data_N=[layout(N)['data_N'] for N in RATES], header_N=[layout(N)['header_N'] for N in RATES],
        optimizer='Adam', learning_rate=1e-4, betas=[.9, .999], eps=1e-8, weight_decay=0.,
        effective_batch_size=16, gradient_clipping=None, scheduler='registered_calibration_plateau', train_sources=20000,
        calibration_sources=1000, calibration_snrs=list(CAL_SNRS), calibration_noise_seeds=list(CAL_SEEDS),
        calibration_interval=2500, first_milestone=20000, extension_steps=20000,
        maximum_steps=240000, learning_rate_stages=[1e-4,3e-5,1e-5], earliest_plateau_decision=40000,
        plateau_relative_improvement=.002, development_read=False, holdout_read=False)
    if any(config.get(k) != value for k, value in required.items()):
        raise RuntimeError('Swin registered architecture, data, training or stopping scope differs')
    if config['microbatch_size'] not in (1, 2, 4, 8, 16):
        raise RuntimeError('Microbatch must divide the original effective batch16')


def optimizer_for(model, config):
    return torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=tuple(config['betas']),
        eps=config['eps'], weight_decay=config['weight_decay'])


def train_update(model, optimizer, images, N, snr, noise, microbatch):
    model.train(); optimizer.zero_grad(set_to_none=True)
    total = 0.
    for start in range(0, len(images), microbatch):
        end = min(start+microbatch, len(images))
        reconstructed = model.training_forward(images[start:end], snr, RATES[N], noise[start:end])
        loss = (reconstructed-images[start:end]).square().mean()
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite Swin training loss')
        (loss*((end-start)/len(images))).backward()
        total += float(loss.detach())*(end-start)/len(images)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
        raise FloatingPointError('Nonfinite Swin training gradient')
    if not bool(torch.stack([(g != 0).any() for g in grads]).any()):
        raise RuntimeError('Swin MSE produced no training gradient')
    optimizer.step()
    return total


def qualification(model, population, config, output, bindings):
    """Real training probe; restore random initial weights and all RNG afterward."""
    path = output/'qualification.json'
    initial_sha = model_hash(model)
    if path.exists():
        value = read(path)
        if value.get('status') != 'PASS' or value['initial_model_sha256'] != initial_sha or value['bindings'] != bindings or value['config_sha256'] != identity(config):
            raise RuntimeError('Swin qualification identity changed')
        return value
    initial, random_state = cpu_tree(model.state_dict()), rng_state()
    generator = torch.Generator().manual_seed(2026100491)
    images = population.batch(list(range(16)), 'cuda:0')
    noise = torch.randn((16, layout(2048)['data_N'], 2), generator=generator).cuda()
    optimizer = optimizer_for(model, config)
    torch.cuda.reset_peak_memory_stats()
    update_seconds = []
    def timed_update():
        torch.cuda.synchronize(); began=time.perf_counter()
        loss=train_update(model,optimizer,images,2048,1,noise,config['microbatch_size'])
        torch.cuda.synchronize(); update_seconds.append(time.perf_counter()-began)
        return loss
    try:
        started = time.perf_counter()
        timed_update()
        after1 = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()), rng=rng_state())
        loss_a = timed_update()
        torch.cuda.synchronize(); seconds = time.perf_counter()-started
        after2 = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()))
        model.load_state_dict(after1['model'], strict=True); optimizer.load_state_dict(after1['optimizer']); restore_rng(after1['rng'])
        loss_b = timed_update()
        if loss_a != loss_b or not exact_tree(after2['model'], model.state_dict()) or not exact_tree(after2['optimizer'], optimizer.state_dict()):
            raise RuntimeError('Populated Swin model/Adam/RNG resume failed exact replay')
        model.eval()
        frame_probes = []
        with torch.no_grad():
            for N in RATES:
                torch.cuda.synchronize(); start = time.perf_counter()
                source = images[:2]
                iq, power, indices = model.encode_data(source, 7, RATES[N])
                ideal = model.receive_data(iq, 7, power, indices)
                received, ledgers = physical_batch(model, source, N, 7, np.zeros((2,N,2)))
                error = float((ideal-received).abs().max())
                torch.cuda.synchronize()
                if not all(r['header_accepted'] for r in ledgers) or error > 1e-6 or not bool(torch.isfinite(received).all()):
                    raise RuntimeError('Actual paid Swin metadata noiseless roundtrip failed')
                frame_probes.append(dict(N=N, frames=2, actual_header=True, noiseless_max_abs=error,
                    energies=[r['actual_energy'] for r in ledgers], seconds=time.perf_counter()-start))
        value = dict(status='PASS', real_training_probe=True, formal_training_updates=0,
            initial_model_sha256=initial_sha, config_sha256=identity(config), bindings=bindings,
            source_role='first16 original training sources; discarded probe weights',
            source_ids=population.ids[:16], populated_adam_resume_exact=True,
            microbatch_size=config['microbatch_size'], effective_batch_size=16,
            probe_updates=3, first_two_updates_and_resume_snapshot_seconds=seconds,
            pure_training_update_seconds=update_seconds,
            pure_training_update_median_seconds=statistics.median(update_seconds),
            pure_training_update_timing='train_update plus GPU synchronization; excludes model snapshots and input copies',
            physical_roundtrip_probes=frame_probes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            device=torch.cuda.get_device_name(0), torch=torch.__version__, synthetic=False)
    finally:
        model.load_state_dict(initial, strict=True); restore_rng(random_state)
        model.zero_grad(set_to_none=True)
        del optimizer
    if model_hash(model) != initial_sha: raise RuntimeError('Training probe changed initial model')
    value['initial_model_and_rng_restored'] = True
    register(path, value)
    return value


def calibrate(model, population, config, output, step, model_path, regsha, stop):
    model.eval(); saved_rng = rng_state()
    modelsha = sha(model_path)
    completion_path = output/'calibration'/f'{step:06d}'/'completion.json'
    completed = read(completion_path) if completion_path.exists() else None
    cells, started = [], time.perf_counter()
    try:
        for N in RATES:
            for snr in CAL_SNRS:
                for seed in CAL_SEEDS:
                    if stop[0]: raise PauseRequested('Requested pause during calibration')
                    path = output/'calibration'/f'{step:06d}'/f'N{N}_snr{snr}_seed{seed}.json'
                    context = dict(registration_sha256=regsha, checkpoint_sha256=modelsha, step=step,
                        N=N, snr=snr, seed=seed, source_ids=population.ids)
                    if path.exists():
                        cell = read(path)
                        if cell.get('status') != 'COMPLETE' or cell['context'] != context or len(cell['rows']) != 1000:
                            raise RuntimeError('Existing Swin calibration cell differs')
                        if cell['payload_sha256'] != identity({k:v for k,v in cell.items() if k != 'payload_sha256'}):
                            raise RuntimeError('Calibration cell checksum differs')
                        if any(r['source_index'] != i or r['source_id'] != population.ids[i] or not math.isfinite(r['mse']) or r['mse'] < 0 for i,r in enumerate(cell['rows'])):
                            raise RuntimeError('Calibration source row identity differs')
                        if cell['mean_mse'] != math.fsum(r['mse'] for r in cell['rows'])/1000 or cell['header_failures'] != sum(not r['header_accepted'] for r in cell['rows']):
                            raise RuntimeError('Calibration aggregate differs from its rows')
                    else:
                        rows = []
                        for start in range(0, len(population), config['microbatch_size']):
                            if stop[0]: raise PauseRequested('Requested calibration pause')
                            if start % (config['microbatch_size']*8) == 0: available_gpu()
                            indices = list(range(start, min(start+config['microbatch_size'], len(population))))
                            image = population.batch(indices, 'cuda:0')
                            noises = np.stack([standard_noise(population.ids[i], seed, N, snr) for i in indices])
                            recon, ledgers = physical_batch(model, image, N, snr, noises)
                            mse = (recon-image).square().mean(dim=(1,2,3)).cpu().tolist()
                            for i, value, ledger in zip(indices, mse, ledgers):
                                if not math.isfinite(value) or value < 0: raise RuntimeError('Invalid actual calibration MSE')
                                rows.append(dict(source_index=i, source_id=population.ids[i], mse=value, **ledger))
                            write(output/'status.json', dict(status='CALIBRATING', step=step, N=N, snr=snr,
                                seed=seed, source_complete=len(rows), cell_complete=len(cells), total_cells=10,
                                frames_per_full_calibration=10000, updated=time.time()))
                        cell = dict(status='COMPLETE', context=context, rows=rows,
                            mean_mse=math.fsum(r['mse'] for r in rows)/len(rows),
                            header_failures=sum(not r['header_accepted'] for r in rows), synthetic=False)
                        cell['payload_sha256'] = identity(cell)
                        write(path, cell)
                    cells.append((path, cell))
        by_N = {str(N): math.fsum(c['mean_mse'] for _,c in cells if c['context']['N']==N)/5 for N in RATES}
        by_cell = {f"N{c['context']['N']}_snr{c['context']['snr']}":c['mean_mse'] for _,c in cells}
        result = dict(step=step, mean_mse_by_N=by_N, mean_mse_by_cell=by_cell, mean_mse=math.fsum(by_N.values())/2,
            checkpoint=str(model_path), checkpoint_sha256=modelsha,
            cells={str(p):sha(p) for p,_ in cells}, source_count=1000, rows=10000,
            header_failures=sum(c['header_failures'] for _,c in cells),
            seconds=completed['seconds'] if completed else time.perf_counter()-started,
            selection_uses_development=False, physical_metadata=True)
        register(completion_path, result)
        return result
    finally:
        restore_rng(saved_rng); model.train()


def run(args):
    import fcntl
    output, root, vendor = Path(args.output).resolve(), Path(args.root).resolve(), Path(args.vendor).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'run.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/'completion.json').exists():
        completion = read(output/'completion.json'); verify_bindings(completion['outputs'])
        verify_bindings(read(output/'registration.json')['bindings'])
        print('Swin training is complete; selected artifacts verified', flush=True); return
    if (output/'failure.json').exists():
        raise RuntimeError('Previous training failure requires diagnosis, not automatic restart')
    available_gpu(); configure_runtime()
    config = read(args.config); validate_config(config)
    phy = configure_phy(root)
    source_files = [*HERE.glob('swin_*.py'), Path(args.config).resolve(), phy, phy.with_name('token_trellis.cpp')]
    vendor_files = [p for p in vendor.rglob('*.py') if '__pycache__' not in p.parts]
    if not vendor_files: raise RuntimeError('Complete official Swin source is absent')
    code_bindings = {str(p.resolve()):sha(p) for p in [*source_files, *vendor_files]}
    train = RGBPopulation(args.image_cache, args.reference_registration, 'train')
    cal = RGBPopulation(args.image_cache, args.reference_registration, 'calibration')
    bindings = {**code_bindings, **train.bindings, **cal.bindings}
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    model = build_official(vendor); initial = model_hash(model)
    qualified = qualification(model, train, config, output, bindings)
    if args.qualification_only: return
    registration = dict(status='REGISTERED', config=config, bindings=bindings,
        qualification_sha256=sha(output/'qualification.json'), initial_model_sha256=initial,
        initialization='random', parameters=sum(p.numel() for p in model.parameters()),
        train_ids=train.ids, calibration_ids=cal.ids, torch=torch.__version__,
        cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
        numerical=dict(fp32=True, TF32=False, deterministic=True, threads=6, interop_threads=2),
        development_read=False, holdout_read=False, synthetic=False)
    register(output/'registration.json', registration); regsha = sha(output/'registration.json')
    optimizer = optimizer_for(model, config)
    order = SourceOrder(len(train), config['order_seed'])
    channel = torch.Generator().manual_seed(config['channel_seed'])
    state = dict(step=0, limit=20000, history=[], decisions=[], last_calibration=-1,
        checkpoint_serial=0, training_seconds=0., finished=False, learning_rate=1e-4, last_lr_step=0)
    if (output/'latest.json').exists():
        rec = read(output/'latest.json')
        if sha(rec['path']) != rec['sha256']: raise RuntimeError('Training resume checkpoint changed')
        payload = torch.load(rec['path'], map_location='cpu')
        if payload['registration_sha256'] != regsha: raise RuntimeError('Training resume registration changed')
        model.load_state_dict(payload['model'], strict=True); optimizer.load_state_dict(payload['optimizer'])
        order.load_state_dict(payload['order']); channel.set_state(payload['channel_rng']); restore_rng(payload['global_rng'])
        state = payload['state']
        if state['step'] != rec['step'] or not 0 <= state['step'] <= state['limit'] <= 240000:
            raise RuntimeError('Resumed step/budget differs')
        if any(g['lr'] != state['learning_rate'] for g in optimizer.param_groups):
            raise RuntimeError('Resumed learning-rate stage differs')
    restore_training_log(output/'training_log.jsonl',state['step'])
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))

    def checkpoint(reason):
        state['checkpoint_serial'] += 1
        path = output/'resume'/f"slot_{state['checkpoint_serial'] % 2}.pt"
        atomic_torch(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            order=order.state_dict(), channel_rng=channel.get_state(), global_rng=rng_state(),
            state=copy.deepcopy(state), registration_sha256=regsha))
        write(output/'latest.json', dict(path=str(path), sha256=sha(path), step=state['step'], reason=reason))

    try:
        while not state['finished']:
            if stop[0]: raise PauseRequested('User requested safe pause')
            if state['step'] % config['calibration_interval'] == 0 and state['last_calibration'] != state['step']:
                verify_bindings(code_bindings); available_gpu(); checkpoint('before_calibration')
                path = output/'checkpoints'/f"model_{state['step']:06d}.pt"
                if not path.exists():
                    atomic_torch(path, dict(model=model.state_dict(), step=state['step'], registration_sha256=regsha))
                else:
                    old = torch.load(path, map_location='cpu')
                    if old['registration_sha256'] != regsha or old['step'] != state['step'] or not exact_tree(old['model'], model.state_dict()):
                        raise RuntimeError('Existing calibration model differs')
                result = calibrate(model, cal, config, output, state['step'], path, regsha, stop)
                state['history'].append(result); state['last_calibration'] = state['step']
                write(output/'convergence.json', dict(history=state['history'], decisions=state['decisions'],
                    note='Actual paid calibration MSE only; no scientific convergence claim'))
                if state['step'] >= 20000 and state['step'] % 20000 == 0:
                    decision = plateau_decision(state['history'], state['step'], config['maximum_steps'], state['learning_rate'], state['last_lr_step'])
                    state['decisions'].append(decision); register(output/f"decision_{state['step']:06d}.json", decision)
                    state['limit'], state['finished'] = decision['next_limit'], not decision['extend']
                    if decision['lower_learning_rate']:
                        state['learning_rate'] = decision['next_learning_rate']; state['last_lr_step'] = state['step']
                        for group in optimizer.param_groups: group['lr'] = state['learning_rate']
                checkpoint('calibration_complete')
                if state['finished']: break
            if state['step'] >= state['limit']: raise RuntimeError('Uncalibrated training limit')
            if state['step'] % 100 == 0: available_gpu()
            indices = order.next(config['effective_batch_size'])
            N = list(RATES)[state['step'] % 2]
            snr = TRAIN_SNRS[int(torch.randint(len(TRAIN_SNRS), (1,), generator=channel))]
            noise = torch.randn((len(indices), layout(N)['data_N'], 2), generator=channel).cuda()
            image = train.batch(indices, 'cuda:0')
            started = time.perf_counter()
            loss = train_update(model, optimizer, image, N, snr, noise, config['microbatch_size'])
            torch.cuda.synchronize(); seconds = time.perf_counter()-started
            state['step'] += 1; state['training_seconds'] += seconds
            if state['step'] % 25 == 0:
                row = dict(step=state['step'], epoch=order.epoch, N=N, snr=snr, mse=loss, learning_rate=state['learning_rate'],
                    seconds=seconds, images_per_second=len(indices)/seconds,
                    memory_allocated=torch.cuda.memory_allocated(), memory_reserved=torch.cuda.memory_reserved())
                with (output/'training_log.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
                write(output/'status.json', dict(status='TRAINING', limit=state['limit'], **row, updated=time.time()))
                print('SWIN_TRAIN', state['step'], 'MSE', loss, 'seconds', seconds, flush=True)
            if state['step'] % config['resume_interval'] == 0: checkpoint('regular')
        selected = min(state['history'], key=lambda item:(item['mean_mse'], item['step']))
        selected = dict(selected, registration_sha256=regsha, completed_step=state['step'],
            budget_truncated=state['decisions'][-1]['budget_truncated'], selection='complete physical calibration MSE only',
            initialized_from_pretrained=False, development_read=False, holdout_read=False)
        register(output/'selected_swin.json', selected)
        write(output/'convergence.json', dict(history=state['history'], decisions=state['decisions']))
        completion = dict(status='SWIN_TRAINING_COMPLETE', selected_step=selected['step'], completed_step=state['step'],
            budget_truncated=selected['budget_truncated'], scientific_convergence_proven=False,
            registration_sha256=regsha, outputs={str(p):sha(p) for p in (output/'selected_swin.json',
                output/'registration.json', output/'qualification.json', output/'convergence.json', Path(selected['checkpoint']))},
            training_updates=state['step'], development_read=False, holdout_read=False, synthetic=False)
        register(output/'completion.json', completion); write(output/'status.json', completion)
    except PauseRequested as error:
        checkpoint('safe_pause'); write(output/'status.json', dict(status='PAUSED', step=state['step'], reason=str(error)))
        print('SWIN_SAFE_PAUSE', str(error), flush=True)
        raise SystemExit(75)
    except BaseException as error:
        write(output/'failure.json', dict(status='FAILED_REQUIRES_REVIEW', step=state['step'], error=repr(error),
            last_valid_checkpoint=str(output/'latest.json'), no_automatic_restart=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--vendor', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(HERE/'swin_training_protocol.json'))
    parser.add_argument('--image-cache', required=True); parser.add_argument('--reference-registration', required=True)
    parser.add_argument('--qualification-only', action='store_true')
    arguments=parser.parse_args()
    try:
        run(arguments)
    except PauseRequested as error:
        write(Path(arguments.output)/'status.json',dict(status='PAUSED',stage='startup',reason=str(error)))
        print('SWIN_SAFE_STARTUP_PAUSE',str(error),flush=True)
        raise SystemExit(75)

