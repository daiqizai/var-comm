"""Explicit user-selected 80k evaluation; the original training remains paused.

Only admission and output paths are adapted. Physical packets, decoder,
posterior sampler, metric functions, populations and numerical flags are the
original frozen implementations. No SWIN_TRAINING_COMPLETE receipt is created.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

STEP = 80000
CHECKPOINT_SHA = '8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21'
READY = 'USER_SELECTED_FIXED_MILESTONE_EVALUATION_READY'
VERSION = 'EXTERNAL-EVALUATION-20261004-USER-FIXED80K-R1'
HERE = Path(__file__).resolve().parent


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def require(value, message):
    if not value: raise RuntimeError(message)
def verify(bindings):
    for p, expected in bindings.items(): require(sha(p) == expected, 'Frozen binding changed: ' + p)
def identity(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def seal(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, 'Registered fixed80k receipt differs: ' + str(path)); return
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
def paths(root):
    root = Path(root).resolve(); base = root / 'outputs/EXTERNAL-COMPARISON-20261004'
    revision = base / 'fixed80k_revision'
    return dict(root=root, base=base, origin=base/'swin_training', revision=revision,
        view=revision/'training_view', runtime=base/'runtime', request=revision/'user_request.json',
        registration=revision/'fixed80k_registration.json', config=revision/'external_eval_config.json',
        output=revision/'evaluation', result=root/'results/external_comparison_20261004/fixed80k_revision/evaluation',
        hifi=revision/'hifi_qualification')


def validate_calibration(cal, origin, regsha):
    origin = Path(origin).resolve(); checkpoint = origin/'checkpoints/model_080000.pt'
    require(cal.get('step') == STEP and cal.get('checkpoint') == str(checkpoint)
        and cal.get('checkpoint_sha256') == CHECKPOINT_SHA, 'Not the exact 80k calibration checkpoint')
    require(cal.get('source_count') == 1000 and cal.get('rows') == 10000
        and cal.get('selection_uses_development') is False and cal.get('physical_metadata') is True,
        '80k calibration population/protocol differs')
    verify(cal['cells']); require(len(cal['cells']) == 10, '80k full calibration requires ten cells')
    cells = {}; source_ids = None
    for file in cal['cells']:
        path = Path(file).resolve()
        require(path.parent == origin/'calibration/080000', 'Calibration cell is outside exact 80k')
        cell = read(path); context = cell['context']; rows = cell['rows']
        require(cell.get('status') == 'COMPLETE' and cell.get('synthetic') is False,
            'Calibration cell is incomplete or synthetic')
        require(cell.get('payload_sha256') == identity({k:v for k,v in cell.items() if k != 'payload_sha256'}),
            'Calibration cell payload differs')
        require(context.get('step') == STEP and context.get('registration_sha256') == regsha
            and context.get('checkpoint_sha256') == CHECKPOINT_SHA and context.get('seed') == 4101,
            'Calibration cell provenance differs')
        pair = (context.get('N'), context.get('snr'))
        require(pair not in cells and pair[0] in (1024, 2048) and pair[1] in (1,4,7,10,13),
            'Calibration cell coverage differs')
        require(len(rows) == 1000 and len(context['source_ids']) == 1000 and len(set(context['source_ids'])) == 1000,
            'Calibration source count differs')
        if source_ids is None: source_ids = context['source_ids']
        require(context['source_ids'] == source_ids, 'Calibration source order differs')
        require(all(r['source_index'] == i and r['source_id'] == source_ids[i]
            and math.isfinite(r['mse']) and r['mse'] >= 0 for i,r in enumerate(rows)), 'Calibration source values differ')
        require(cell['mean_mse'] == math.fsum(r['mse'] for r in rows)/1000
            and cell['header_failures'] == sum(not r['header_accepted'] for r in rows), 'Calibration aggregate differs')
        cells[pair] = cell
    by_n = {str(n): math.fsum(cells[n,s]['mean_mse'] for s in (1,4,7,10,13))/5 for n in (1024,2048)}
    require(cal['mean_mse_by_N'] == by_n and cal['mean_mse'] == math.fsum(by_n.values())/2,
        'Calibration total differs')
    require(cal['mean_mse_by_cell'] == {f'N{n}_snr{s}':c['mean_mse'] for (n,s),c in cells.items()},
        'Calibration per-cell totals differ')
    return {str(checkpoint): CHECKPOINT_SHA, **cal['cells']}


def register(root):
    p = paths(root); origin = p['origin']; view = p['view']
    require(p['request'].is_file(), 'Explicit user request record is required')
    request = read(p['request'])
    require(request.get('requested_checkpoint_step') == STEP and request.get('training_must_not_resume_automatically') is True
        and request.get('selection_rule') == 'exact_user_requested_checkpoint_not_calibration_best'
        and request.get('scientific_convergence_claimed') is False, 'User request must select exact 80k without convergence or automatic training resume')
    status = read(origin/'status.json')
    require(status.get('status') == 'PAUSED' and status.get('step', 0) >= STEP, 'Original training must be safely paused after 80k')
    require(not (origin/'completion.json').exists(), 'This revision is for a paused, unfinished training run')
    if p['registration'].exists():
        validate_selection(view, root); return read(p['registration'])
    view.mkdir(parents=True, exist_ok=True)
    for name in ('registration.json', 'qualification.json'):
        target = view/name
        if target.exists(): require(sha(target) == sha(origin/name), 'Original receipt copy differs')
        else: shutil.copyfile(origin/name, target)
    regsha = sha(view/'registration.json'); calpath = origin/'calibration/080000/completion.json'; cal = read(calpath)
    inputs = validate_calibration(cal, origin, regsha)
    require(sha(cal['checkpoint']) == CHECKPOINT_SHA, 'Exact 80k checkpoint bytes differ')
    require(read(view/'qualification.json').get('status') == 'PASS', 'Original training qualification did not pass')
    snapshot = p['revision']/'origin_pause_snapshot.json'; seal(snapshot, status)
    selected = dict(cal, registration_sha256=regsha, selected_by='explicit_user_fixed_step',
        selection='User selected exact step 80000; calibration is verification, not best-checkpoint selection',
        selected_step=STEP, completed_step=status['step'], evaluated_checkpoint_step=STEP, actual_training_pause_step=status['step'],
        budget_truncated=True, budget_truncation_reason='user_requested_pause_before_registered_convergence',
        training_finished=False, scientific_convergence_proven=False, development_read=False, holdout_read=False)
    seal(view/'selected_swin.json', selected)
    inputs.update({str(f):sha(f) for f in (p['request'],snapshot,calpath,origin/'registration.json',origin/'qualification.json',
        p['runtime']/'external_eval_config.json', HERE/'fixed80k_adapter.py')})
    pause_proof = p['revision']/'pause_verified.json'
    require(pause_proof.is_file(), 'Separate original-process exit verification is required')
    paused = read(pause_proof)
    require(len(paused.get('processes',{})) == 4 and all(v.get('exists') is False for v in paused['processes'].values())
        and paused.get('swin_training/status.json') == status
        and paused.get('checkpoint_80000',{}).get('sha256') == CHECKPOINT_SHA,
        'All original pipeline processes must have exited and the 80k checkpoint must be verified')
    inputs[str(pause_proof)] = sha(pause_proof)
    receipt = dict(status=READY, selected_step=STEP, selected_checkpoint_sha256=CHECKPOINT_SHA,
        original_training_output=str(origin), training_view=str(view), actual_training_pause_step=status['step'],
        selection='explicit_user_fixed_step', calibration_verification_only=True, synthetic=False,
        training_finished=False, scientific_convergence_proven=False, user_requested_pause=True,
        budget_truncated=True, training_updates=0, policy_selection_updates=0, development_read=False, holdout_read=False,
        input_bindings=inputs, outputs={str(view/n):sha(view/n) for n in ('selected_swin.json','registration.json','qualification.json')})
    seal(p['registration'], receipt)
    completion = dict(status=READY, registration_sha256=regsha, synthetic=False, selected_step=STEP,
        selected_checkpoint_sha256=CHECKPOINT_SHA, selected_checkpoint_training_steps=STEP,
        actual_training_pause_step=status['step'], completed_step=status['step'], training_finished=False, scientific_convergence_proven=False,
        user_requested_pause=True, budget_truncated=True, development_read=False, holdout_read=False,
        outputs={**receipt['outputs'],str(p['registration']):sha(p['registration'])})
    seal(view/'completion.json', completion)
    config = read(p['runtime']/'external_eval_config.json')
    config.update(version=VERSION, training_output=str(view), output=str(p['output']), result=str(p['result']),
        hifi_qualification_path=str(p['hifi']/'qualification.json'), fixed80k_registration=str(p['registration']))
    seal(p['config'], config)
    validate_selection(view, root)
    return receipt


def validate_selection(training_output, root=None):
    view = Path(training_output).resolve()
    if root is None: root = view.parents[3]
    p = paths(root); require(view == p['view'], 'Unregistered fixed80k training view')
    registration = read(p['registration']); done = read(view/'completion.json'); selected = read(view/'selected_swin.json')
    for item in (registration, done):
        require(item.get('status') == READY and item.get('synthetic') is False and item.get('selected_step') == STEP
            and item.get('selected_checkpoint_sha256') == CHECKPOINT_SHA and item.get('training_finished') is False
            and item.get('scientific_convergence_proven') is False and item.get('user_requested_pause') is True
            and item.get('budget_truncated') is True, 'Fixed milestone admission or non-convergence disclosure differs')
    verify(registration['input_bindings']); verify(registration['outputs']); verify(done['outputs'])
    require(selected.get('step') == STEP and selected.get('checkpoint_sha256') == CHECKPOINT_SHA
        and selected.get('selected_by') == 'explicit_user_fixed_step' and selected.get('development_read') is False
        and selected.get('holdout_read') is False and selected.get('training_finished') is False
        and selected.get('scientific_convergence_proven') is False, 'Exact user-selected checkpoint differs')
    require(selected['checkpoint'] == str(p['origin']/'checkpoints/model_080000.pt'), 'Checkpoint is outside original exact milestone')
    require(selected['registration_sha256'] == done['registration_sha256'] == sha(view/'registration.json'), 'Training registration differs')
    return selected, {str(f):sha(f) for f in (p['registration'],view/'completion.json',view/'selected_swin.json',view/'registration.json',view/'qualification.json')}


def load_selected(training_output, vendor, device='cuda:0'):
    import torch
    from swin_model import build_official
    selected, _ = validate_selection(training_output)
    registration = read(Path(training_output)/'registration.json')
    for file in Path(vendor).resolve().rglob('*.py'):
        if '__pycache__' not in file.parts:
            require(registration['bindings'].get(str(file.resolve())) == sha(file), 'Original Swin vendor changed: '+str(file))
    payload = torch.load(selected['checkpoint'], map_location='cpu')
    require(payload['step'] == STEP and payload['registration_sha256'] == selected['registration_sha256'], '80k model payload metadata differs')
    model = build_official(vendor, device); model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    return model, selected


def install(root):
    """Install explicit, checksum-bound admission adapters in this process only."""
    p = paths(root); sys.path.insert(0, str(p['runtime']))
    import external_eval as evaluation
    import swin_replay
    original_validate, original_bindings = evaluation.validate_config, evaluation.code_bindings
    def validate_config(config):
        expected = read(p['config']); require(config == expected, 'Fixed80k evaluation config differs')
        require(config['version'] == VERSION and Path(config['fixed80k_registration']) == p['registration'], 'Fixed80k revision differs')
        original = read(p['runtime']/'external_eval_config.json')
        copied = dict(config)
        for key in ('version','training_output','output','result','hifi_qualification_path'): copied[key] = original[key]
        copied.pop('fixed80k_registration')
        require(copied == original, 'Original physical, sampler or population protocol changed')
        original_validate(copied); validate_selection(config['training_output'], root)
        return p['root']
    def code_bindings():
        return {**original_bindings(), str(HERE/'fixed80k_adapter.py'):sha(HERE/'fixed80k_adapter.py'),
            str(p['registration']):sha(p['registration'])}
    def selected_gate(config):
        import torch
        validate_config(config)
        selected, bindings = validate_selection(config['training_output'], root)
        train = Path(config['training_output']); reg = read(train/'registration.json')
        qualification = Path(config['hifi_qualification_path']); proof = read(qualification)
        require(proof.get('status') == 'HIFI_SWIN_QUALIFICATION_PASS'
            and all(proof.get(k) is True for k in ('full_input_gradient_pass','native_forward_parity_pass','model_parameters_unchanged'))
            and proof.get('selected_checkpoint_sha256') == CHECKPOINT_SHA, 'Fixed80k HiFi real qualification missing')
        require(proof.get('torch') == torch.__version__ and proof.get('cuda') == torch.version.cuda, 'Posterior qualification runtime differs')
        from hifi_swin_sampler import ADM_SHA256
        require(proof.get('adm_checkpoint_sha256') == ADM_SHA256 and sha(config['adm_checkpoint']) == ADM_SHA256, 'Frozen ADM differs')
        own = code_bindings()
        for name in ('hifi_swin_operator.py','hifi_swin_sampler.py','hifi_swin_schedule.py','hifi_swin_tests.py','hifi_swin_schedule_tests.py'):
            file = str(p['runtime']/name)
            require(proof.get('source_bindings',{}).get(file) == own[file], 'HiFi source differs: '+name)
        for name in ('swin_model.py','swin_protocol.py','swin_replay.py'):
            file = str(p['runtime']/name)
            require(reg['bindings'].get(file) == own[file], 'Original physical Swin adapter changed: '+name)
        for file in (HERE/'fixed80k_adapter.py',p['registration']):
            require(proof.get('source_bindings',{}).get(str(file)) == sha(file), 'Qualification lacks fixed80k adapter binding')
        verify(proof['source_bindings']); verify(proof['input_bindings'])
        return selected, {**bindings,str(Path(selected['checkpoint'])):CHECKPOINT_SHA,
            str(qualification):sha(qualification),str(Path(config['adm_checkpoint'])):ADM_SHA256}
    evaluation.validate_config = validate_config; evaluation.code_bindings = code_bindings
    evaluation.selected_gate = selected_gate; swin_replay.load_selected = load_selected
    return evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--stage', required=True, choices=('register','qualify','reconstruct','score'))
    parser.add_argument('--launch-id', default=None)
    args = parser.parse_args(); p = paths(args.root)
    if args.stage == 'register': print(json.dumps(register(args.root), indent=2)); return
    evaluation = install(args.root)
    if args.stage == 'qualify':
        import hifi_qualification as qualification
        original = qualification.source_bindings
        def source_bindings(*a, **kw):
            return {**original(*a, **kw),str(HERE/'fixed80k_adapter.py'):sha(HERE/'fixed80k_adapter.py'),
                str(p['registration']):sha(p['registration'])}
        qualification.source_bindings = source_bindings; qualification.load_selected = load_selected
        controller = read(p['base']/'controller_config.json'); train=controller['training']; hifi=controller['hifi']
        qualification.run(argparse.Namespace(root=str(p['root']),swin_output=str(p['view']),swin_vendor=train['vendor'],
            diffcom_vendor=hifi['diffcom_vendor'],adm_checkpoint=hifi['adm_checkpoint'],image_cache=train['image_cache'],
            reference_registration=train['reference_registration'],output=str(p['hifi'])))
    else:
        sys.argv = [str(HERE/'fixed80k_adapter.py'),'--config',str(p['config']),'--stage',args.stage]
        if args.launch_id: sys.argv += ['--launch-id',args.launch_id]
        evaluation.main()


if __name__ == '__main__': main()
