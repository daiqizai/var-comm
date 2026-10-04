"""Detached, receipt-gated Swin qualification/training and optional evaluation.

This module does not import Torch, mutate Git, or signal an earlier experiment.
All paths/interpreters/environment values are explicit in a frozen JSON config.
Run: python external_controller.py --config /absolute/config.json [--detach]
Exit 75 is a safe pause, not permission to retry a scientific failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


VERSION = 'EXTERNAL-CONTROLLER-20261004-R1'
OLD_NAME = 'HISTORICAL-METRICS-R3-20261003'
HERE = Path(__file__).resolve().parent


class Waiting(RuntimeError):
    pass


class Paused(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def register(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Immutable controller record changed: '+str(path))
    else:
        write(path, value)


def verify(bindings, *, nonempty=True):
    require(isinstance(bindings, dict) and (bindings or not nonempty), 'Missing file bindings')
    for name, digest in bindings.items():
        require(Path(name).is_absolute() and re.fullmatch('[0-9a-f]{64}', str(digest)),
                'Malformed bound file identity: '+str(name))
        require(Path(name).is_file() and sha(name) == digest, 'Bound file changed: '+name)


def absolute(value):
    require(isinstance(value, str) and Path(value).is_absolute(), 'Explicit absolute path required')
    require('..' not in Path(value).parts, 'Parent traversal is not an explicit canonical dependency')
    return Path(value)


def process(pid):
    """Read Linux process identity without mistaking a reused PID for its owner."""
    require(type(pid) is int and pid > 1, 'Invalid process PID')
    folder = Path('/proc')/str(pid)
    try:
        raw = (folder/'stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        command = (folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        return dict(pid=pid, start_ticks=fields[19], state=fields[0],
                    command=[v for v in command if v])
    except FileNotFoundError:
        return None


def live(record, reader=process):
    require(type(record.get('pid')) is int and record['pid'] > 1
            and str(record.get('start_ticks', '')).isdecimal(), 'Missing exact PID/start_ticks')
    current = reader(record['pid'])
    return bool(current and current.get('state') != 'Z'
                and str(current.get('start_ticks')) == str(record['start_ticks']))


def gpu_pids():
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    return parse_gpu_pids(result.stdout)


def parse_gpu_pids(value):
    values = [line.strip() for line in value.splitlines() if line.strip()]
    require(all(v.isdecimal() and int(v) > 1 for v in values), 'Unreadable GPU ownership query')
    return [int(v) for v in values]


def checked_publication(publication):
    require(publication.get('status') == 'PUSHED' and publication.get('checks') == 'PASS'
            and re.fullmatch('[0-9a-f]{40}', str(publication.get('commit', '')))
            and publication.get('remote_commit') == publication['commit'],
            'Historical results lack a checked push with exact remote commit')


def old_gate(config, reader=process):
    """Require the complete old queue, published results and all old owner exits."""
    old = config['old']; folder = absolute(old['output'])
    require(folder == absolute(config['root'])/'outputs'/OLD_NAME, 'Unexpected historical output')
    pinned = {str(folder/(key+'.json')):old[key+'_sha256']
              for key in ('queue_registration', 'controller_launch', 'controller_dispatch')}
    verify(pinned)
    queue, launch, dispatch = (read(folder/(key+'.json'))
                              for key in ('queue_registration', 'controller_launch', 'controller_dispatch'))
    qsha = old['queue_registration_sha256']
    require(queue.get('status') == 'REGISTERED' and queue.get('training_updates') == 0
            and queue.get('policy_selection_updates') == 0, 'Historical queue scope differs')
    require(launch.get('queue_registration_sha256') == qsha
            and dispatch.get('queue_registration_sha256') == qsha
            and dispatch.get('status') == 'CONTROLLER_LAUNCHED'
            and launch.get('source_bindings') == queue['source_bindings']
            and (launch.get('pid'), str(launch.get('start_ticks'))) ==
                (dispatch.get('pid'), str(dispatch.get('start_ticks'))),
            'Historical dispatch/launch/queue identities differ')
    if old.get('phase2_recovery'):
        return phase2_recovery_gate(config,reader,pinned,queue,launch)
    complete_path, publication_path = folder/'completion.json', folder/'results_publication.json'
    owner_live = live(launch, reader)
    if not complete_path.is_file() or not publication_path.is_file():
        if owner_live:
            raise Waiting('Historical R3 is still running or publishing')
        raise RuntimeError('Historical owner exited without complete published results; review required')
    done, publication = read(complete_path), read(publication_path)
    checked_publication(publication)
    require(done.get('status') == 'HISTORICAL_METRICS_COMPLETE' and done.get('stop') is True
            and done.get('training_updates') == done.get('policy_selection_updates') == 0
            and done.get('publication') == publication
            and done.get('queue_registration_sha256') == qsha
            and done.get('source_bindings') == queue['source_bindings']
            and (done.get('pid'), str(done.get('start_ticks'))) ==
                (launch['pid'], str(launch['start_ticks']))
            and publication.get('phase') == 'results'
            and publication.get('queue_registration_sha256') == qsha
            and publication.get('runtime_source_bindings') == queue['source_bindings'],
            'Historical full completion/publication identity differs')
    if owner_live:
        raise Waiting('Historical publication is complete; waiting for its controller to exit')
    for path in sorted((folder/'launches').glob('*.json')):
        previous = read(path)
        require(previous.get('queue_registration_sha256') == qsha, 'Old child queue identity differs')
        if live(previous, reader):
            raise Waiting('Historical child is still alive: '+str(previous.get('job')))
    verify(queue['source_bindings'])
    for field in ('inputs', 'published_files', 'source_bindings'):
        verify(publication.get(field))
    studies = [item['study'] for item in queue['jobs']]
    require(len(studies) == len(set(studies)) and len(studies) == 12, 'Incomplete historical selected queue')
    receipts = {}
    for study in studies:
        inherited = queue.get('inherited_studies', {}).get(study)
        study_folder = absolute(inherited['out']) if inherited else folder/study
        if inherited:
            require(study_folder == absolute(config['root'])/'outputs'/'HISTORICAL-METRICS-R2-20261003'/study,
                    'Inherited historical study path differs')
        path = study_folder/'completion.json'; receipt = read(path)
        require(publication['inputs'].get(str(path)) == sha(path)
                and receipt.get('status') == 'HISTORICAL_STUDY_METRICS_COMPLETE'
                and receipt.get('study') == study and receipt.get('sources') == 100
                and receipt.get('parity_passed') is True and receipt.get('synthetic') is False
                and receipt.get('training_updates') == receipt.get('policy_selection_updates') == 0,
                'Unfinished or unbound historical study: '+study)
        receipts[str(path)] = sha(path)
    return dict(status='HISTORICAL_R3_COMPLETE_PUSHED_AND_EXITED', commit=publication['commit'],
                bindings={**pinned, str(complete_path):sha(complete_path),
                          str(publication_path):sha(publication_path), **receipts},
                original_processes_signalled=False)


def phase2_recovery_gate(config,reader,pinned,original,old_launch):
    """Admit the separately registered 81+19-source completion; never rewrite R3."""
    root=absolute(config['root']);old_folder=root/'outputs'/OLD_NAME
    option=config['old']['phase2_recovery'];folder=absolute(option['output'])
    require(folder==root/'outputs/HISTORICAL-PHASE2-RECOVERY-20261004-R1','Unknown historical recovery revision')
    qp=folder/'queue_registration.json'
    require(sha(qp)==option['queue_registration_sha256'],'Recovery queue identity differs')
    queue=read(qp);proof=queue.get('phase2_recovery',{})
    require(queue.get('status')=='REGISTERED' and queue.get('training_updates')==queue.get('policy_selection_updates')==0
        and queue.get('jobs')==original['jobs'] and queue.get('coverage_manifest')==original.get('coverage_manifest')
        and queue.get('contrasts')==original.get('contrasts')
        and queue.get('inherited_studies')==original.get('inherited_studies')
        and proof.get('original_queue_sha256')==config['old']['queue_registration_sha256']
        and proof.get('inherited_sources')==81 and proof.get('remaining_sources')==19
        and proof.get('scope_changed') is False and proof.get('new_metric_offset_applied') is False
        and proof.get('strict_grid_parity_tolerances_changed') is False
        and queue.get('replay_adapter_overrides')=={'OPTIONAL_PHASE2_MAIN':'historical_phase2_alias'},
        'Recovery scope or original R3 identity differs')
    four={'SELECTED_NOISELESS_REFERENCES','OPTIONAL_RX_STEP2_A','OPTIONAL_RX_STEP2_B','OPTIONAL_H6_13DB'}
    require(set(queue.get('completed_r3_studies',{}))==four,'Recovery must inherit exactly the other four R3 studies')
    for owner in [old_launch,*[read(p) for p in [old_folder/'bootstrap_launch.json',*sorted((old_folder/'launches').glob('*.json'))]]]:
        if live(owner,reader):raise Waiting('Waiting for original R3/bootstrap/child exit before recovery handoff')
    completion_path=folder/'completion.json';publication_path=folder/'results_publication.json'
    launch_path=folder/'controller_launch.json'
    if not completion_path.is_file() or not publication_path.is_file():
        if launch_path.exists() and not live(read(launch_path),reader):
            raise RuntimeError('Recovery owner exited without complete published results; review required')
        raise Waiting('Waiting for independently registered Phase2 recovery and normal publication')
    launch=read(launch_path);done=read(completion_path);publication=read(publication_path);checked_publication(publication)
    qsha=sha(qp)
    require(launch.get('queue_registration_sha256')==qsha and launch.get('source_bindings')==queue['source_bindings']
        and done.get('status')=='HISTORICAL_METRICS_COMPLETE' and done.get('stop') is True
        and done.get('training_updates')==done.get('policy_selection_updates')==0
        and done.get('publication')==publication and done.get('queue_registration_sha256')==qsha
        and done.get('source_bindings')==queue['source_bindings']
        and (done.get('pid'),str(done.get('start_ticks')))==(launch['pid'],str(launch['start_ticks']))
        and publication.get('phase')=='results' and publication.get('queue_registration_sha256')==qsha
        and publication.get('runtime_source_bindings')==queue['source_bindings'],'Recovery completion/publication differs')
    owners=[launch,*[read(p) for p in sorted((folder/'launches').glob('*.json'))]]
    for name in ('bootstrap_launch.json','controller_dispatch.json'):
        if (folder/name).exists():owners.append(read(folder/name))
    for owner in owners:
        if live(owner,reader):raise Waiting('Recovery is published; waiting for its final owner to exit')
    verify(queue['source_bindings']);verify(queue['input_proof_bindings']);verify(queue['shared_metric_bindings'])
    dedicated=[root/'experiments/historical-phase2-recovery-20261004-r1',
               root/'results/historical_phase2_recovery_20261004_r1']
    report_path=root/'reports/historical_phase2_recovery_20261004_r1.md'
    require(all(Path(p)==report_path or any(base in Path(p).parents for base in dedicated)
                for p in publication.get('published_files',{})),
            'Recovery publication may bind only immutable dedicated artifacts, never shared status/release files')
    for field in ('inputs','published_files','source_bindings'):verify(publication.get(field))
    auditpath=folder/'phase2_completion_audit.json';audit=read(auditpath)
    require(publication['inputs'].get(str(auditpath))==sha(auditpath)
        and audit.get('status')=='PHASE2_RECOVERY_100_SOURCES_VERIFIED' and audit.get('sources')==100
        and audit.get('rows')==4500 and audit.get('inherited_sources')==81 and audit.get('new_sources')==19
        and audit.get('inherited_rows')==3645 and audit.get('new_rows')==855
        and audit.get('new_pure_strict_grid_proofs')==285 and audit.get('historical_alias_rows_disclosed')==1500
        and audit.get('new_metric_offset_applied') is False and audit.get('original_files_written') is False
        and audit.get('synthetic') is False,'Complete 81+19 strict-grid audit required')
    verify(audit['bindings'])
    studies=[item['study'] for item in queue['jobs']]
    require(len(studies)==len(set(studies))==12,'Recovery omitted original selected studies')
    receipts={};total=0
    for study in studies:
        if study in queue.get('inherited_studies',{}):
            item=queue['inherited_studies'][study];study_folder=absolute(item['out'])
            require(study_folder==root/'outputs/HISTORICAL-METRICS-R2-20261003'/study,'R2 inheritance path differs')
        elif study in four:
            item=queue['completed_r3_studies'][study];study_folder=absolute(item['out'])
            require(study_folder==old_folder/study,'R3 inheritance path differs')
        else:
            require(study=='OPTIONAL_PHASE2_MAIN','Unregistered recovery study');study_folder=folder/study
        p=study_folder/'completion.json';receipt=read(p)
        require(publication['inputs'].get(str(p))==sha(p)
            and receipt.get('status')=='HISTORICAL_STUDY_METRICS_COMPLETE' and receipt.get('study')==study
            and receipt.get('sources')==100 and receipt.get('parity_passed') is True
            and receipt.get('synthetic') is False and receipt.get('training_updates')==receipt.get('policy_selection_updates')==0,
            'Incomplete recovered historical study: '+study)
        receipts[str(p)]=sha(p);total+=receipt['frames']
    require(total==56700,'Historical selected row count changed')
    return dict(status='HISTORICAL_PHASE2_RECOVERY_COMPLETE_PUSHED_AND_ALL_OWNERS_EXITED',commit=publication['commit'],
        bindings={**pinned,str(qp):qsha,str(launch_path):sha(launch_path),str(completion_path):sha(completion_path),
            str(publication_path):sha(publication_path),str(auditpath):sha(auditpath),**receipts},
        original_r3_complete_claimed=False,original_processes_signalled=False)


def required_sources(config):
    train = config['training']; script = absolute(train['script'])
    require(script.name == 'swin_train.py', 'Unexpected training entry')
    paths = [Path(__file__).resolve(), script, absolute(train['config']),
             absolute(train['reference_registration']), absolute(train['python'])]
    paths += sorted(script.parent.glob('swin_*.py'))
    vendor = absolute(train['vendor'])
    vendor_files = sorted(p for p in vendor.rglob('*') if p.is_file()
                          and p.suffix in ('.py', '.json', '.yaml', '.yml')
                          and '__pycache__' not in p.parts and '.git' not in p.parts)
    require(vendor_files, 'Official Swin vendor source is absent')
    paths += vendor_files
    hifi = config.get('hifi')
    if hifi:
        paths += [absolute(hifi[key]) for key in ('python', 'preflight_script', 'qualification_script', 'adm_checkpoint')]
        require(Path(hifi['preflight_script']).name == 'hifi_preflight.py'
                and Path(hifi['qualification_script']).name == 'hifi_qualification.py', 'Unexpected HiFi entry')
        paths += sorted(Path(hifi['preflight_script']).parent.glob('hifi_*.py'))
        diffcom_files = sorted(p for p in absolute(hifi['diffcom_vendor']).rglob('*') if p.is_file()
                              and p.suffix in ('.py', '.json', '.yaml', '.yml')
                              and '__pycache__' not in p.parts and '.git' not in p.parts)
        require(diffcom_files, 'Official diffusion source is absent')
        paths += diffcom_files
    evaluation = config.get('evaluation')
    if evaluation:
        entry = absolute(evaluation['script'])
        require(entry.name == 'external_eval.py', 'Unexpected evaluation entry')
        paths += [entry, absolute(evaluation['config']), absolute(evaluation['reconstruction_python']),
                  absolute(evaluation['score_python'])]
        paths += [p for pattern in ('hifi_*.py', 'step0_*.py', 'external_*.py')
                  for p in sorted(entry.parent.glob(pattern))]
    return paths


def validate_config(config):
    require(config.get('version') == VERSION and config.get('mode') in ('train', 'qualification_only'),
            'Controller version/mode is not registered')
    root, output = absolute(config['root']), absolute(config['output'])
    require(root != output and root in output.parents and output != root/'outputs'/OLD_NAME
            and root/'outputs'/OLD_NAME not in output.parents, 'Controller output is not isolated')
    require(type(config.get('poll_seconds', 15)) is int and 1 <= config.get('poll_seconds', 15) <= 60,
            'poll_seconds must be 1..60')
    verify(config.get('bindings'))
    for path in required_sources(config):
        require(config['bindings'].get(str(path)) == sha(path), 'Unregistered launch dependency: '+str(path))
    training = config['training']; training_output = absolute(training['output'])
    require(root in training_output.parents and training_output != output
            and output not in training_output.parents and training_output not in output.parents
            and training_output != root/'outputs'/OLD_NAME and root/'outputs'/OLD_NAME not in training_output.parents,
            'Training output must be an independent new directory')
    require(absolute(training['image_cache']).is_dir(), 'Explicit original RGB cache is absent')
    validate_environment(training['environment'])
    require(config['mode'] == 'qualification_only' or config.get('hifi'),
            'Formal training requires the registered early HiFi preflight')
    if config.get('hifi'):
        hifi = config['hifi']; validate_environment(hifi['environment'])
        for key in ('preflight_output', 'qualification_output'):
            path = absolute(hifi[key])
            require(root in path.parents and path not in (output, training_output)
                    and root/'outputs'/OLD_NAME not in path.parents, 'HiFi output is not isolated')
        require(hifi['preflight_output'] != hifi['qualification_output'], 'Early and selected HiFi checks must be distinct')
    if config.get('evaluation'):
        evaluation = config['evaluation']; evaluation_output = absolute(evaluation['output'])
        require(evaluation_output not in (output, training_output) and root in evaluation_output.parents
                and root/'outputs'/OLD_NAME not in evaluation_output.parents,
                'Evaluation output must be isolated')
        specification = read(evaluation['config'])
        require(specification.get('root') == config['root']
                and specification.get('output') == evaluation['output']
                and specification.get('training_output') == training['output']
                and specification.get('hifi_qualification_path') == str(Path(config['hifi']['qualification_output'])/'qualification.json')
                and specification.get('swin_vendor') == training['vendor']
                and specification.get('diffcom_vendor') == config['hifi']['diffcom_vendor']
                and specification.get('adm_checkpoint') == config['hifi']['adm_checkpoint'],
                'Evaluation paths do not match frozen controller configuration')
        validate_environment(evaluation['reconstruction_environment'])
        validate_environment(evaluation['score_environment'])
    return config


def validate_environment(values):
    require(isinstance(values, dict) and values.get('CUDA_VISIBLE_DEVICES') == '0',
            'Only explicitly registered GPU0 launches are supported')
    require(all(isinstance(k, str) and isinstance(v, str) and '\0' not in k+v
                and '=' not in k for k, v in values.items()), 'Malformed launch environment')
    require(not any(k in values for k in ('HISTORICAL_TORCH_METADATA_MANIFEST', 'LD_PRELOAD')),
            'Earlier metadata workaround or unregistered preload is forbidden')


def stage_spec(config, stage, token):
    training = config['training']
    if stage in ('qualification', 'training'):
        command = [training['python'], '-u', training['script'], '--root', config['root'],
                   '--vendor', training['vendor'], '--output', training['output'],
                   '--config', training['config'], '--image-cache', training['image_cache'],
                   '--reference-registration', training['reference_registration']]
        if stage == 'qualification':
            command.append('--qualification-only')
        return command, training['environment'], Path(training['output'])
    if stage in ('hifi_preflight', 'hifi_qualification'):
        hifi = config['hifi']; prefix = 'preflight' if stage == 'hifi_preflight' else 'qualification'
        command = [hifi['python'], '-u', hifi[prefix+'_script'], '--root', config['root'],
                   '--swin-output', training['output'], '--swin-vendor', training['vendor'],
                   '--diffcom-vendor', hifi['diffcom_vendor'], '--adm-checkpoint', hifi['adm_checkpoint'],
                   '--image-cache', training['image_cache'], '--reference-registration', training['reference_registration'],
                   '--output', hifi[prefix+'_output']]
        if stage == 'hifi_preflight':
            command += ['--config', training['config']]
        return command, hifi['environment'], Path(hifi[prefix+'_output'])
    require(stage in ('reconstruct', 'score') and config.get('evaluation'),
            'Evaluation has no frozen real interface')
    evaluation = config['evaluation']; kind = 'reconstruction' if stage == 'reconstruct' else 'score'
    command = [evaluation[kind+'_python'], '-u', evaluation['script'], '--config', evaluation['config'],
               '--stage', stage, '--launch-id', token]
    return command, evaluation[kind+'_environment'], Path(evaluation['output'])


def failure_files(config, stage):
    if stage in ('qualification', 'training'):
        return [Path(config['training']['output'])/'failure.json']
    if stage in ('hifi_preflight', 'hifi_qualification'):
        _, _, out = stage_spec(config, stage, '')
        return [out/'failure.json']
    out = Path(config['evaluation']['output'])
    return [out/'failure.json', out/(stage+'_failure.json')]


def reject_scientific_failure(config, stage):
    for path in failure_files(config, stage):
        require(not path.exists(), 'Scientific failure requires review, not automatic retry: '+str(path))


def completion(config, stage):
    """A process exit alone never establishes scientific success."""
    reject_scientific_failure(config, stage)
    if stage == 'qualification':
        out = Path(config['training']['output']); path = out/'qualification.json'
        if not path.exists():
            return None
        receipt = read(path); train_config = read(config['training']['config'])
        require(receipt.get('status') == 'PASS' and receipt.get('synthetic') is False
                and receipt.get('real_training_probe') is True and receipt.get('formal_training_updates') == 0
                and receipt.get('populated_adam_resume_exact') is True
                and receipt.get('initial_model_and_rng_restored') is True
                and receipt.get('config_sha256') == identity(train_config), 'Invalid real Swin qualification')
        verify(receipt.get('bindings'))
        for name, digest in config['bindings'].items():
            if name in receipt['bindings']:
                require(receipt['bindings'][name] == digest, 'Qualification launch/source identity differs')
        require(receipt['bindings'].get(config['training']['script']) == sha(config['training']['script'])
                and receipt['bindings'].get(config['training']['config']) == sha(config['training']['config']),
                'Qualification did not bind the registered trainer/config')
    elif stage == 'training':
        out = Path(config['training']['output']); path = out/'completion.json'
        if not path.exists():
            return None
        receipt = read(path)
        require(receipt.get('status') == 'SWIN_TRAINING_COMPLETE' and receipt.get('synthetic') is False
                and receipt.get('development_read') is False and receipt.get('holdout_read') is False
                and type(receipt.get('training_updates')) is int and receipt['training_updates'] > 0
                and receipt.get('completed_step') == receipt['training_updates']
                and receipt.get('registration_sha256') == sha(out/'registration.json'),
                'Invalid Swin training completion')
        verify(receipt.get('outputs'))
        reg = read(out/'registration.json'); verify(reg.get('bindings'))
        require(reg.get('status') == 'REGISTERED' and reg.get('initialization') == 'random'
                and reg.get('config') == read(config['training']['config'])
                and reg.get('qualification_sha256') == sha(out/'qualification.json')
                and reg.get('development_read') is False and reg.get('holdout_read') is False
                and reg.get('synthetic') is False, 'Training registration or population scope differs')
        completion(config, 'qualification')
        selected_path = out/'selected_swin.json'; selected = read(selected_path)
        require(receipt['outputs'].get(str(selected_path)) == sha(selected_path)
                and selected.get('registration_sha256') == sha(out/'registration.json')
                and selected.get('step') == receipt.get('selected_step')
                and receipt['outputs'].get(selected['checkpoint']) == sha(selected['checkpoint']),
                'Selected trained checkpoint identity differs')
    elif stage in ('hifi_preflight', 'hifi_qualification'):
        command, _, out = stage_spec(config, stage, '')
        path = out/('preflight.json' if stage == 'hifi_preflight' else 'qualification.json')
        if not path.exists():
            return None
        receipt = read(path)
        request = {command[i][2:].replace('-', '_'):str(Path(command[i+1]).resolve())
                   for i in range(3, len(command), 2)}
        require(receipt.get('request') == request and receipt.get('synthetic') is False
                and receipt.get('scientific_result') is False and receipt.get('qualification_only') is True
                and receipt.get('full_input_gradient_pass') is True
                and receipt.get('native_forward_parity_pass') is True
                and receipt.get('model_parameters_unchanged') is True
                and receipt.get('source_role') == 'original_calibration'
                and receipt.get('source_indices') == [0, 1]
                and receipt.get('development_read') is False and receipt.get('holdout_read') is False,
                'HiFi operator/gradient/real ADM qualification differs')
        if stage == 'hifi_preflight':
            original = Path(config['training']['output'])/'qualification.json'; admitted = read(original)
            require(receipt.get('status') == 'ENGINEERING_PREFLIGHT' and receipt.get('passed') is True
                    and receipt.get('selected_checkpoint_qualification') is False
                    and receipt.get('training_qualification_replacement') is False
                    and receipt.get('initialization') == 'random' and receipt.get('formal_training_updates') == 0
                    and receipt.get('swin_qualification_sha256') == sha(original)
                    and receipt.get('initial_model_sha256') == admitted.get('initial_model_sha256'),
                    'Early engineering preflight cannot replace selected checkpoint qualification')
        else:
            selected = read(Path(config['training']['output'])/'selected_swin.json')
            require(receipt.get('status') == 'HIFI_SWIN_QUALIFICATION_PASS'
                    and receipt.get('selected_checkpoint_sha256') == sha(selected['checkpoint'])
                    and receipt.get('development_read') is False and receipt.get('holdout_read') is False,
                    'HiFi qualification did not use the selected checkpoint and calibration population')
        verify(receipt.get('source_bindings')); verify(receipt.get('input_bindings'))
        require(receipt['source_bindings'].get(command[2]) == sha(command[2]), 'HiFi entry source identity differs')
        probes = receipt.get('real_ADM_probes', [])
        require(len(probes) == 2 and {item.get('N') for item in probes} == {1024, 2048}
                and all(item.get('sampler_receipt', {}).get('NFE') == 2
                        and item['sampler_receipt'].get('diagnostic_probe') is True for item in probes),
                'HiFi engineering checks require real two-step probes at both budgets')
    else:
        out = Path(config['evaluation']['output'])
        path = out/('reconstruction_completion.json' if stage == 'reconstruct' else 'completion.json')
        if not path.exists():
            return None
        receipt = read(path)
        expected = 'EXTERNAL_RECONSTRUCTIONS_COMPLETE' if stage == 'reconstruct' else 'EXTERNAL_EVALUATION_COMPLETE'
        require(receipt.get('status') == expected and receipt.get('synthetic') is False
                and receipt.get('sources') == 100 and receipt.get('rows') == 3600
                and receipt.get('physical_frames') == 1800 and receipt.get('sampler_step_limit') is None
                and receipt.get('selection_uses_development') is False,
                'Invalid full external evaluation completion')
        if stage == 'score':
            selected = read(Path(config['training']['output'])/'selected_swin.json')
            require(receipt.get('reconstruction_completion_sha256') == sha(out/'reconstruction_completion.json')
                    and receipt.get('selected_checkpoint_sha256') == sha(selected['checkpoint'])
                    and receipt.get('holdout_access') is False
                    and receipt.get('paired_bootstrap_unit') == 'source_mean_after_three_noise_repeats',
                    'Final metric population, selected checkpoint or pairing differs')
        verify(receipt.get('bindings')); verify(receipt.get('outputs'))
        require(receipt['bindings'].get(config['evaluation']['config']) == sha(config['evaluation']['config'])
                and receipt['bindings'].get(config['evaluation']['script']) == sha(config['evaluation']['script']),
                'Evaluation completion lacks registered source/config identity')
    return dict(path=str(path), sha256=sha(path), status=receipt['status'])


def ready_to_signal(config, stage, launch):
    """Qualification has no handler; wait for it. Never kill during model loading."""
    if stage in ('qualification', 'hifi_preflight', 'hifi_qualification'):
        return False
    if stage == 'training':
        # The trainer installs handlers before its first new checkpoint/status.
        out = Path(config['training']['output'])
        for path in (out/'latest.json', out/'status.json'):
            if path.exists() and path.stat().st_mtime_ns > launch['started_ns']:
                value = read(path)
                if path.name == 'latest.json' or value.get('status') == 'TRAINING':
                    return True
        return False
    path = Path(config['evaluation']['output'])/(stage+'_status.json')
    if not path.exists():
        return False
    value = read(path)
    return (value.get('pid') == launch['pid'] and value.get('launch_id') == launch['launch_id']
            and value.get('safe_pause_handler_installed') is True)


def paused_receipt(config, stage, launch):
    _, _, out = stage_spec(config, stage, launch['launch_id'])
    simple = stage in ('qualification', 'training', 'hifi_preflight', 'hifi_qualification')
    path = out/('status.json' if simple else stage+'_status.json')
    if not path.exists() or path.stat().st_mtime_ns <= launch['started_ns']:
        return False
    receipt = read(path)
    if receipt.get('status') != 'PAUSED':
        return False
    return simple or receipt.get('launch_id') == launch['launch_id']


def signal_owned(launch, reader=process, sender=None):
    descriptor = None
    if sender is None:
        require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'),
                'Linux pidfd is required for a race-free owned-child stop')
        try:
            descriptor = os.pidfd_open(launch['pid'])
        except ProcessLookupError:
            return False
    try:
        if not live(launch, reader):
            return False
        current = reader(launch['pid'])
        require(current and current.get('command') == launch['command'], 'Owned child command changed')
        if sender is None:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        else:
            sender(launch['pid'], signal.SIGTERM)
        return True
    finally:
        if descriptor is not None:
            os.close(descriptor)


class Controller:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve(); self.config = validate_config(read(self.config_path))
        self.config_sha = sha(self.config_path); self.out = Path(self.config['output'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop = False; self.child = None; self.poll = self.config.get('poll_seconds', 15)

    def status(self, status, **extra):
        write(self.out/'status.json', dict(status=status, time=time.time(), config_sha256=self.config_sha, **extra))

    def verify(self):
        require(sha(self.config_path) == self.config_sha, 'Controller config changed while running')
        verify(self.config['bindings'])

    def wait_gate(self):
        while True:
            if self.stop:
                raise Paused('Stop requested before new work started')
            try:
                gate = old_gate(self.config)
                owners = gpu_pids()
                if owners:
                    raise Waiting('Waiting for GPU0 owners to exit: '+str(owners))
                register(self.out/'historical_gate.json', gate)
                return gate
            except Waiting as error:
                self.status('WAITING_FOR_HISTORICAL_PUBLICATION_OR_GPU', reason=str(error))
                time.sleep(self.poll)

    def existing_child(self, stage):
        path = self.out/(stage+'_current_launch.json')
        if not path.exists():
            return None
        record = read(path)
        require(record.get('config_sha256') == self.config_sha and record.get('stage') == stage,
                'Existing child belongs to another registered configuration')
        command, environment, _ = stage_spec(self.config, stage, record['launch_id'])
        require(record.get('command') == command and record.get('environment') == environment
                and record.get('source_bindings') == self.config['bindings'], 'Existing child launch identity differs')
        require((self.out/'launches'/(record['launch_id']+'.json')).is_file()
                and read(self.out/'launches'/(record['launch_id']+'.json')) == record,
                'Current launch lacks its immutable record')
        return record

    def child_done(self, stage, launch, code):
        reject_scientific_failure(self.config, stage)
        result_path = self.out/'exits'/(launch['launch_id']+'.json')
        if code == 75 or (code is None and paused_receipt(self.config, stage, launch)):
            require(paused_receipt(self.config, stage, launch), 'Exit75 lacks a fresh scientific pause receipt')
            result = dict(status='PAUSED', stage=stage, launch_id=launch['launch_id'], returncode=code,
                          config_sha256=self.config_sha)
            register(result_path, result)
            raise Paused('Scientific stage paused safely: '+stage)
        require(code in (None, 0), 'Scientific stage exited unsuccessfully: '+stage+' code='+str(code))
        proof = completion(self.config, stage)
        require(proof is not None, 'Scientific stage exited without verified completion: '+stage)
        register(result_path, dict(status='COMPLETE', stage=stage, launch_id=launch['launch_id'],
                                  returncode=code, config_sha256=self.config_sha, completion=proof))
        return proof

    def stage(self, stage):
        self.verify(); reject_scientific_failure(self.config, stage)
        launch = self.existing_child(stage)
        child_live = bool(launch and live(launch))
        if launch and not child_live:
            exit_path = self.out/'exits'/(launch['launch_id']+'.json')
            if exit_path.exists():
                saved = read(exit_path)
                require(saved.get('config_sha256') == self.config_sha and saved.get('launch_id') == launch['launch_id']
                        and saved.get('status') in ('COMPLETE', 'PAUSED'), 'Previous child exit receipt differs')
                if saved['status'] == 'COMPLETE':
                    proof = completion(self.config, stage)
                    require(proof == saved['completion'], 'Completed stage artifacts changed')
                    return proof
                # A deliberately resumed controller may restart only a safe pause.
                launch = None
            else:
                return self.child_done(stage, launch, None)
        if not child_live:
            proof = completion(self.config, stage)
            if proof:
                return proof
            self.wait_gate(); self.verify()
            if self.stop:
                raise Paused('Stop requested before '+stage)
            token = stage+'_'+uuid.uuid4().hex
            command, environment, _ = stage_spec(self.config, stage, token)
            env = os.environ.copy(); env.pop('HISTORICAL_TORCH_METADATA_MANIFEST', None)
            env.pop('LD_PRELOAD', None); env.update(environment)
            logpath = self.out/'logs'/(token+'.log'); logpath.parent.mkdir(parents=True, exist_ok=True)
            started_ns = time.time_ns()
            with logpath.open('ab', buffering=0) as logfile:
                self.child = subprocess.Popen(command, cwd=self.config['root'], env=env,
                    stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
            who = process(self.child.pid)
            require(who is not None, 'Could not obtain launched child PID/start_ticks')
            launch = dict(pid=self.child.pid, start_ticks=who['start_ticks'], command=command,
                          environment=environment, stage=stage, launch_id=token, started_ns=started_ns,
                          config_sha256=self.config_sha, source_bindings=self.config['bindings'], log=str(logpath))
            register(self.out/'launches'/(token+'.json'), launch)
            write(self.out/(stage+'_current_launch.json'), launch)
        else:
            require(process(launch['pid']).get('command') == launch['command'], 'Adopted child command differs')
            self.status('ADOPTING_REGISTERED_CHILD', stage=stage, child=launch)
        signalled = False
        while live(launch):
            if self.stop and not signalled and ready_to_signal(self.config, stage, launch):
                signalled = signal_owned(launch)
            self.status('PAUSING' if self.stop else 'RUNNING', stage=stage, child=launch,
                        safe_stop_signal_sent=signalled,
                        stop_waits_for_qualification_or_handler=self.stop and not signalled)
            time.sleep(self.poll)
        code = self.child.wait() if self.child is not None else None
        self.child = None
        self.verify()
        result = self.child_done(stage, launch, code)
        if self.stop:
            raise Paused('Stop requested; completed current safe stage: '+stage)
        return result

    def run_stages(self, stages, active):
        """A stop during adoption must still reach and safely pause the live child."""
        proofs = {}
        active_index = stages.index(active[0]) if active else -1
        for index, stage in enumerate(stages):
            if active and index < active_index:
                proof = completion(self.config, stage)
                require(proof is not None, 'Active child lacks a completed prerequisite: '+stage)
                proofs[stage] = proof
                continue
            if self.stop and index != active_index:
                raise Paused('Stop requested between scientific stages')
            proofs[stage] = self.stage(stage)
        return proofs

    def run(self):
        import fcntl
        lock = (self.out/'controller.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            require(not (self.out/'failure.json').exists(), 'Controller failure requires review before restart')
            register(self.out/'registration.json', dict(version=VERSION, config_sha256=self.config_sha,
                configuration=self.config, source_bindings=self.config['bindings'],
                original_sources_changed=False, git_operations=False, automatic_failure_retry=False))
            own = process(os.getpid()); require(own is not None, 'Missing controller process identity')
            own_record = dict(pid=own['pid'], start_ticks=own['start_ticks'], config_sha256=self.config_sha,
                              command=[sys.executable, *sys.argv], source_bindings=self.config['bindings'])
            register(self.out/'controller_launches'/(str(own['pid'])+'_'+str(own['start_ticks'])+'.json'), own_record)
            write(self.out/'controller_launch.json', own_record)
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
            signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))
            # Resume any exact live child before GPU-idle checking; it owns GPU0.
            stages = (['qualification'] if self.config['mode'] == 'qualification_only'
                      else ['qualification', 'hifi_preflight', 'training', 'hifi_qualification'])
            if self.config.get('evaluation') and self.config['mode'] != 'qualification_only':
                stages += ['reconstruct', 'score']
            active = []
            for stage in stages:
                previous = self.existing_child(stage)
                if previous and live(previous):
                    active.append(stage)
            require(len(active) <= 1, 'Multiple previously launched scientific children are alive')
            if not active:
                self.wait_gate()
            else:
                register(self.out/'historical_gate.json', old_gate(self.config))
            proofs = self.run_stages(stages, active)
            if self.config['mode'] == 'qualification_only':
                terminal = 'QUALIFICATION_ONLY_COMPLETE'
            elif not self.config.get('evaluation'):
                terminal = 'TRAINING_AND_HIFI_QUALIFICATION_COMPLETE_AWAITING_EVALUATION_REGISTRATION'
            else:
                terminal = 'EXTERNAL_TRAINING_AND_EVALUATION_COMPLETE'
            done = dict(status=terminal, config_sha256=self.config_sha, stages=proofs,
                        publication_performed=False, stop=True)
            register(self.out/'completion.json', done); self.status(terminal, stages=proofs)
            return 0
        except Paused as error:
            self.status('PAUSED', reason=str(error), safe_to_resume_same_configuration=True)
            return 75
        except BaseException as error:
            # Never kill a surviving scientific worker on a controller error.
            # Its exact launch remains visible, and no automatic retry occurs.
            failure = dict(status='FAILED_REQUIRES_REVIEW', error=repr(error), time=time.time(),
                           config_sha256=self.config_sha, no_automatic_restart=True,
                           surviving_child_pid=self.child.pid if self.child is not None else None)
            if not (self.out/'failure.json').exists():
                write(self.out/'failure.json', failure)
            self.status('FAILED_REQUIRES_REVIEW', error=repr(error)); raise
        finally:
            lock.close()


def detach(config_path):
    """Launch the same checked controller in a new session, without a shell."""
    config_path = Path(config_path).resolve(); config = validate_config(read(config_path))
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (out/'failure.json').exists(), 'Failed controller requires review, not relaunch')
        for name in ('dispatch.json', 'controller_launch.json'):
            path = out/name
            if path.exists():
                previous = read(path)
                require(previous.get('config_sha256') == sha(config_path), 'Prior controller configuration differs')
                if live(previous):
                    print(json.dumps(dict(status='ALREADY_RUNNING', **previous)), flush=True)
                    return 0
        token = 'controller_'+uuid.uuid4().hex
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(config_path)]
        logpath = out/'logs'/(token+'.log'); logpath.parent.mkdir(parents=True, exist_ok=True)
        with logpath.open('ab', buffering=0) as stream:
            child = subprocess.Popen(command, cwd=config['root'], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        who = process(child.pid); require(who is not None, 'Missing detached controller identity')
        record = dict(pid=child.pid, start_ticks=who['start_ticks'], command=command,
                      config_sha256=sha(config_path), source_bindings=config['bindings'], log=str(logpath))
        register(out/'dispatches'/(token+'.json'), record); write(out/'dispatch.json', record)
        print(json.dumps(dict(status='DETACHED_CONTROLLER_LAUNCHED', **record)), flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    return detach(args.config) if args.detach else Controller(args.config).run()


if __name__ == '__main__':
    raise SystemExit(main())
