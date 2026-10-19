"""Root-owned exact-path publication. No science or statistical recomputation."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse
import ast
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
BASE = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
PY = str(RT/'envs/frozen_visual_py310/bin/python')
A = BASE/'manuscript_completion_20261019/final_publication_v1'
P = R/'results/manuscript_completion_20261019_v1/final_publication_preparation_v1'
S = R/'experiments/manuscript-completion-20261019/final_publication_v1'
WINDOWS_WORKSPACE = PureWindowsPath('C:/Users/11946/Documents/ChatGPT/comm')
SHARED_WAIT_SOURCE = R/'experiments/arpc-n2048-adaptive-20261019/shared.py'
SHARED_WAIT_SHA = 'beab2a0b405b2d04cd5a0dcf4019743204e5daef5965b63ddaac506ac615f67e'
NATIVE_OBSERVER_SOURCE = R/'experiments/arpc-n2048-adaptive-20261019/b2b3_analysis_v1/native.py'
NATIVE_OBSERVER_SHA = 'd02813fe2cdb3be74129c9a7cd7afd0474307278daa75f5aab644ad38088a889'
DOCS = ('README.md', 'RESEARCH_STATUS.md', 'PROGRESS.md', 'EXPERIMENTS.md')
HELPERS = ('publication.py', 'invoke.py', 'push.py', 'sync.py', 'README.md')
SOURCE_ROOTS = (
    'experiments/n1024-consistency-20261019',
    'experiments/manuscript-completion-20261019/final_publication_v1',
    *('experiments/arpc-n2048-adaptive-20261019/'+n for n in (
        'b2b3_revision_v1', 'b2b3_analysis_v1', 'b2b3_analysis_qualification_continuation_v1',
        'b2b3_analysis_qualification_continuation_v2', 'b2b3_metadata_compatibility_v1',
        'b2b3_report_v1', 'b3_metrics_v1')))
RESULT_ROOTS = (
    'results/n1024_consistency_20261019_v1',
    'results/manuscript_completion_20261019_v1',
    'paper/manuscripts/nestcom_entropy_v6_20261019',
    *('results/arpc_n2048_adaptive_20261019_v1/'+n for n in (
        'b2b3_revision_v1', 'b2b3_revision_preparation_v1', 'b2b3_analysis_v1',
        'b2b3_analysis_preparation_v1', 'b2b3_analysis_metadata_failure_v1',
        'b2b3_metadata_compatibility_v1', 'b2b3_metadata_compatibility_preparation_v1',
        'b2b3_report_v1', 'b2b3_report_preparation_v1', 'b3_metrics_v1',
        'b3_metrics_preparation_v1', 'b2b3_execution_receipts_v1')))
REQUIRED_CLOSURES = {
    'N1024_audit', 'N1024_repair_delivery', 'N1024_report',
    'B2_transport', 'B2_native', 'B2_DINO', 'B2_selection',
    'B3_transport', 'B3_native', 'B3_metrics', 'B3_statistics', 'B3_report',
    'manuscript_compile', 'manuscript_delivery'}
REQUIRED_REVIEWS = {'N1024_utilization', 'N1024_quality', 'B3_figures', 'manuscript_PDF'}
REQUIRED_PACKAGES = {'N1024_audit','N1024_repair','N1024_report','B2B3_transport',
    'B2B3_analysis','analysis_continuation_v1','analysis_continuation_v2',
    'metadata_compatibility','B3_metrics','B2B3_report','manuscript','publisher'}
FORBIDDEN_PARTS = {'publish','outputs','checkpoints','datasets','data','weights','vendor',
    'third_party','.venv','venv','__pycache__','_migration_backup'}
FORBIDDEN_SUFFIXES = {'.pt','.pth','.ckpt','.model','.safetensors','.npy','.npz','.pkl',
    '.pickle','.so','.o','.whl','.pem','.key','.p12','.zip','.tgz','.gz','.sqlite','.db',
    '.log','.lock','.ttf','.otf','.woff','.woff2'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=R, text=True).strip()


def safe_file(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts, 'Absolute normalized path required')
    require(any(p.is_relative_to(root) for root in (R,BASE)),
            'Evidence must be beneath the exact repository or runtime evidence root')
    for q in (p, *p.parents):
        require(not q.is_symlink(), 'Symlink in evidence or publication path')
    require(p.resolve() == p and p.is_file(), 'Missing/nonregular evidence')
    require(stat.S_ISREG(p.stat().st_mode), 'Evidence is not a regular file')
    return p


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    p = safe_file(path); b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=sha(b))


def checked(d, limit=10000000):
    require(set(d) == {'path','bytes','sha256'} and type(d['bytes']) is int
            and 0 <= d['bytes'] <= limit and re.fullmatch('[0-9a-f]{64}',d['sha256']),
            'Invalid light evidence descriptor')
    p = safe_file(d['path']); before = p.stat(); b = p.read_bytes(); after = p.stat()
    require((before.st_ino,before.st_size,before.st_mtime_ns) ==
            (after.st_ino,after.st_size,after.st_mtime_ns), 'Evidence changed while read')
    require(len(b) == d['bytes'] and sha(b) == d['sha256'], 'Evidence SHA/size mismatch: '+str(p))
    return b


def load(d):
    return json.loads(checked(d))


def put(path, value):
    p = Path(path)
    fd = os.open(p, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd,'w',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def pointer(obj, path):
    require(path == '' or path.startswith('/'), 'Invalid JSON pointer')
    if path:
        for k in path.split('/')[1:]:
            k = k.replace('~1','/').replace('~0','~')
            obj = obj[int(k)] if isinstance(obj,list) else obj[k]
    return obj


def assertions(record, checks):
    require(isinstance(checks,list) and checks, 'Empty evidence assertions')
    for item in checks:
        actual = pointer(record,item['pointer']); expected = item['equals']
        require(type(actual) == type(expected) and actual == expected,
                'Evidence assertion differs: '+item['pointer'])


def artifact_binding(record, binding):
    member = pointer(record,binding['pointer'])
    require(isinstance(member,dict), 'Artifact field is not a recorded descriptor')
    d = binding['descriptor']
    require(member['bytes'] == d['bytes'] and member['sha256'] == d['sha256'],
            'Manifest/review artifact SHA or bytes differs')
    raw = member['path']; require(isinstance(raw,str) and raw, 'Missing artifact path')
    windows = PureWindowsPath(raw)
    if windows.drive or '\\' in raw:
        require(binding.get('path_mapping') == 'WINDOWS_WORKSPACE_TO_REPOSITORY_V1'
                and windows.is_absolute() and '..' not in windows.parts
                and windows.is_relative_to(WINDOWS_WORKSPACE), 'Unapproved Windows artifact mapping')
        relative = windows.relative_to(WINDOWS_WORKSPACE)
        require(relative.parts and not any(':' in part for part in relative.parts), 'Invalid Windows artifact path')
        recorded = R.joinpath(*relative.parts)
        require('relative_base' not in binding, 'Windows mapping cannot also use a relative base')
    else:
        require('path_mapping' not in binding, 'Path mapping is only for actual Windows workspace records')
        recorded = Path(raw)
    if not recorded.is_absolute():
        base = Path(binding['relative_base'])
        require(base.is_absolute() and '..' not in base.parts and '..' not in recorded.parts
                and any(base.is_relative_to(root) for root in (R,BASE)), 'Invalid relative artifact base')
        recorded = base/recorded
    require(str(recorded) == d['path'], 'Manifest/review artifact path differs')
    checked(d)


def identity(pid):
    p = Path('/proc')/str(pid)/'stat'
    text = p.read_text(); rest = text[text.rindex(') ')+2:].split()
    return dict(pid=pid,start_ticks=int(rest[19]),host=os.uname().nodename)


def gone(d):
    require(d['host'] == os.uname().nodename, 'Cross-host PID requires execution-host proof')
    p = Path('/proc')/str(d['pid'])/'stat'
    return not p.exists() or identity(d['pid'])['start_ticks'] != d['start_ticks']


def process_fields(record, spec, host, hostless_proof=False):
    row = pointer(record,spec['pointer'])
    result = dict(pid=row[spec.get('pid_key','pid')],
                  start_ticks=row[spec.get('start_ticks_key','start_ticks')],host=host)
    require(type(result['pid']) is int and result['pid'] > 0
            and type(result['start_ticks']) is int and result['start_ticks'] > 0,
            'Invalid PID/start generation')
    host_keys = {key for key in ('host','hostname') if key in row}
    if spec.get('host_key'): host_keys.add(spec['host_key'])
    if host_keys:
        require(all(row[key] == host for key in host_keys), 'Recorded process host differs')
    else:
        require(hostless_proof, 'Hostless process needs an exact linked host-provenance record')
    return result


def host_provenance(spec, launch, host):
    if 'host_provenance' not in spec: return False
    proof = spec['host_provenance']
    if proof.get('schema') == 'EXACT_HISTORICAL_FIXED_SSH_V1':
        return exact_historical_fixed_ssh(spec,launch,host,proof)
    require(proof['record'] in spec.get('identity_records',[])
            and proof['anchor_record'] == spec['launch']
            and proof['anchor_pointer'] == '/parent_process', 'Host provenance is not linked to the launch controller')
    anchored = pointer(launch,proof['anchor_pointer'])
    witnessed = process_fields(load(proof['record']),dict(pointer=proof['pointer'],
        **({'host_key':proof['host_key']} if 'host_key' in proof else {})),host)
    require(type(anchored['pid']) is int and type(anchored['start_ticks']) is int
            and witnessed['pid'] == anchored['pid'] and witnessed['start_ticks'] == anchored['start_ticks'],
            'Hostless launch controller differs from host-aware provenance generation')
    for key in ('host','hostname'):
        if key in anchored: require(anchored[key] == host, 'Host provenance contradicts recorded host')
    return True


def exact_historical_fixed_ssh(spec, launch, host, proof):
    """One preserved hostless audit, accepted as reconstructed SSH provenance."""
    root = R/'results/n1024_consistency_20261019_v1/audit_host_provenance_v1'
    control = BASE/'n1024_consistency_20261019_v1/audit_control_v2'
    original = BASE/'n1024_consistency_20261019_v1/audit_v2'
    fixed_host = 'aibox-rfbd2ce574fd-5df9766c45-vz6d9'
    child = dict(pid=3316552,start_ticks=208506583)
    parent = dict(pid=3316543,start_ticks=208506575)
    wait = dict(path=str(control/'actual_wait.json'),bytes=559,
        sha256='db9e4d805626f45804d8c62a15651c84ab95a72922adb0d4d6ffa47115c6b300')
    start = dict(path=str(control/'launch.json'),bytes=510,
        sha256='3ff48698558cbdb7f57e96d017fea624d7a8bfaf03628b17b76ed3e712761752')
    require(spec['id'] == 'N1024_audit' and spec['receipt'] == wait and spec['launch'] == start
            and host == fixed_host == os.uname().nodename
            and spec.get('object_pointer','') == '' and spec.get('argv_pointer','/argv') == '/argv'
            and spec.get('receipt_schema','EXPLICIT_ACTUAL_WAIT_V1') == 'EXPLICIT_ACTUAL_WAIT_V1',
            'Fixed-SSH route is limited to the exact original main-host N1024 audit')
    require(proof['record']['path'] == str(root/'root_acceptance.json')
            and proof['record'] in spec.get('identity_records',[]), 'Unbound historical route acceptance')
    accepted = load(proof['record'])
    reconstruction = dict(path=str(root/'audit_host_provenance_reconstruction_v1.json'),bytes=11612,
        sha256='8d32ca69f807b6d8540ea4e4ea94012f90bbef2854fdfcf0f37ecbd6cde45fa5')
    require(accepted['schema'] == 'N1024_FIXED_SSH_HOST_MAPPING_ACCEPTANCE_V1'
            and accepted['status'] == 'ROOT_ACCEPTED_HISTORICAL_RECONSTRUCTION'
            and accepted['execution_host'] == host and accepted['reconstruction'] == reconstruction
            and accepted['ssh_alias'] == 'DevMachineModel-H100-4C-P1660-01'
            and accepted['actual_wait'] == wait and accepted['launch'] == start
            and accepted['process'] == child and accepted['parent_process'] == parent
            and accepted['original_inline_host_present'] is False
            and accepted['host_aware_same_generation_anchor_found'] is False
            and accepted['current_alias_and_proc_are_historical_host_records'] is False
            and isinstance(accepted['basis'],str) and accepted['basis'].strip()
            and isinstance(accepted['reviewer'],str) and accepted['reviewer'].strip(),
            'Historical mapping must preserve its explicit limits and exact generations')
    rec = load(reconstruction)
    require(rec['schema'] == 'N1024_HISTORICAL_FIXED_SSH_PROVENANCE_RECONSTRUCTION_V1'
            and rec['historical_process'] == child and rec['historical_controller'] == parent
            and rec['historical_actual_wait'] == load(wait)
            and all(value is True for value in rec['checks'].values())
            and rec['historical_route']['original_inline_host_present'] is False
            and rec['historical_route']['host_aware_same_generation_anchor_found'] is False,
            'Pinned historical reconstruction differs')
    fixed = {
      'launch_v2.py':(2322,'b1fc9958c407c43c829aadd9c35768a2c3daade30587071557eff6ab7fb11c84'),
      'launch_remote_audit_v2.py':(28005,'8fa26cd00a9adc9293b4445b66f5545be24b70633f492a8832bd140690721e5b'),
      'audit_v2_launch_stdout.txt':(1971,'010095a62c7dc73a70dcd3d3600e8759df352440c5adfbcc487c454a048dcc9a'),
      'audit_v2_launch_stderr.txt':(0,'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
      'inner_launch.json':(357,'487347bd45c1170cf9aa1caedf555a3cd5b14b5c76ed62836e6238dea6d8413a'),
      'control_stdout.txt':(1882,'023f2277de44443c3b3d5be6fc3e5189d31d75d145bbd2673197035ef310382d'),
      'control_stderr.txt':(0,'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')}
    require(set(accepted['copied_evidence']) == set(fixed), 'Fixed-SSH evidence set differs')
    content = {}
    for name,(size,digest) in fixed.items():
        expected = dict(path=str(root/name),bytes=size,sha256=digest)
        require(accepted['copied_evidence'][name] == expected, 'Fixed-SSH evidence descriptor differs')
        content[name] = checked(expected)
    audit = dict(path=str(R/'experiments/n1024-consistency-20261019/audit_v2.py'),bytes=19672,
        sha256='7880c430e05091aec478761678108b018d0aa545e16f983e1f5a1fd40c726d91')
    require(accepted['audit_source'] == audit, 'Original audit source binding differs'); checked(audit)
    require(checked(dict(path=str(original/'launch.json'),bytes=357,sha256=fixed['inner_launch.json'][1]))
            == content['inner_launch.json'], 'Original inner launch differs from retained copy')
    for src,dst in (('stdout.txt','control_stdout.txt'),('stderr.txt','control_stderr.txt')):
        require(checked(dict(path=str(control/src),bytes=fixed[dst][0],sha256=fixed[dst][1])) == content[dst],
                'Original control log differs from retained copy')
    inner = json.loads(content['inner_launch.json']); actual = load(wait)
    require(launch['process'] == inner['process'] == actual['process'] == child
            and launch['parent_process'] == inner['parent_process'] == actual['parent_process'] == parent
            and launch['source_sha256'] == audit['sha256']
            and launch['argv'] == actual['argv'] == [str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',audit['path']]
            and inner['argv'] == [audit['path']], 'Original exact SSH process/command chain differs')
    require(content['audit_v2_launch_stdout.txt'].decode().splitlines()[0]
            == 'ACTUAL_WAIT 0 121.4116426508408', 'Original SSH actual-exit output differs')
    return True


def shared_wait_contract(spec, obj, launch, parent):
    producer = spec['producer']; source = checked(producer)
    require(producer['path'] == str(SHARED_WAIT_SOURCE) and producer['sha256'] == SHARED_WAIT_SHA
            and subprocess.check_output(['git','show',parent+':'+SHARED_WAIT_SOURCE.relative_to(R).as_posix()],cwd=R) == source,
            'Only the exact unchanged original shared.wait_child producer may omit actual_wait')
    tree = ast.parse(source); functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'wait_child']
    require(len(functions) == 1, 'Frozen actual-wait producer is absent')
    calls = [n for n in ast.walk(functions[0]) if isinstance(n,ast.Call)]
    require(any(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
                and n.func.value.id == 'p' and n.func.attr == 'wait'
                and any(k.arg == 'timeout' and isinstance(k.value,ast.Name) and k.value.id == 'timeout' for k in n.keywords)
                for n in calls), 'Frozen producer is not an actual bounded child.wait contract')
    require(spec.get('object_pointer','') == '' and spec.get('argv_pointer','/argv') == '/argv'
            and spec.get('returncode_key','returncode') == 'returncode'
            and 'actual_wait' not in obj
            and set(obj) == {'process','returncode','seconds','timed_out','terminated','killed','metadata_error'}
            and obj['process'] == launch['process'], 'Frozen shared-wait receipt/launch schema differs')
    require(type(obj['seconds']) in (int,float) and obj['seconds'] >= 0
            and all(obj[key] is False for key in ('timed_out','terminated','killed'))
            and obj['metadata_error'] is None, 'Frozen shared wait interrupted or invalid')


def verify_wait(spec, parent):
    wait_data = load(spec['receipt']); obj = pointer(wait_data,spec.get('object_pointer',''))
    launch = load(spec['launch']); mode = spec.get('receipt_schema','EXPLICIT_ACTUAL_WAIT_V1')
    require(mode in ('EXPLICIT_ACTUAL_WAIT_V1','FROZEN_SHARED_WAIT_CHILD_V1'), 'Unknown wait schema')
    if mode == 'FROZEN_SHARED_WAIT_CHILD_V1': shared_wait_contract(spec,obj,launch,parent)
    else: require(obj[spec.get('actual_wait_key','actual_wait')] is True, 'No actual child.wait evidence')
    rc = obj[spec.get('returncode_key','returncode')]
    require(type(rc) is int and rc == 0, 'Current stage actual wait is not zero')
    for key in ('timed_out','terminated','killed','timeout'):
        if key in obj:
            require(obj[key] is False, 'Actual wait interrupted')
    if 'metadata_error' in obj:
        require(obj['metadata_error'] is None, 'Actual wait metadata error')
    require(pointer(launch,spec.get('argv_pointer','/argv')) == spec['argv']
            and isinstance(spec['argv'],list) and spec['argv'], 'Actual command differs')
    host = spec['execution_host']; require(isinstance(host,str) and host, 'Missing execution host')
    hostless_proof = host_provenance(spec,launch,host)
    generations = []
    require(len(spec['processes']) >= 2, 'Both worker and controller generations are required')
    for process in spec['processes']:
        require(process['record'] in (spec['receipt'],spec['launch']) or
                process['record'] in spec.get('identity_records',[]), 'Unbound process identity record')
        generations.append(process_fields(load(process['record']),process,host,hostless_proof))
    require(len({(x['pid'],x['start_ticks'],x['host']) for x in generations}) == len(generations),
            'Repeated process generation cannot stand for both child and controller')
    actual_child = process_fields(obj,spec.get('wait_child_identity',{'pointer':'/process'}),host,hostless_proof)
    launch_child = process_fields(launch,spec.get('launch_child_identity',{'pointer':'/process'}),host,hostless_proof)
    controller = process_fields(launch,spec.get('launch_parent_identity',{'pointer':'/parent_process'}),host,hostless_proof)
    require(actual_child == launch_child and actual_child != controller
            and actual_child in generations and controller in generations,
            'Actual wait/launch child or bound controller generation differs')
    if mode == 'FROZEN_SHARED_WAIT_CHILD_V1':
        expected = [process_fields(launch,{'pointer':'/'+key},host) for key in ('process','parent_process')]
        require(generations == expected, 'Frozen shared wait must bind its actual child and controller')
    if host == os.uname().nodename:
        require(all(gone(x) for x in generations), 'Current stage process generation is still live')
    else:
        obs = load(spec['host_observation'])
        observation_mode = spec.get('host_observation_schema','FINAL_PUBLICATION_PROCESS_OBSERVATION_V1')
        if observation_mode == 'FROZEN_NATIVE_CLOSURE_OBSERVATION_V1':
            source = spec['observer_source']; checked(source)
            require(source['path'] == str(NATIVE_OBSERVER_SOURCE) and source['sha256'] == NATIVE_OBSERVER_SHA,
                    'Unknown native execution-host observer source')
            observed = [process_fields(obs,{'pointer':'/'+key},host) for key in ('child','parent')]
            require(obs['status'] == 'CLOSED_ON_EXECUTION_HOST' and obs['host'] == host
                    and obs['actual_wait'] == spec['receipt'] and obs['launch'] == spec['launch']
                    and obs['child_gone'] is True and obs['parent_gone'] is True and observed == generations
                    and obs['not_before_unix'] == launch['started_unix']+obj['seconds']
                    and obs['observed_unix'] >= obs['not_before_unix'], 'Frozen native host closure differs')
        else:
            require(observation_mode == 'FINAL_PUBLICATION_PROCESS_OBSERVATION_V1'
                and obs['schema'] == 'FINAL_PUBLICATION_PROCESS_OBSERVATION_V1'
                and obs['status'] == 'CLOSED_ON_EXECUTION_HOST'
                and obs['observer_host'] == host and obs['execution_host'] == host
                and obs['wait'] == spec['receipt'] and obs['launch'] == spec['launch']
                and obs['argv'] == spec['argv'] and obs['processes'] == generations
                and obs['all_process_generations_gone'] is True,
                'Cross-host process proof differs; OTHER_HOST is not gone')
            checked(obs['observer_source'])
        require(type(obs['observed_unix']) in (int,float) and
                obs['observed_unix'] >= safe_file(spec['receipt']['path']).stat().st_mtime,
                'Host observation predates actual wait')
    return {'id':spec['id'],'receipt':spec['receipt'],'launch':spec['launch'],
            'processes':generations,'host_observation':spec.get('host_observation')}


def allowed(relative):
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == relative,
            'Bad repository-relative path')
    require(not (set(p.parts)&FORBIDDEN_PARTS) and p.suffix.lower() not in FORBIDDEN_SUFFIXES
            and not p.name.lower().startswith(('.env','id_rsa','id_ed25519')), 'Excluded file type')
    return relative in DOCS or any(relative.startswith(root+'/') for root in SOURCE_ROOTS+RESULT_ROOTS)


def file_map(desc):
    rows = desc['files']; out = {}
    require(isinstance(rows,list) and rows, 'No exact files')
    for row in rows:
        require(set(row) == {'path','bytes','sha256'} and allowed(row['path']), 'Outside positive publication scope')
        require(row['path'] not in out, 'Duplicate exact path')
        checked(dict(row,path=str(R/row['path']))); out[row['path']] = row
    require(set(DOCS) <= set(out), 'Four current root documents required')
    return out


def evidence(desc):
    require(set(desc['closures']) == REQUIRED_CLOSURES, 'Final closure role set incomplete')
    waits = desc['actual_waits']; require(len({x['id'] for x in waits}) == len(waits), 'Duplicate stage waits')
    wait_ids = {x['id'] for x in waits}; proof = [verify_wait(x,desc['parent']) for x in waits]
    for role, row in desc['closures'].items():
        assertions(load(row['record']),row['assertions'])
        require(set(row['wait_ids']) <= wait_ids and row['wait_ids'], 'Closure not bound to actual exit: '+role)
        for member in row['light_members']: checked(member)
        for manifest in row['manifests']:
            actual = load(manifest['record'])
            for binding in manifest['member_bindings']:
                artifact_binding(actual,binding)
    require(set(desc['visual_reviews']) == REQUIRED_REVIEWS, 'Required final visual review missing')
    for role, review in desc['visual_reviews'].items():
        actual = load(review['record']); assertions(actual,review['assertions'])
        status=pointer(actual,review.get('status_pointer','/status'))
        require(isinstance(status,str) and status.startswith(('PASS','COMPLETE'))
                and not any(t in status.upper() for t in ('PENDING','INTERMEDIATE','NOT_RUN')),
                'Actual review is not final PASS/COMPLETE')
        require(review['opened_bindings'], 'No actual opened artifact evidence: '+role)
        for binding in review['opened_bindings']:
            require(pointer(actual,binding['opened_pointer']) is True, 'Artifact was not actually opened')
            artifact_binding(actual,binding)
        require(review['final_not_intermediate'] is True, 'Intermediate review cannot close final delivery')
    require(desc['history']['qualification_attempts'] and desc['history']['preserved_failures'],
            'Preserve multi-attempt qualification and failure lineage')
    for item in desc['history']['qualification_attempts']+desc['history']['preserved_failures']:
        require(item['description'] and item['records'], 'Historical evidence lacks explanation')
        for d in item['records']: checked(d)
    approved = {d['path']:d for item in desc['history']['preserved_failures'] for d in item['records']}
    for path in desc['failure_locations']:
        p = Path(path)
        if p.exists():
            require(path in approved and pin(p) == approved[path], 'Unknown/changed failure preserved; stop')
    require(desc['failure_locations'], 'Explicit stage failure locations missing')
    return proof


def descriptor(d, check_parent=True):
    require(os.name == 'posix' and Path.cwd() == R and git('rev-parse','--show-toplevel') == str(R),
            'Only authoritative Linux Git root')
    require(Path(d['path']) == P/'authorization.json', 'Wrong final descriptor path')
    v = load(d)
    require(v['schema'] == 'FINAL_PUBLICATION_AUTHORIZATION_V1'
            and v['status'] == 'ROOT_REVIEWED_COMPLETE' and v['repository'] == str(R)
            and v['branch'] == 'main' and v['publication'] == str(A)
            and re.fullmatch('[0-9a-f]{40}',v['parent']), 'Descriptor is not actual final authorization')
    require(v['scientific_calls'] == 0 and v['large_backup_deferred'] is True,
            'Publication is not scientific execution or backup')
    require(isinstance(v['commit_message'],str) and 1<=len(v['commit_message'])<=1000,
            'Provide the factual reviewed commit description')
    require(git('branch','--show-current') == 'main', 'Wrong branch')
    if check_parent: require(git('rev-parse','HEAD') == v['parent'], 'Current parent differs; do not rebase/reset')
    require(git('remote','get-url','origin') in (
        'https://github.com/daiqizai/var-comm','https://github.com/daiqizai/var-comm.git'), 'Remote changed')
    files = file_map(v)
    require(set(v['source_packages'])==REQUIRED_PACKAGES,'Source package inventory incomplete')
    for package, paths in v['source_packages'].items():
        require(paths and len(paths)==len(set(paths)) and set(paths)<=set(files),
                'Source package missing exact reviewed file identity: '+package)
    require(set(v['publication_sources']) == set(HELPERS), 'Missing publisher source binding')
    for name, d2 in v['publication_sources'].items():
        require(d2['path'] == str(S/name) and d2 == pin(S/name), 'Publisher source changed')
        row = dict(d2,path=(S/name).relative_to(R).as_posix())
        require(files.get(row['path']) == row, 'Publisher source absent from exact scope')
    require(safe_file(__file__) == S/'publication.py', 'Unexpected publisher executable')
    require(set(v['repository_tools']) == {'update_repository_manifest.py','verify_repository.py','run_cpu_checks.py'},
            'Original repository tools required')
    for name, d2 in v['repository_tools'].items():
        require(d2['path'] == str(R/'tools'/name), 'Tool path differs'); checked(d2)
        require(subprocess.check_output(['git','show',v['parent']+':tools/'+name],cwd=R) == checked(d2),
                'Repository tool differs from original parent')
    require(set(v['documents']) == set(DOCS), 'Document history incomplete')
    for name,item in v['documents'].items():
        before = subprocess.check_output(['git','show',v['parent']+':'+name],cwd=R)
        require(item['before'] == {'bytes':len(before),'sha256':sha(before)}, 'Root document before identity differs')
        require(item['after'] == files[name], 'Root document final identity differs')
    for d2 in v.get('additional_evidence',[]): checked(d2)
    return v,files,evidence(v)


def environment():
    clean = {k:v for k,v in os.environ.items() if not k.upper().startswith(('GIT_TRACE','GCM_TRACE'))
             and k.upper() not in ('GIT_CURL_VERBOSE','GCM_DEBUG')}
    return dict(clean,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2',GIT_TERMINAL_PROMPT='0')


def run(name, argv, timeout, env=None):
    out=A/(name+'.stdout.txt'); err=A/(name+'.stderr.txt')
    parent=identity(os.getpid()); started=time.time(); mono=time.monotonic()
    flags=dict(timed_out=False,terminated=False,killed=False,metadata_error=None)
    with out.open('xb') as so, err.open('xb') as se:
        child=subprocess.Popen(argv,cwd=R,env=env or environment(),stdout=so,stderr=se,start_new_session=True)
        process=identity(child.pid)
        put(A/(name+'.launch.json'),dict(argv=argv,process=process,parent_process=parent,started_unix=started))
        try: rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            flags['timed_out']=True; flags['terminated']=True
            os.killpg(child.pid,signal.SIGTERM)
            try: rc=child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                flags['killed']=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=30)
    receipt=dict(actual_wait=True,returncode=rc,seconds=time.monotonic()-mono,
        process=process,parent_process=parent,argv=argv,stdout=pin(out),stderr=pin(err),**flags)
    put(A/(name+'.actual_wait.json'),receipt)
    require(type(rc) is int and rc==0 and not any(flags.values()), 'Command failed; no automatic retry: '+name)
    require(gone(process),'Child not gone after actual wait')
    return pin(A/(name+'.actual_wait.json'))


def index_matches(rows):
    require(set(git('diff','--cached','--name-only').splitlines()) == {r['path'] for r in rows}, 'Exact index scope differs')
    require(not git('diff','--name-only'), 'Unstaged tracked changes remain')
    for row in rows:
        data=subprocess.check_output(['git','show',':'+row['path']],cwd=R)
        require(len(data)==row['bytes'] and sha(data)==row['sha256'], 'Index differs from reviewed SHA')
        checked(dict(row,path=str(R/row['path'])))


def checks(d):
    v,files,proof=descriptor(d)
    require(not A.exists(), 'Publication directory exists; inspect retained attempt, never overwrite')
    require(not git('diff','--cached','--name-only'), 'Index already contains changes')
    paths=set(files)|{str((P/'authorization.json').relative_to(R))}
    require(set(git('diff','--name-only').splitlines()) <= paths, 'Unreviewed tracked changes')
    untracked=git('ls-files','--others','--exclude-standard','--','*.py','*.cpp','*.h','*.sh').splitlines()
    require(set(untracked)<=paths, 'Untracked self-written source absent from exact publication scope')
    exclusions=v['excluded_untracked']
    require(isinstance(exclusions,dict) and all(isinstance(reason,str) and reason.strip() for reason in exclusions.values()),
            'Every deliberate light-file exclusion needs an exact path and reason')
    for path in git('ls-files','--others','--exclude-standard').splitlines():
        in_scope=any(path.startswith(root+'/') for root in SOURCE_ROOTS+RESULT_ROOTS)
        if in_scope and path not in paths:
            require(path in exclusions,'Untracked light file omitted without explicit explanation: '+path)
    ignored=subprocess.run(['git','check-ignore','--no-index','--',*sorted(paths)],cwd=R,capture_output=True,text=True)
    require(ignored.returncode in (0,1) and not ignored.stdout.strip(), 'Existing Git filter excludes requested path')
    A.mkdir(parents=False,exist_ok=False)
    put(A/'scope.json',dict(descriptor=d,parent=v['parent'],files=sorted(paths),closure=proof,
                            root_process=identity(os.getpid()),scientific_calls=0))
    for start in range(0,len(paths),80):
        subprocess.run(['git','add','--sparse','--',*sorted(paths)[start:start+80]],cwd=R,check=True)
    update=run('manifest_update',[PY,'-B','tools/update_repository_manifest.py'],300)
    subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True)
    paths.add('release_manifest.json')
    rows=[dict(pin(R/p),path=p) for p in sorted(paths)]
    require(all(x['bytes']<=10000000 for x in rows),'Original size limit exceeded')
    index_matches(rows);put(A/'staged_files.json',rows)
    passed={}
    for name,argv,limit in (
        ('diff_check',['git','diff','--cached','--check'],120),
        ('verify_repository',[PY,'-B','tools/verify_repository.py'],1200),
        ('run_cpu_checks',[PY,'-B','tools/run_cpu_checks.py'],1200)):
        passed[name]=run(name,argv,limit)
    put(A/'checks.json',passed);index_matches(rows)
    descriptor(d)
    put(A/'checks_complete.json',dict(status='PASS',parent=v['parent'],descriptor=d,
        index_tree=git('write-tree'),manifest_update=update,checks_sha256=pin(A/'checks.json')['sha256'],
        staged_files_sha256=pin(A/'staged_files.json')['sha256'],files=len(rows),scientific_calls=0))
    print(json.dumps({'status':'CHECKS_PASS_NOT_COMMITTED','files':len(rows),'publication':str(A)}),flush=True)


def checked_state(d, committed=False):
    v,_,_=descriptor(d,check_parent=not committed)
    state=json.loads((A/'checks_complete.json').read_text()); rows=json.loads((A/'staged_files.json').read_text())
    require(state['status']=='PASS' and state['parent']==v['parent'] and state['descriptor']==d
        and state['staged_files_sha256']==pin(A/'staged_files.json')['sha256']
        and state['checks_sha256']==pin(A/'checks.json')['sha256'],'Frozen checked state differs')
    x=json.loads((A/'checks.json').read_text())
    require(set(x)=={'diff_check','verify_repository','run_cpu_checks'},'Incomplete original checks')
    for receipt in [state['manifest_update'],*x.values()]:
        w=load(receipt)
        require(w['actual_wait'] is True and type(w['returncode']) is int and w['returncode']==0
                and all(w[k] is False for k in ('timed_out','terminated','killed'))
                and w['metadata_error'] is None and gone(w['process']) and gone(w['parent_process']),
                'Actual check/controller failed or unfinished at phase boundary')
        checked(w['stdout']);checked(w['stderr'])
    require(len(rows)==state['files'],'Staged count changed')
    if not committed:
        index_matches(rows);require(git('write-tree')==state['index_tree'],'Checked tree changed')
    return v,state,rows


def commit(d):
    v,state,rows=checked_state(d)
    put(A/'commit_attempt.json',dict(parent=v['parent'],started_unix=time.time(),normal_commit=True,descriptor=d))
    name,email=subprocess.check_output(['git','show','-s','--format=%an%n%ae',v['parent']],cwd=R,text=True).splitlines()
    argv=['git','-c','user.name='+name,'-c','user.email='+email,'commit','-m',v['commit_message']]
    wait=run('commit',argv,180)
    head=git('rev-parse','HEAD')
    require(git('rev-parse','HEAD^')==v['parent'] and not git('status','--porcelain','--untracked-files=no'),
            'Committed lineage/tree differs')
    require(set(git('diff-tree','--no-commit-id','--name-only','-r','HEAD').splitlines())=={x['path'] for x in rows},
            'Committed exact scope differs')
    put(A/'commit_verified.json',dict(parent=v['parent'],head=head,normal_commit=True,descriptor=d,
        checks_sha256=state['checks_sha256'],staged_files_sha256=state['staged_files_sha256'],actual_wait=wait))
    print(json.dumps({'status':'COMMITTED_NOT_PUSHED','head':head}),flush=True)


def push(d, auth):
    v,state,rows=checked_state(d,committed=True)
    c=json.loads((A/'commit_verified.json').read_text());head=git('rev-parse','HEAD')
    require(c['normal_commit'] is True and c['descriptor']==d and c['head']==head
        and git('rev-parse','HEAD^')==v['parent'] and c['parent']==v['parent']
        and c['checks_sha256']==state['checks_sha256'] and c['staged_files_sha256']==state['staged_files_sha256']
        and not git('status','--porcelain','--untracked-files=no'),'Commit proof differs')
    cw=load(c['actual_wait']);require(cw['actual_wait'] is True and type(cw['returncode']) is int
        and cw['returncode']==0 and all(cw[k] is False for k in ('timed_out','terminated','killed'))
        and cw['metadata_error'] is None and gone(cw['process']) and gone(cw['parent_process']),
        'Commit child/controller not closed at push boundary')
    checked(cw['stdout']);checked(cw['stderr'])
    for row in rows:
        data=subprocess.check_output(['git','show',head+':'+row['path']],cwd=R)
        require(len(data)==row['bytes'] and sha(data)==row['sha256'],'Committed file SHA differs')
    require(not(A/'push_attempt.json').exists() and not(A/'push_verified.json').exists(),'Push already attempted; inspect outcome')
    require(isinstance(auth,str) and re.fullmatch('[A-Za-z0-9+/=]+',auth),'Missing in-memory authentication')
    env=dict(environment(),GIT_CONFIG_COUNT='3',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
        GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+auth,GIT_CONFIG_KEY_1='http.https://github.com.proxy',
        GIT_CONFIG_VALUE_1='http://127.0.0.1:23098',GIT_CONFIG_KEY_2='http.version',GIT_CONFIG_VALUE_2='HTTP/1.1')
    def remote():
        z=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=R,env=env,text=True,timeout=45).split()
        require(len(z)==2 and z[1]=='refs/heads/main','Remote reference unavailable');return z[0]
    before=remote();require(before in (v['parent'],head),'Remote advanced; preserve state, no force/reset')
    put(A/'push_attempt.json',dict(parent=v['parent'],head=head,remote_before=before,normal_push=True,started_unix=time.time()))
    wait=run('push',['git','push','origin','main'],180,env)
    require(remote()==head,'Push remote SHA not confirmed')
    receipt=dict(parent=v['parent'],head=head,verified_remote=head,normal_push=True,branch='main',
        repository='https://github.com/daiqizai/var-comm',descriptor=d,actual_wait=wait,
        staged_files=pin(A/'staged_files.json'))
    put(A/'push_verified.json',receipt);print(json.dumps(receipt),flush=True)


def main():
    # Process-local only: inherited trace destinations must never capture credentials.
    for key in list(os.environ):
        if key.upper().startswith(('GIT_TRACE','GCM_TRACE')) or key.upper() in ('GIT_CURL_VERBOSE','GCM_DEBUG'):
            del os.environ[key]
    parser=argparse.ArgumentParser()
    parser.add_argument('role',choices=('checks','commit','push'))
    parser.add_argument('--descriptor',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--bytes',required=True,type=int);args=parser.parse_args()
    d=dict(path=args.descriptor,sha256=args.sha256,bytes=args.bytes)
    try:
        if args.role=='checks': checks(d)
        elif args.role=='commit': commit(d)
        else: push(d,json.load(sys.stdin)['auth'])
    except BaseException as exc:
        if A.exists() and not(A/(args.role+'_failure.json')).exists():
            # Never persist credential-bearing input, environment or subprocess repr.
            put(A/(args.role+'_failure.json'),dict(type=type(exc).__name__,role=args.role,
                message='Publication stopped; inspect exact persisted stage receipts. No automatic retry.',
                outcome_requires_diagnosis=True))
        raise


if __name__=='__main__': main()
