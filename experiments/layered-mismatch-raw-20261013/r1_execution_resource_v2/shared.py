"""Metadata, resource and finite-owner utilities; no science at import time."""
from __future__ import annotations
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
import contract

BASE = Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
RT = BASE / 'code/liulu/var_comm_runtime_20261010'
TMP = BASE / 'liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013/r1_execution_v1'
OUT = ROOT / 'results/layered_mismatch_raw_20261013/r1_execution_v1'
ES = ROOT / 'experiments/entropy-partial-mismatch-20261010/scripts'
TS = ROOT / 'experiments/wcl-evidence-closure-20261009/scripts'
TIMING_OUT = ROOT / 'results/baseline_strength_20261011/timing/six_method_v6'
TIMING_RUN = BASE / 'liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/six_method_timing_v6'
TIMING_REGISTRATION_SHA = '54aafaca38b177266ef2c21cac23fce4dee2a0658e9bd9a0cfc5d2836baa269a'
PY_CPU = str(RT / 'envs/frozen_ldpc_py311/bin/python')
PY_VISUAL = str(RT / 'envs/frozen_visual_py310/bin/python')
MASTER_SCHEMA = 'LAYERED_RAW_R1_EXECUTION_REGISTRATION_V1'
VISUAL_SCHEMA = 'LAYERED_RAW_R1_RX_VISUAL_REGISTRATION_V1'
FLAGS = dict(matmul_tf32=False, cudnn_tf32=False, precision='highest', cudnn_benchmark=False,
             cudnn_deterministic=True, deterministic=True, threads=6, interop_threads=2)
RESOURCE = dict(prelaunch_free_GiB=15, runtime_free_plus_own_GiB=14, allocator_cap_GiB=12,
                absolute_free_GiB=2, math_threads=6, interop_threads=2, scheduling_CPUs=2,
                shared_environment=True, exclusive_host_required=False, timing_priority=True)
require = contract.require
digest = contract.digest
canonical = contract.canonical


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path).absolute()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(value):
    path = Path(value['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and sha(path) == value['sha256'], 'Changed bound file: ' + str(path))
    if 'bytes' in value:
        require(path.stat().st_size == value['bytes'], 'Bound file size changed')
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def readpin(value):
    return read(checked(value))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
    return pin(path)


def status(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(canonical(value) + '\n', encoding='utf8'); os.replace(temporary, path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_runtime(runtime):
    root = Path(runtime['root'])
    require(root == TMP and runtime['python_cpu'] == PY_CPU and runtime['python_visual'] == PY_VISUAL,
            'Original environments and personal temporary root required')
    for stage in ('phy', 'projection', 'visual', 'score', 'selection'):
        require(Path(runtime[stage + '_directory']) == root / stage, 'Registered stage storage differs')
    require(Path(runtime['stop_file']) == OUT / 'STOP' and runtime['max_seconds'] > 0
            and runtime['deadline_unix'] > 0 and type(runtime['target_gpu']) is int and 0 <= runtime['target_gpu'] < 4,
            'Finite registered resource/STOP boundary required')
    for key in ('cpu_affinity', 'phy_cpu_affinity'):
        require(len(runtime[key]) == 2 and len(set(runtime[key])) == 2
                and all(type(x) is int and x >= 0 for x in runtime[key]), 'Two CPU slots required')
    require(not set(runtime['cpu_affinity']) & set(runtime['phy_cpu_affinity']), 'CPU slots overlap')


def validate_registration(registration):
    reg = readpin(registration)
    require(reg['schema'] in (MASTER_SCHEMA, VISUAL_SCHEMA)
            and reg['status'] == ('REGISTERED_R1_EXECUTION_V1' if reg['schema'] == MASTER_SCHEMA else 'REGISTERED_R1_VISUAL_V1'),
            'Reviewed real R1 registration required')
    require(reg['plan_id'] == contract.PLAN_ID and reg['caps'] == contract.R1_CAPS
            and reg['numeric_flags'] == FLAGS and reg['resource_policy'] == RESOURCE
            and reg['automatic_retry'] is False and reg['automatic_R2'] is False
            and reg['training_updates'] == reg['additional_qualification_frames'] == reg['bootstrap_calls'] == 0,
            'Frozen finite R1 scope differs')
    validate_runtime(reg['runtime'])
    require(0 < reg['expected_physical_frames'] <= 36000 and 0 < reg['expected_logical_var_images'] <= 46800,
            'Physical/logical limits expanded')
    for path, expected in reg['code_bindings'].items():
        require(sha(path) == expected, 'Bound code changed: ' + path)
    require(reg['code_bindings'][str(Path(__file__))] == sha(__file__), 'Different shared entry loaded')
    for value in reg['pins'].values():
        checked(value)
    require(reg['pins']['common_ledger']['path'] == str(ROOT / 'experiments/entropy-main-common500-20261013/common.py'),
            'Unqualified scientific call ledger')
    if reg['schema'] == MASTER_SCHEMA:
        checked(reg['visual_request'])
        require(reg['namespace'] == contract.NAMESPACE and reg['metric_names'] == ['psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine'],
                'R1 metric or noise scope changed')
        nominations = readpin(reg['pins']['nominations'])
        require(nominations['status'] == 'FROZEN_R1_NOMINATIONS' and nominations['selection_uses_holdout'] is False,
                'Actual frozen R0 nominations required')
        contract.validate_shortlist(nominations['rows'])
        checked(reg['phy_registration']); checked(reg['visual_registration'])
    else:
        forbidden = {'sources', 'source_manifest', 'cal200_sources', 'nominations', 'metric_config', 'phy_registration', 'visual_request'}
        require(not (set(reg) | set(reg['pins'])) & forbidden, 'Visual registration exposes source or TX selection records')
        require(set(reg['pins']) == {'common_ledger'}, 'Visual process only receives its qualified ledger binding')
        checked(reg['model_bundle'])
    return reg


def process_identity(pid):
    try:
        path = Path('/proc') / str(pid); stat = (path / 'stat').read_text()
        tail = stat[stat.rfind(')') + 2:].split()
        return dict(pid=int(pid), start_ticks=int(tail[19]), state=tail[0])
    except (OSError, ValueError):
        return None


def guard_for(reg, out, parent):
    launch = read(Path(reg['runtime']['root']) / 'owner_launch.json')
    expected = launch['process']
    require(parent == expected['pid'], 'Worker parent differs from actual owner')
    def guard():
        actual = process_identity(parent)
        require(os.getppid() == parent and actual is not None and actual['state'] != 'Z'
                and actual['start_ticks'] == expected['start_ticks'], 'Actual owner lost')
        require(time.time() < min(reg['runtime']['deadline_unix'], launch['deadline_unix']), 'Registered total deadline exhausted')
        require(not any(p.exists() for p in (Path(reg['runtime']['stop_file']), Path(reg['runtime']['root'])/'STOP', Path(out)/'STOP')),
                'Registered STOP requested')
    return guard


def load_ledger(reg):
    return load(checked(reg['pins']['common_ledger']), '_r1_frozen_scientific_ledger').Ledger


def resource_adapter_sources(resource):
    """Fail closed if the original current-process-only resource guard changes."""
    assessor = inspect.getsource(resource.assess_snapshot)
    replacements = {
        'free>=20*GiB if prelaunch else free>=4*GiB': 'free>=15*GiB if prelaunch else free>=2*GiB',
        'free+own>=20*GiB': 'free+own>=14*GiB',
        '0<=own<=16*GiB': '0<=own<=12*GiB'}
    for old, new in replacements.items():
        require(assessor.count(old) == 1, 'Original assessor shape changed: ' + old)
        assessor = assessor.replace(old, new)
    observer = inspect.getsource(resource.own_allocator_observation)
    require(observer.count('<=16*GiB') == 1, 'Original own-allocator observation changed')
    observer = observer.replace('<=16*GiB', '<=12*GiB').replace('frozen16GiB cap', 'reviewed12GiB cap')
    return assessor, observer


def resource_device(reg, out):
    # Same reviewed per-device adapter used for R0; only storage binding and
    # explicitly registered shared-device threshold differ from the old source.
    for path in (ES, TS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import h800_alternate_gpu_resource_v3 as resource
    namespace = dict(vars(resource)); policy = dict(resource.POLICY_BASE)
    policy.update(schema='R1_RAW_SHARED_RESOURCE_V2', prelaunch_free_bytes=15 << 30,
                  runtime_free_plus_own_reserved_bytes=14 << 30,
                  runtime_free_bytes=2 << 30, own_allocator_cap_bytes=12 << 30)
    namespace['POLICY_BASE'] = policy
    fn = resource.policy
    namespace['policy'] = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    assessor, observer = resource_adapter_sources(resource)
    exec(compile(assessor + '\n' + observer, resource.__file__, 'exec'), namespace)
    fn = resource.DeviceAdapter.resource_snapshot
    snapshot = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    def bind_evidence(self, directory, request_sha):
        path = resource.g.inside(directory, base=TMP)
        require(path.is_dir() and resource.g.SHA_RE.fullmatch(request_sha), 'Resource evidence binding invalid')
        self.evidence_directory = path; self.request_sha = request_sha
    adapter = type('R1RawSharedDevice', (resource.DeviceAdapter,),
                   dict(resource_snapshot=snapshot, bind_resource_evidence=bind_evidence))
    device = adapter(reg['device_receipt'], reg['runtime']['target_gpu'])
    device.bind_resource_evidence(out, digest(reg))
    return device, resource.g


def admission(reg, gate):
    device, gate_module = resource_device(reg, gate)
    try:
        snapshot, _ = device.resource_snapshot(gate_module.helper(), prelaunch=True)
        return dict(status='READY', snapshot=snapshot, model_calls=0)
    except Exception as error:
        if type(error).__name__ != 'ResourceBusy':
            raise
        return dict(status='WAIT', reason=str(error), model_calls=0)
