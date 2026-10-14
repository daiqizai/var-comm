"""Explicit supplemental resource registration; preserve original v6 science."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import resource_adapter as adapter

ROOT = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
TMP = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
BASE = ROOT / 'experiments/baseline-delivery-20261012'
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/baseline_strength_20261011/timing/six_method_v6_resource_v1'
OLD_OUT = OUT.parent / 'six_method_v6'
RUN = TMP / 'six_method_timing_v6'
CONTROL = TMP / 'six_method_timing_v6_resource_v1'
OLD_REG_SHA = '54aafaca38b177266ef2c21cac23fce4dee2a0658e9bd9a0cfc5d2836baa269a'
V2 = BASE / 'timing_revision_v2/timing_owner_v2.py'
V6 = BASE / 'timing_revision_v6/timing_owner_v6.py'
POST = BASE / 'timing_postprocess_v1/postprocess.py'
ORIGINAL_PINS = {
    str(V2): 'ef35e3c0b1f1ed50840ca6473b596bad865e791ce28743e197b49fae88f0dc50',
    str(V6): '88392f7ba2483410b0c8ba5ab38b3b033f7ae33c12d38eef5e18b5a1e3af8d08',
    str(POST): '7a3414ff2088c00d8a7d4fff61d6baf93c67640f4ccc0861ecbb8c3449dbb8f2',
    str(BASE / 'timing_owner.py'): 'c59d4cf22c1c8da36861d9af8e28ade8d2c08d6219013ab4f03d6d88e26997d8',
}
HISTORICAL = {
    'case_reservations.json': '6d1cf07012757c6c3f852e82409f4af409353a45cceafbe2c53e07ca5a274b21',
    'resource_snapshots.jsonl': '01cdbbbdf99daffba27202297dd2be8e4487df523756949a053a78ea578a64dc',
    'numerical_runtime.json': '7f204f8d0bf24654d50f95f19b1223b1589856e7653c2a50f017b7b8534fb056',
    'worker_failure.json': '41398d8cfdb71169e9958e41581858a238fd1d57dda858a8eaa0097f2f6ee60c',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    p = Path(path)
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def checked(item):
    assert set(item) == {'path', 'sha256', 'bytes'}
    p = Path(item['path'])
    assert p.is_absolute() and p.is_file() and not p.is_symlink()
    assert type(item['bytes']) is int and pin(p) == item
    return read(p)


def write_new(path, value):
    p = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def proc(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = p.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], state=fields[0],
            argv=p.joinpath('cmdline').read_bytes().decode().split('\0')[:-1])
    except FileNotFoundError:
        return None


def binding_sources():
    for p, h in ORIGINAL_PINS.items():
        assert sha(p) == h, p
    assert sha(OLD_OUT / 'registration.json') == OLD_REG_SHA
    return {name: pin(HERE / name) for name in
        ('resource_adapter.py', 'timing_resource_owner.py', 'test_resource_adapter.py', 'README.md')}


def historical_evidence():
    p = TMP / 'six_method_timing_v5'
    for name, h in HISTORICAL.items():
        assert sha(p / name) == h
    journal = read(p / 'case_reservations.json')
    assert len(journal['cases']) == 577
    assert all(x['state'] == 'COMPLETE' for x in journal['cases'][:576])
    assert journal['cases'][-1]['state'] == 'RESERVED'
    assert read(p / 'worker_failure.json')['traceback'].endswith("KeyError: 'IndependentProvider'\n")
    rows = [json.loads(x) for x in (p / 'resource_snapshots.jsonl').read_text().splitlines()]
    own = [float(x[3]) for r in rows for x in r['resident_compute_processes'] if int(x[1]) == 1720426]
    assert len(rows) == 578 and len(own) == 577 and min(own) == 2644 and max(own) == 3128
    return dict(files=[pin(p / n) for n in HISTORICAL], boundary_samples=len(rows),
        own_memory_samples=len(own), minimum_own_MiB=min(own), maximum_observed_own_MiB=max(own),
        continuous_or_allocator_peak_measured=False, remaining_four_methods_peak_established=False,
        complete_calls=576, failed_partial_calls=1, failed_partial_is_OOM=False)


@contextlib.contextmanager
def no_prior_attempt():
    import fcntl
    handles = []
    try:
        for version in range(1, 7):
            p = OLD_OUT.parent / ('six_method_v%d' % version) / 'owner.lock'
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
            handles.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        forbidden = [RUN, TMP / 'six_method_timing_v6_owner', TMP / 'six_method_timing_v6_dispatch_v1']
        forbidden += [OLD_OUT / n for n in ('dispatch.json', 'launch.json', 'worker_launch.json', 'actual_wait.json', 'completion.json')]
        assert not any(p.exists() for p in forbidden), 'Original v6 was already attempted'
        assert not CONTROL.exists(), 'Resource wrapper was already attempted'
        yield dict(observed_unix=time.time(), all_six_original_owner_locks_unheld=True,
            all_exact_attempt_paths_absent=[str(p) for p in forbidden], scientific_calls=0)
    finally:
        for fd in reversed(handles):
            os.close(fd)


def register(authorization):
    auth = checked(authorization)
    assert auth['schema'] == 'TIMING_V6_STARTUP_RESOURCE_AUTHORIZATION_V1'
    assert auth['status'] == 'AUTHORIZED_18_TO_8_GIB'
    assert auth['user_instruction'] == '计时放开一下门槛'
    assert auth['startup_free_MiB'] == 8192 and auth['previous_startup_free_MiB'] == 18432
    assert auth['scientific_calls'] == 0 and auth['review_evidence']
    for item in auth['review_evidence']:
        checked(item)
    sources = binding_sources()
    assert auth['expected_source_sha256'] == {n: p['sha256'] for n, p in sources.items()}
    evidence = historical_evidence()
    _, proof = adapter.changed_function(V2.read_text(), 'resource_snapshot', adapter.STARTUP_OLD, adapter.STARTUP_NEW)
    _, postproof = adapter.changed_function(POST.read_text(), 'validate_resources', adapter.POST_OLD, adapter.POST_NEW)
    old = read(OLD_OUT / 'registration.json')
    assert old['GPU_index'] == 0 and old['GPU_memory_fraction_cap_GiB'] == 16 and old['free_safety_GiB'] == 2
    assert old['max_seconds'] == 13069 and old['remaining_complete_frames'] == 1152
    with no_prior_attempt() as observation:
        OUT.mkdir(exist_ok=False)
        reg = dict(schema='TIMING_V6_STARTUP_RESOURCE_REVISION_V1', status='REGISTERED_NOT_EXECUTED',
            original_registration=pin(OLD_OUT / 'registration.json'), authorization=authorization,
            sources=sources, original_source_sha256=ORIGINAL_PINS, never_started_observation=observation,
            historical_evidence=evidence, adapter_proof=proof, postprocess_adapter_proof=postproof,
            prior_startup_free_MiB=18432, startup_free_MiB=8192, GPU_index=0,
            initial_GPU_utilization_max_percent=25, allocator_cap_GiB=16, runtime_free_floor_MiB=2048,
            inherited_complete_calls=576, remaining_complete_calls=1152, remaining_seconds=13069,
            scientific_math_changed=False, numeric_runtime_changed=False, full_resource_conditions_identical=False,
            new_peak_or_uncontended_latency_claimed=False, automatic_retries=False,
            note='Only startup free-memory admission changes. Original v6 registration remains immutable historical metadata.')
        write_new(OUT / 'registration.json', reg)
    return pin(OUT / 'registration.json')


def verify_revision(expected_sha):
    assert isinstance(expected_sha, str) and len(expected_sha) == 64
    assert sha(OUT / 'registration.json') == expected_sha, 'Supplemental registration differs from root binding'
    reg = read(OUT / 'registration.json')
    assert reg['schema'] == 'TIMING_V6_STARTUP_RESOURCE_REVISION_V1'
    expected = dict(status='REGISTERED_NOT_EXECUTED', startup_free_MiB=8192, prior_startup_free_MiB=18432,
        GPU_index=0, initial_GPU_utilization_max_percent=25, allocator_cap_GiB=16, runtime_free_floor_MiB=2048,
        inherited_complete_calls=576, remaining_complete_calls=1152, remaining_seconds=13069,
        scientific_math_changed=False, numeric_runtime_changed=False, full_resource_conditions_identical=False,
        new_peak_or_uncontended_latency_claimed=False, automatic_retries=False)
    for k, value in expected.items():
        assert type(reg[k]) is type(value) and reg[k] == value, k
    assert reg['sources'] == binding_sources()
    auth = checked(reg['authorization']); checked(reg['original_registration'])
    assert auth['schema'] == 'TIMING_V6_STARTUP_RESOURCE_AUTHORIZATION_V1' and auth['status'] == 'AUTHORIZED_18_TO_8_GIB'
    assert auth['expected_source_sha256'] == {n: p['sha256'] for n, p in reg['sources'].items()}
    assert reg['original_source_sha256'] == ORIGINAL_PINS
    assert reg['adapter_proof'] == adapter.changed_function(V2.read_text(), 'resource_snapshot', adapter.STARTUP_OLD, adapter.STARTUP_NEW)[1]
    assert reg['postprocess_adapter_proof'] == adapter.changed_function(POST.read_text(), 'validate_resources', adapter.POST_OLD, adapter.POST_NEW)[1]
    assert reg['original_registration']['sha256'] == OLD_REG_SHA
    assert reg['historical_evidence'] == historical_evidence()
    return reg


def load_v6():
    sys.path.insert(0, str(V6.parent))
    import timing_owner_v6 as v6
    return v6


def install(v6, expected_sha):
    revision = verify_revision(expected_sha)
    proof = adapter.install_snapshot(v6, V2.read_text(), str(V2))
    assert proof == revision['adapter_proof']
    # Original run dispatches __file__ to _worker. Its old source bytes, source
    # pins, scientific functions and numerical-runtime comparison are unchanged.
    v6.old.__file__ = str(HERE / 'timing_resource_owner.py')
    return revision


def execute(role, expected_sha):
    assert role in ('run', '_worker')
    if role == '_worker':
        parent = read(CONTROL / 'owner_resource_execution.json')
        expected_sha = parent['resource_registration']['sha256']
        assert parent['resource_registration']['path'] == str(OUT / 'registration.json')
        checked(parent['resource_registration'])
        original_owner = read(OLD_OUT / 'launch.json')['owner']
        assert original_owner['argv'][-4:] == [str(HERE / 'timing_resource_owner.py'), 'run', '--expected-resource-registration', expected_sha]
        assert parent['process']['argv'] == original_owner['argv']
    verify_revision(expected_sha)
    v6 = load_v6(); revision = install(v6, expected_sha)
    if role == 'run':
        with no_prior_attempt():
            CONTROL.mkdir(exist_ok=False)
    else:
        owner = checked(read(CONTROL / 'owner_resource_execution.json')['resource_registration'])
        assert owner == revision
        original = read(OLD_OUT / 'launch.json')['owner']
        assert original['pid'] == os.getppid() and proc(os.getppid())['start_ticks'] == original['start_ticks']
    label = 'owner' if role == 'run' else 'worker'
    execution = dict(schema='TIMING_V6_ACTUAL_RESOURCE_EXECUTION_V1', role=role,
        process=proc(os.getpid()), parent_process=proc(os.getppid()),
        resource_registration=pin(OUT / 'registration.json'), original_registration=revision['original_registration'],
        sources=revision['sources'], original_source_sha256=ORIGINAL_PINS,
        actual_startup_free_MiB=8192, historical_startup_free_MiB=18432,
        actual_worker_entry=str(HERE / 'timing_resource_owner.py'), adapter_proof=revision['adapter_proof'])
    write_new(CONTROL / (label + '_resource_execution.json'), execution)
    try:
        (v6.run if role == 'run' else v6.worker)()
        assert (OLD_OUT / 'launch.json').exists(), 'Resource gate changed before actual launch; no retry'
        verify_revision(expected_sha)
        write_new(CONTROL / (label + '_returned.json'), dict(status='ORIGINAL_V6_RETURNED_SUCCESS',
            process=execution['process'], original_completion=pin(OLD_OUT / 'completion.json'),
            resource_registration=pin(OUT / 'registration.json')))
    except BaseException:
        write_new(CONTROL / (label + '_failure.json'), dict(status='STOPPED_NO_RETRY', traceback=traceback.format_exc()))
        raise


def closed_binding(expected_sha):
    revision = verify_revision(expected_sha)
    assert not list(CONTROL.glob('*failure.json'))
    owner = read(CONTROL / 'owner_resource_execution.json'); worker = read(CONTROL / 'worker_resource_execution.json')
    original_owner = read(OLD_OUT / 'launch.json')['owner']; original_worker = read(OLD_OUT / 'worker_launch.json')
    for receipt, role in ((owner, 'owner'), (worker, 'worker')):
        assert receipt['schema'] == 'TIMING_V6_ACTUAL_RESOURCE_EXECUTION_V1'
        assert receipt['role'] == ('run' if role == 'owner' else '_worker')
        assert receipt['resource_registration'] == pin(OUT / 'registration.json')
        assert receipt['original_registration'] == revision['original_registration']
        assert receipt['original_source_sha256'] == ORIGINAL_PINS and receipt['adapter_proof'] == revision['adapter_proof']
        assert receipt['sources'] == revision['sources'] and receipt['actual_startup_free_MiB'] == 8192
        identity = receipt['process']; current = proc(identity['pid'])
        assert current is None or current['start_ticks'] != identity['start_ticks'], 'Resource process still exists'
        returned = read(CONTROL / (role + '_returned.json'))
        assert returned['status'] == 'ORIGINAL_V6_RETURNED_SUCCESS'
        assert returned['resource_registration'] == pin(OUT / 'registration.json')
        assert returned['process'] == identity and returned['original_completion'] == pin(OLD_OUT / 'completion.json')
    assert owner['process']['pid'] == original_owner['pid'] and owner['process']['start_ticks'] == original_owner['start_ticks']
    assert owner['process']['argv'] == original_owner['argv']
    assert owner['process']['argv'][-4:] == [str(HERE / 'timing_resource_owner.py'), 'run', '--expected-resource-registration', expected_sha]
    assert worker['process']['pid'] == original_worker['pid'] and worker['parent_process']['pid'] == original_owner['pid']
    assert worker['parent_process']['start_ticks'] == original_owner['start_ticks']
    assert worker['process']['argv'][-2:] == [str(HERE / 'timing_resource_owner.py'), '_worker']
    wait = read(OLD_OUT / 'actual_wait.json')
    assert type(wait['exit_code']) is int and wait['actual_child_waited'] is True and wait['timed_out'] is False
    assert wait == dict(actual_child_waited=True, exit_code=0, timed_out=False)
    return dict(resource_registration=pin(OUT / 'registration.json'),
        execution=[pin(CONTROL / n) for n in ('owner_resource_execution.json', 'worker_resource_execution.json', 'owner_returned.json', 'worker_returned.json')],
        actual_wait=pin(OLD_OUT / 'actual_wait.json'), admission=pin(OLD_OUT / 'resource_wait.json'),
        original_registration=revision['original_registration']), original_worker['pid']


def postprocess(output, expected_sha):
    binding, worker_pid = closed_binding(expected_sha)
    spec = importlib.util.spec_from_file_location('_frozen_timing_postprocess', POST)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = module.validate_resources
    relaxed, proof = adapter.compile_function(POST.read_text(), vars(module), str(POST),
        'validate_resources', adapter.POST_OLD, adapter.POST_NEW)
    assert proof == verify_revision(expected_sha)['postprocess_adapter_proof']
    module.validate_resources = adapter.resource_validator(original, relaxed,
        checked(binding['original_registration']), checked(binding['admission']), worker_pid)
    write_new(CONTROL / 'postprocess_binding.json', dict(binding=binding, proof=proof,
        process=proc(os.getpid()), original_statistics_functions_changed=False))
    module.deliver(OLD_OUT, Path(output), OLD_REG_SHA)
    completion = pin(Path(output) / 'completion.json')
    write_new(Path(output) / 'resource_revision.json', dict(
        schema='TIMING_V6_DELIVERED_STARTUP_RESOURCE_REVISION_V1', binding=binding,
        proof=proof, original_postprocess_completion=completion,
        startup_free_MiB=8192, prior_startup_free_MiB=18432, allocator_cap_GiB=16,
        runtime_free_floor_MiB=2048, shared_environment=True,
        full_resource_conditions_identical=False, original_statistics_functions_changed=False))
    write_new(CONTROL / 'postprocess_returned.json', dict(
        status='ORIGINAL_POSTPROCESS_AND_RESOURCE_SIDECAR_RETURNED_SUCCESS',
        original_completion=completion, resource_revision=pin(Path(output) / 'resource_revision.json'),
        binding=pin(CONTROL / 'postprocess_binding.json')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('register', 'verify', 'run', '_worker', 'postprocess'))
    p.add_argument('--authorization'); p.add_argument('--authorization-sha256'); p.add_argument('--authorization-bytes', type=int)
    p.add_argument('--output')
    p.add_argument('--expected-resource-registration')
    a = p.parse_args()
    if a.command == 'register':
        print(json.dumps(register(dict(path=a.authorization, sha256=a.authorization_sha256, bytes=a.authorization_bytes))))
    elif a.command == 'verify':
        verify_revision(a.expected_resource_registration); print('RESOURCE_METADATA_VERIFIED_ZERO_MODEL_CALLS')
    elif a.command == 'postprocess':
        assert a.output; postprocess(a.output, a.expected_resource_registration)
    else:
        execute(a.command, a.expected_resource_registration)


if __name__ == '__main__':
    main()
