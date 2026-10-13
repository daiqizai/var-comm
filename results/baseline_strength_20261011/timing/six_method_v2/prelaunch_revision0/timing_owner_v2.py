"""Finite six-method timing revision; explicit shared-H800 authorization.

Original registration and scientific sources are immutable. This wrapper needs
the coordinator's separately pinned E1-free timing window before launching.
prepare/readiness perform no model, channel, packet or quality calls.
"""
import argparse
import ast
import copy
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
sys.path.insert(0, str(ORIGINAL))
import timing_owner as old
from timing_contract import exclusive_snapshot as old_snapshot
import timing_audit_v2 as audit

ROOT, TMP = old.ROOT, old.TMP
PRIOR = old.OUT
OUT = ROOT / 'results/baseline_strength_20261011/timing/six_method_v2'
RUN = TMP / 'six_method_timing_v2'
RELEASE = OUT / 'coordination_release.json'
old.OUT, old.RUN = OUT, RUN
old.__file__ = __file__
read, write, sha, pin = old.read, old.write, old.sha, old.pin


def predecessor_unused():
    for parent in (PRIOR, OUT):
        assert not (parent / 'launch.json').exists(), str(parent) + ': existing attempt; do not replay'
        assert not (parent / 'actual_wait.json').exists(), str(parent) + ': prior actual child'
    assert not (TMP / 'six_method_timing_v1').exists(), 'Original runtime attempt exists'


def prepare(gpu_index):
    predecessor_unused()
    assert gpu_index == 0, 'The inherited admitted visual-device binding is H800 GPU0'
    assert not (OUT / 'registration.json').exists(), 'Already registered; do not overwrite'
    reg = read(PRIOR / 'registration.json')
    for p, h in {**reg['code_sha256'], **reg['input_sha256']}.items():
        assert sha(p) == h, p
    reg = copy.deepcopy(reg)
    reg.update(schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V2',
               previous_registration=pin(PRIOR / 'registration.json'),
               shared_resource_authorization='User explicitly withdrew all-GPU exclusive requirement; record shared environment',
               GPU_index=gpu_index, GPU_memory_fraction_cap_GiB=16, free_safety_GiB=2,
               initial_GPU_utilization_max_percent=25,
               coordination_release=str(RELEASE),
               timing_scope='same H800 and software; shared resident workloads; not an uncontended latency claim',
               audit_revision='static catalogue before/after case; named pure provenance hash time excluded and reported',
               old_registration_modified=False, scientific_algorithm_modified=False)
    for p in [Path(__file__), HERE / 'timing_audit_v2.py', HERE / 'test_timing_v2.py']:
        reg['code_sha256'][str(p)] = sha(p)
    write(OUT / 'registration.json', reg)
    write(OUT / 'preparation.json', dict(status='REGISTERED_NOT_EXECUTED',
          model_calls=0, packet_calls=0, full_cases=0, total_case_cap=1728,
          measured_case_cap=864, digital_packet_cap=2304,
          prior_registration=reg['previous_registration']))
    print('REGISTERED_V2_NO_SCIENTIFIC_CALLS')


def verify():
    reg = read(OUT / 'registration.json')
    assert sha(reg['previous_registration']['path']) == reg['previous_registration']['sha256']
    for p, h in {**reg['code_sha256'], **reg['input_sha256']}.items():
        assert sha(p) == h, p
    return reg


def coordinated():
    if not RELEASE.exists():
        return dict(released=False, reason='WAIT_E1_COORDINATOR_WINDOW')
    value = read(RELEASE)
    assert value['status'] == 'TIMING_WINDOW_RELEASED'
    assert value['quality_jobs_stopped'] is True
    assert value['no_quality_science_until_timing_actual_exit'] is True
    assert value['evidence'], 'Coordinator must pin E1 stage/closure receipts'
    for item in value['evidence']:
        assert sha(item['path']) == item['sha256'], item['path']
    return dict(released=True, release=pin(RELEASE), evidence=value['evidence'])


def resource_snapshot(repository, allowed_observer=None):
    reg = read(OUT / 'registration.json')
    original = old_snapshot(repository, allowed_observer)
    gpu_rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,utilization.memory',
        '--format=csv,noheader,nounits'], text=True)
    parsed = list(csv.reader(gpu_rows.splitlines()))
    entry = next(row for row in parsed if int(row[0]) == reg['GPU_index'])
    idx, uuid, name = int(entry[0]), entry[1].strip(), entry[2].strip()
    total, free, util, mem_util = map(float, entry[3:])
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    selected = [row for row in csv.reader(processes.splitlines()) if row and row[0].strip() == uuid]
    own = [row for row in selected if int(row[1]) == os.getpid()]
    constructing_or_running = allowed_observer is not None
    memory_ok = free >= 2048 if constructing_or_running else free >= 18432
    utilization_ok = constructing_or_running or util <= reg['initial_GPU_utilization_max_percent']
    coordination = coordinated()
    snapshot = dict(admitted=(coordination['released'] and memory_ok and utilization_ok and
                          not original['conflicting_repository_python_pids']),
                checked_unix=time.time(), selected_GPU=dict(index=idx, uuid=uuid, name=name,
                    memory_total_MiB=total, memory_free_MiB=free,
                    utilization_percent=util, memory_utilization_percent=mem_util),
                resident_compute_processes=selected, other_GPU_pids=original['gpu_pids'],
                conflicting_repository_python_pids=original['conflicting_repository_python_pids'],
                coordination=coordination, memory_admitted=memory_ok,
                initial_low_load_admitted=utilization_ok, shared_environment=True,
                exclusive_or_uncontended_claimed=False, jobs_stopped=0,
                GPU_global_settings_changed=False)
    if RUN.exists():
        with (RUN / 'resource_snapshots.jsonl').open('a') as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + '\n')
    return snapshot


def readiness():
    assert not (RUN / 'failure.json').exists(), 'Failure requires independent audit'
    snapshot = resource_snapshot(ROOT)
    write(OUT / 'resource_wait.json', dict(status='RESOURCE_READY_NOT_STARTED' if snapshot['admitted']
          else 'WAIT_SHARED_TIMING_WINDOW', **snapshot))
    return snapshot


def adapt_run():
    import inspect
    source = inspect.getsource(old.run)
    original = "CUDA_VISIBLE_DEVICES='0'"
    replacement = "CUDA_VISIBLE_DEVICES=str(reg['GPU_index'])"
    assert source.count(original) == 1
    changed = source.replace(original, replacement)
    assert ast.dump(ast.parse(changed.replace(replacement, original)), include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
    exec(compile(changed, __file__, 'exec'), old.__dict__)


def worker():
    import timing_runtime as runtime
    import timing_endpoints as endpoints
    original_build = runtime.build
    original_visual = runtime.visual_backend

    def shared_visual(*args, **kwargs):
        # Correct only the resource metadata in the original in-memory adapter.
        import inspect
        source = inspect.getsource(original_visual)
        a = 'shared_GPU=False,timing_measurement=True'
        b = 'shared_GPU=True,timing_measurement=True'
        assert source.count(a) == 1
        source = source.replace(a, b)
        a = 'Original timing6/2 and six-CPU affinity; exclusive admission checked by owner'
        b = 'Original timing6/2 and six-CPU affinity; user-authorized shared-H800 window'
        assert source.count(a) == 1
        source = source.replace(a, b)
        namespace = dict(vars(runtime))
        exec(compile(source, runtime.__file__, 'exec'), namespace)
        return namespace['visual_backend'](*args, **kwargs)

    runtime.visual_backend = shared_visual

    def build(registration, reservations, out, boundary):
        import torch
        device = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0, 16 * 1024**3 / device.total_memory), 0)
        system = original_build(registration, reservations, out, boundary)
        proof = audit.install(system, endpoints)
        write(OUT / 'actual_audit_adapter.json', proof)
        components = dict(system.backend.models)
        components.update(P_communication=system.endpoints['P100000'].model,
                          Swin=system.endpoints['Swin200000'].model)
        counts = {name:sum(p.numel() for p in model.parameters()) for name,model in components.items()}
        write(OUT / 'actual_model_parameter_counts.json', dict(
            component_parameter_counts=counts,
            counts_are_full_loaded_component_counts=True,
            P_communication_parameters_are_not_the_total_system=True,
            all_models_frozen=all(not p.requires_grad for model in components.values() for p in model.parameters()),
            runtime=pin(RUN / 'numerical_runtime.json')))
        return system

    runtime.build = build
    old.worker()
    journal = read(RUN / 'case_reservations.json')
    rows = []
    for item in journal['cases']:
        assert item['state'] == 'COMPLETE'
        t = item['timings']
        assert abs(t['raw_wall_tx_seconds'] - t['audit_excluded_tx_seconds'] - t['tx_seconds']) < 1e-7
        assert abs(t['raw_wall_rx_seconds'] - t['audit_excluded_rx_seconds'] - t['rx_seconds']) < 1e-7
        rows.append(dict(**item['case'], **{k:t[k] for k in ('tx_seconds','rx_seconds',
            'outer_seconds','channel_seconds','raw_wall_tx_seconds','raw_wall_rx_seconds',
            'raw_wall_outer_seconds','audit_excluded_tx_seconds','audit_excluded_rx_seconds','audit_excluded_seconds')}))
    with (OUT / 'timing_wall_and_audit.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    completion = read(OUT / 'completion.json')
    completion.update(shared_environment=True, exclusive_or_uncontended_claimed=False,
        raw_wall_and_audit=pin(OUT / 'timing_wall_and_audit.csv'),
        per_boundary_resource_history=pin(RUN / 'resource_snapshots.jsonl'),
        pure_provenance_exclusions=pin(OUT / 'actual_audit_adapter.json'))
    write(OUT / 'completion.json', completion)


old.verify, old.readiness, old.exclusive_snapshot = verify, readiness, resource_snapshot


def run():
    # Same lock as v1, held until the actual v2 child exits. No concurrent old
    # launch can consume any of the shared 1,728-call budget.
    lock = (PRIOR / 'owner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    predecessor_unused()
    adapt_run()
    old.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','readiness','run','_worker'])
    parser.add_argument('--gpu-index', type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == 'prepare': prepare(args.gpu_index)
        elif args.command == 'readiness': print(json.dumps(readiness(), indent=2))
        elif args.command == 'run': run()
        else: worker()
    except BaseException:
        if args.command in ('run','_worker') and (OUT / 'launch.json').exists():
            write(RUN / ('worker_failure.json' if args.command == '_worker' else 'failure.json'),
                  dict(status='STOPPED_NO_RETRY', traceback=traceback.format_exc()))
        raise
