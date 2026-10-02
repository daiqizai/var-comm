"""Publish the unchanged concurrent metric runtime after scientific delivery.

This entry point performs CPU repository checks. It neither launches scoring
nor claims that CPU tests qualify reconstructed images or metric models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


SUFFIXES = {'.py', '.md'}
MAX_RECEIPT_BYTES = 8_000_000
ORIGINS = {'git@github.com:daiqizai/var-comm.git', 'https://github.com/daiqizai/var-comm.git'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8_000_000), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                              allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def verify(mapping, label='bindings'):
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError('Missing ' + label)
    for path, digest in mapping.items():
        if not re.fullmatch(r'[0-9a-f]{64}', str(digest)) or sha(path) != digest:
            raise RuntimeError('Changed ' + label + ': ' + path)


def published(record):
    if (record.get('status') != 'PUSHED' or record.get('checks') != 'PASS'
            or not re.fullmatch(r'[0-9a-f]{40}', str(record.get('commit', '')))
            or record.get('commit') != record.get('remote_commit')):
        raise RuntimeError('A checked normal-push receipt is required')


def source_files(folder):
    folder = Path(folder).resolve(); answer = []
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.is_symlink() or path.suffix not in SUFFIXES:
            raise RuntimeError('Unexpected runtime source artifact: ' + str(path))
        answer.append(path)
    if not answer:
        raise RuntimeError('Frozen runtime bundle is empty')
    return answer


def bindings(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(paths)}


def process_snapshot(proc=Path('/proc')):
    if not proc.is_dir():
        raise RuntimeError('Linux process identity evidence is required')
    answer = {}
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / 'stat').read_text(); fields = stat[stat.rfind(')') + 2:].split()
            answer[int(path.name)] = dict(state=fields[0], start_ticks=fields[19],
                command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            answer[int(path.name)] = dict(unreadable=True)
    return answer


def require_exited(record, processes, label):
    if (type(record.get('pid')) is not int or record['pid'] <= 1
            or not str(record.get('start_ticks', '')).isdigit()):
        raise RuntimeError('Missing registered ' + label + ' process identity')
    proc = processes.get(record['pid'])
    if proc and proc.get('unreadable'):
        raise RuntimeError('Cannot inspect registered ' + label)
    if proc and proc.get('state') != 'Z' and proc.get('start_ticks') == str(record['start_ticks']):
        raise RuntimeError(label + ' has not exited')


def parent_gate(root, processes):
    parent = root / 'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
    files = [parent / name for name in ('completion.json', 'supervisor_status.json', 'supervisor_launch.json')]
    complete, status, launch = map(read, files)
    if (complete.get('status') != 'AUTHORIZED_TWO_METHODS_COMPLETE'
            or complete.get('synthetic') is not False or complete.get('training_updates') != 0
            or complete.get('stop') is not True or status.get('status') != 'COMPLETE'
            or status.get('stopped_after_authorized_scope') is not True):
        raise RuntimeError('Original two-method experiment is not fully complete')
    pub = complete.get('publication', {}); published(pub); verify(pub['source_bindings'], 'original source')
    publication_path = parent / 'm2_publication.json'
    if read(publication_path) != pub:
        raise RuntimeError('Original final publication and completion differ')
    files.append(publication_path); require_exited(launch, processes, 'original supervisor')
    # The speed controller may have taken over the original supervisor lock.
    speed = root / 'outputs/METRIC-SPEED-20261002/controller_completion.json'
    if speed.exists():
        record = read(speed)
        if (record.get('status') != 'COMPLETE' or record.get('training_updates') != 0
                or record.get('original_completion_sha256') != sha(files[0])):
            raise RuntimeError('Final receiver controller completion differs')
        verify(record['source_bindings'], 'receiver controller source')
        require_exited(record, processes, 'receiver controller'); files.append(speed)
    elif status.get('pid') != launch['pid']:
        raise RuntimeError('Unknown controller took over the original supervisor')
    markers = ('/experiments/scale-causal-partial-residual-20261002/',
               '/experiments/metric-speed-20261002/')
    live = [pid for pid, p in processes.items() if p.get('state') != 'Z'
            and any(m in p.get('command', '').replace('\\', '/') for m in markers)]
    if live:
        raise RuntimeError('Original scientific workers remain active: ' + str(live))
    return dict(status='PARENT_DELIVERED_AND_EXITED', inputs=bindings(files),
                parent_commit=pub['commit'], training_updates=0)


def concurrent_gate(out, runtime, registration_path, completion_path, processes,
                    controller_pid=None, controller_ticks=None):
    bound = bindings(source_files(runtime)); registration = read(registration_path)
    registered = registration.get('runtime_source_bindings', registration.get('source_bindings', {}))
    verify(registered, 'concurrent runtime registration')
    selected = {p: h for p, h in registered.items() if Path(p).resolve().is_relative_to(runtime.resolve())}
    if selected != bound:
        raise RuntimeError('Runtime bundle inventory changed after registration')
    done = read(completion_path)
    if (done.get('status') != 'SEALED_CACHE_SNAPSHOT_COMPLETE' or done.get('synthetic') is not False
            or done.get('training_updates') != 0 or done.get('policy_selection_updates') != 0
            or done.get('parity_passed') is not True
            or isinstance(done.get('sources_complete'), bool)
            or not isinstance(done.get('sources_complete'), int)
            or not 1 <= done['sources_complete'] <= 100):
        raise RuntimeError('A sealed concurrent cache snapshot with genuine replay parity is required')
    verify(done.get('inputs'), 'concurrent scoring inputs'); verify(done.get('outputs'), 'concurrent scoring outputs')
    if done.get('runtime_source_bindings', done.get('source_bindings')) != registered:
        raise RuntimeError('Concurrent scoring used another source registration')
    if done['inputs'].get(str(registration_path.resolve())) != sha(registration_path):
        raise RuntimeError('Concurrent completion did not bind its runtime registration')
    runtime_registration = out / 'runtime_registration.json'
    worker_launch = out / 'partial_launch.json'
    runtime_record = read(runtime_registration)
    runtime_registered = runtime_record.get('runtime_source_bindings', runtime_record.get('source_bindings', {}))
    verify(runtime_registered, 'controller runtime registration')
    selected_runtime = {p: h for p, h in runtime_registered.items()
                        if Path(p).resolve().is_relative_to(runtime.resolve())}
    if selected_runtime != bound:
        raise RuntimeError('Controller and scorer froze different runtime inventories')
    for path in (runtime_registration, worker_launch):
        if done['inputs'].get(str(path.resolve())) != sha(path):
            raise RuntimeError('Concurrent snapshot did not bind ' + path.name)
    require_exited(read(worker_launch), processes, 'concurrent scorer')
    allow = {os.getpid()}
    controller = None
    if controller_pid is not None:
        controller = read(out / 'controller_launch.json')
        if (type(controller.get('pid')) is not int or controller['pid'] <= 1
                or controller.get('pid') != controller_pid
                or str(controller.get('start_ticks')) != str(controller_ticks)):
            raise RuntimeError('Authorized CPU publication supervisor identity differs')
        p = processes.get(controller_pid)
        if not p or p.get('unreadable') or p.get('start_ticks') != str(controller_ticks) or p.get('state') == 'Z':
            raise RuntimeError('Authorized publication supervisor is not the registered live process')
        if (runtime.resolve() / 'controller.py').as_posix() not in p.get('command', '').replace('\\', '/'):
            raise RuntimeError('Authorized supervisor is running an unexpected entry point')
        allow.add(controller_pid)
    # Any replay/scoring child in this bundle must have exited. Only an explicitly
    # registered CPU publication supervisor can be exempted by the caller.
    marker = runtime.resolve().as_posix() + '/'
    live = [pid for pid, p in processes.items() if pid not in allow and p.get('state') != 'Z'
            and marker in p.get('command', '').replace('\\', '/')]
    if live:
        raise RuntimeError('Concurrent runtime workers remain active: ' + str(live))
    return bound, dict(status='SEALED_CONCURRENT_CACHE_READY', sources_complete=done['sources_complete'],
        inputs=bindings([registration_path, completion_path, runtime_registration, worker_launch]),
        runtime_source_bindings=bound, controller_exception=controller,
        policy_selection_updates=0, training_updates=0, scientific_result=False,
        qualification_claim='No new real-weight qualification is claimed by this publisher')


def predecessor_gate(root, runtime_record, processes):
    """Keep the retired R3/R2/R1 runtimes and their evidence unchanged."""
    previous = runtime_record.get('previous_extension')
    inputs, generations = {}, []
    names = ('METRIC-CONCURRENT-R3-20261002', 'METRIC-CONCURRENT-R2-20261002',
             'METRIC-CONCURRENT-20261002')
    for level, name in enumerate(names):
        if previous is None:
            if level == 0: raise RuntimeError('R4 must bind its retired R3 predecessor')
            break
        prior = root / 'outputs' / name
        if Path(previous.get('out', '')).resolve() != prior.resolve():
            raise RuntimeError('Unexpected concurrent runtime ancestry')
        proof = previous.get('bindings'); verify(proof, 'retired runtime evidence')
        required = [prior / n for n in ('runtime_registration.json', 'admission.json', 'handoff.json',
                                       'controller_launch.json', 'controller_status.json', 'partial_launch.json')]
        if not {str(p) for p in required}.issubset(proof):
            raise RuntimeError('Retired runtime evidence is incomplete')
        reg, admission, handoff, launch, status, worker = map(read, required)
        bound = bindings(source_files(prior / 'runtime'))
        if (reg.get('runtime_source_bindings') != bound or admission.get('source_bindings') != bound
                or handoff.get('new_source_bindings') != bound or launch.get('source_bindings') != bound
                or worker.get('source_bindings') != bound
                or reg.get('admission_sha256') != sha(required[1])
                or reg.get('handoff_sha256') != sha(required[2])
                or launch.get('runtime_registration_sha256') != sha(required[0])
                or worker.get('runtime_registration_sha256') != sha(required[0])
                or worker.get('owner', {}).get('pid') != launch.get('pid')
                or str(worker.get('owner', {}).get('start_ticks')) != str(launch.get('start_ticks'))
                or status.get('pid') != launch.get('pid')):
            raise RuntimeError('Retired runtime source/owner lineage differs')
        if level == 0:
            if (status.get('status') != 'WAITING_FOR_CONCURRENT_ADMISSION'
                    or status.get('reason') != 'M2_SLOWDOWN_OVER_20_PERCENT_THREE_INTERVALS'
                    or status.get('worker_pid') is not None):
                raise RuntimeError('R3 was not retired at its registered slowdown boundary')
        elif status.get('status') != 'FAILED':
            raise RuntimeError('Historical failed runtime evidence differs')
        require_exited(launch, processes, name + ' owner')
        require_exited(worker, processes, name + ' latest worker')
        marker = (prior / 'runtime').as_posix() + '/'
        if any(p.get('state') != 'Z' and marker in p.get('command', '').replace('\\', '/')
               for p in processes.values()):
            raise RuntimeError('A retired concurrent runtime is still active: ' + name)
        inputs.update(proof); inputs.update(bound)
        generations.append(dict(output=str(prior), source_bindings=bound, evidence_bindings=proof,
                                original_source_preserved=True, processes_exited=True))
        previous = reg.get('previous_extension')
    if previous is not None:
        raise RuntimeError('Unexpected additional concurrent runtime ancestor')
    return dict(status='PREDECESSORS_RETIRED_AND_PRESERVED', inputs=inputs, generations=generations)


def inherited_cache_gate(root, registration, completion):
    """Verify unchanged R3 values in every inherited R4 checkpoint."""
    prior = root / 'outputs/METRIC-CONCURRENT-R3-20261002'
    inherited = registration.get('inherited_cache')
    if not isinstance(inherited, dict): raise RuntimeError('R4 must register the preserved R3 cache')
    proof = inherited.get('bindings'); verify(proof, 'inherited R3 cache')
    reg_path, snapshot_path = prior / 'concurrent_registration.json', prior / 'partial_completion.json'
    if any(proof.get(str(p)) != sha(p) for p in (reg_path, snapshot_path)):
        raise RuntimeError('Inherited R3 registration/snapshot is not bound')
    original_reg, snapshot = read(reg_path), read(snapshot_path)
    if (snapshot.get('status') != 'SEALED_CACHE_SNAPSHOT_COMPLETE' or snapshot.get('synthetic') is not False
            or snapshot.get('parity_passed') is not True or snapshot.get('training_updates') != 0
            or snapshot.get('policy_selection_updates') != 0
            or inherited.get('registration_sha256') != sha(reg_path)
            or snapshot.get('source_bindings') != original_reg.get('source_bindings')):
        raise RuntimeError('Original R3 cache snapshot is not verified')
    verify(snapshot['inputs'], 'R3 cache inputs'); verify(snapshot['outputs'], 'R3 cache outputs')
    indexes = inherited.get('source_indexes')
    if (not isinstance(indexes, list) or not indexes or any(type(i) is not int or not 0 <= i < 100 for i in indexes)
            or indexes != sorted(set(indexes)) or len(indexes) != snapshot.get('sources_complete')):
        raise RuntimeError('Inherited source indexes differ from the R3 snapshot')
    old_paths = {str(prior / 'source_checkpoints' / f'{i:03d}.json') for i in indexes}
    if set(snapshot['outputs']) != old_paths:
        raise RuntimeError('Inherited R3 checkpoint coverage differs')
    if any(registration.get(k) != original_reg.get(k) for k in
           ('sources', 'studies', 'modelmanifest_sha256', 'metric_evaluator_identity', 'numerical_runtime')):
        raise RuntimeError('R4 inherited source/model/runtime differs from R3')
    if any(registration.get('asset_inputs', {}).get(p) != h or completion['inputs'].get(p) != h
           for p, h in proof.items()):
        raise RuntimeError('R4 omitted inherited cache bindings')
    def identity(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    exclude = {'registration_sha256', 'payload_sha256', 'inherited_origin'}
    for i in indexes:
        old_path = prior / 'source_checkpoints' / f'{i:03d}.json'
        new_path = root / 'outputs/METRIC-CONCURRENT-R4-20261002/source_checkpoints' / f'{i:03d}.json'
        old, new = read(old_path), read(new_path)
        if (proof.get(str(old_path)) != sha(old_path) or completion['outputs'].get(str(new_path)) != sha(new_path)
                or old.get('registration_sha256') != identity(original_reg)
                or new.get('registration_sha256') != identity(registration)
                or any(p.get('payload_sha256') != identity({k: v for k, v in p.items() if k != 'payload_sha256'})
                       for p in (old, new))
                or identity({k: v for k, v in old.items() if k not in exclude}) !=
                   identity({k: v for k, v in new.items() if k not in exclude})):
            raise RuntimeError('Inherited checkpoint values changed: ' + str(i))
        expected_origin = dict(schema_version=1, source_index=i, checkpoint_path=str(old_path),
            checkpoint_sha256=sha(old_path), registration_path=str(reg_path), registration_sha256=sha(reg_path),
            snapshot_path=str(snapshot_path), snapshot_sha256=sha(snapshot_path),
            original_values_changed=False, original_parity_reused=True)
        if new.get('inherited_origin') != expected_origin:
            raise RuntimeError('Inherited checkpoint provenance differs: ' + str(i))
    total = completion['sources_complete']
    if total < len(indexes) or total != len(completion['outputs']):
        raise RuntimeError('Inherited/new source counts differ from the sealed R4 snapshot')
    return dict(status='UNCHANGED_R3_CACHE_REUSED', inputs=proof, original_output=str(prior),
        source_indexes=indexes, inherited_sources=len(indexes), total_cached_sources=total,
        newly_scored_sources=total - len(indexes), original_files_preserved=True,
        registration_sha256=sha(reg_path), snapshot_sha256=sha(snapshot_path),
        original_metric_and_parity_values_changed=False)


def main(root, *, registration=None, completion=None, controller_pid=None, controller_ticks=None):
    root = Path(root).resolve(); out = root / 'outputs/METRIC-CONCURRENT-R4-20261002'
    runtime = out / 'runtime'; canonical = root / 'experiments/metric-concurrent-20261002'
    result = root / 'results/metric_concurrent_20261002'
    registration = Path(registration) if registration else out / 'concurrent_registration.json'
    completion = Path(completion) if completion else out / 'partial_completion.json'
    if not registration.is_absolute(): registration = root / registration
    if not completion.is_absolute(): completion = root / completion
    processes = process_snapshot(); parent = parent_gate(root, processes)
    bound, concurrent = concurrent_gate(out, runtime, registration, completion, processes,
                                        controller_pid, controller_ticks)
    predecessors = predecessor_gate(root, read(out / 'runtime_registration.json'), processes)
    inherited = inherited_cache_gate(root, read(registration), read(completion))
    if controller_pid is not None:
        gpu_output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True)
        if controller_pid in {int(x.strip()) for x in gpu_output.splitlines() if x.strip().isdigit()}:
            raise RuntimeError('The publication guardian must be CPU-only')
    gate_inputs = {**parent['inputs'], **concurrent['inputs'], **predecessors['inputs'], **inherited['inputs']}
    out.mkdir(parents=True, exist_ok=True)

    def command(args, log=None, capture=False):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        proc = subprocess.run([str(x) for x in args], cwd=root, env=env,
            stdout=subprocess.PIPE if capture else log,
            stderr=subprocess.PIPE if capture else subprocess.STDOUT)
        if log: log.flush()
        if proc.returncode:
            raise RuntimeError('Publication command failed: ' + repr(args))
        return proc.stdout if capture else b''

    def git(*args): return command(['git', *args], capture=True).decode().strip()
    def names(*args): return set(filter(None, command(['git', *args, '-z'], capture=True).decode().split('\0')))

    if (Path(git('rev-parse', '--show-toplevel')).resolve() != root
            or git('branch', '--show-current') != 'main' or git('remote', 'get-url', 'origin') not in ORIGINS):
        raise RuntimeError('The original main worktree and configured repository are required')
    receipt_path = out / 'extension_source_publication.json'
    previous = read(receipt_path) if receipt_path.exists() else None
    with (out / 'publication_checks.log').open('a', encoding='utf-8') as log:
        if previous:
            if previous.get('runtime_source_bindings') != bound or previous.get('checks') != 'PASS':
                raise RuntimeError('Previous runtime publication differs')
            verify(previous['published_files'], 'previous published files'); verify(previous['inputs'], 'publication inputs')
            command(['git', 'fetch', 'origin'], log)
            command(['git', 'cat-file', '-e', previous['commit'] + '^{commit}'], log)
            if previous['status'] == 'PUSHED':
                published(previous)
                command(['git', 'merge-base', '--is-ancestor', previous['commit'], 'origin/main'], log)
                return previous
            if previous['status'] != 'COMMITTED' or git('rev-parse', 'HEAD') != previous['commit']:
                raise RuntimeError('Cannot retry a different pending commit')
            if names('diff', '--name-only') or names('diff', '--cached', '--name-only'):
                raise RuntimeError('Tracked inputs changed after the checked commit')
            for p, h in previous['published_files'].items():
                if hashlib.sha256(command(['git', 'show', 'HEAD:' + Path(p).relative_to(root).as_posix()], capture=True)).hexdigest() != h:
                    raise RuntimeError('Committed bytes differ from checked source')
            command(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'], log)
            record = previous
        else:
            files = source_files(runtime)
            relative = [p.relative_to(runtime) for p in files]
            targets = [canonical / p for p in relative]
            if canonical.exists() and {p.relative_to(canonical) for p in source_files(canonical)} - set(relative):
                raise RuntimeError('Canonical source contains unrelated files')
            for src, target in zip(files, targets):
                if target.exists() and sha(target) != sha(src):
                    raise RuntimeError('Existing canonical source differs: ' + str(target))
            intended = {p.relative_to(root).as_posix() for p in targets}
            provenance = result / 'runtime_provenance.json'
            intended.add(provenance.relative_to(root).as_posix())
            allowed = intended | {'release_manifest.json'}
            dirty = names('diff', '--name-only') | names('diff', '--cached', '--name-only')
            pending_path = out / 'publication_checks_pending.json'
            pending = read(pending_path) if pending_path.exists() else None
            if dirty - intended:
                if dirty - intended != {'release_manifest.json'} or not pending:
                    raise RuntimeError('Unrelated tracked edits must be preserved')
                if (pending.get('runtime_source_bindings') != bound or pending.get('inputs') != gate_inputs
                        or pending.get('base_commit') != git('rev-parse', 'HEAD')
                        or pending.get('manifest_sha256') != sha(root / 'release_manifest.json')
                        or pending.get('manifest_sha256') != hashlib.sha256(command(['git', 'show', ':release_manifest.json'], capture=True)).hexdigest()):
                    raise RuntimeError('Dirty manifest is not this unchanged pending publication')
                verify(pending['published_files'], 'pending publication')
            for src, target in zip(files, targets):
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, target)
            write(provenance, dict(schema_version=1, runtime_source_bindings=bound,
                canonical_source_bindings=bindings(targets), parent_gate=parent, concurrent_gate=concurrent,
                predecessor_gate=predecessors, inherited_cache=inherited,
                byte_preserving_copy=True, source_hot_edits=False, training_updates=0,
                policy_selection_updates=0, scientific_result=False,
                qualification_claim='Repository CPU checks only; replay/model qualification remains in its original receipts'))
            if provenance.stat().st_size > MAX_RECEIPT_BYTES:
                raise RuntimeError('Runtime provenance exceeds the lightweight publication limit')
            checked = bindings([*targets, provenance]); verify(bound, 'runtime bytes')
            if names('diff', '--cached', '--name-only') - allowed:
                raise RuntimeError('Unrelated staged paths must be preserved')
            command(['git', 'add', '--', *sorted(intended)], log)
            command([sys.executable, 'tools/update_repository_manifest.py'], log)
            command(['git', 'add', '--', 'release_manifest.json'], log)
            manifest_sha = sha(root / 'release_manifest.json')
            pending = dict(runtime_source_bindings=bound, inputs=gate_inputs, base_commit=git('rev-parse', 'HEAD'),
                published_files=checked, manifest_sha256=manifest_sha)
            write(pending_path, pending)
            command([sys.executable, 'tools/verify_repository.py'], log)
            command([sys.executable, 'tools/run_cpu_checks.py'], log)
            tests = sorted(canonical.glob('test_*.py'))
            if not tests: raise RuntimeError('Relevant extension CPU tests are missing')
            for test in tests: command([sys.executable, str(test)], log)
            verify(bound, 'runtime bytes'); verify(gate_inputs, 'completion gates'); verify(checked, 'checked source')
            # Completion files and process exits must remain valid until staging finishes.
            parent_gate(root, process_snapshot())
            concurrent_gate(out, runtime, registration, completion, process_snapshot(), controller_pid, controller_ticks)
            predecessor_gate(root, read(out / 'runtime_registration.json'), process_snapshot())
            inherited_cache_gate(root, read(registration), read(completion))
            if (names('diff', '--cached', '--name-only') - allowed or names('diff', '--name-only')
                    or sha(root / 'release_manifest.json') != manifest_sha):
                raise RuntimeError('Publication scope changed during checks')
            for p, h in {**checked, str(root / 'release_manifest.json'): manifest_sha}.items():
                if hashlib.sha256(command(['git', 'show', ':' + Path(p).relative_to(root).as_posix()], capture=True)).hexdigest() != h:
                    raise RuntimeError('Staged bytes differ from checked inputs')
            command(['git', 'fetch', 'origin'], log)
            command(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'], log)
            command(['git', 'merge-base', '--is-ancestor', parent['parent_commit'], 'origin/main'], log)
            if names('diff', '--cached', '--name-only'):
                message = out / 'commit_message.txt'
                message.write_text('Register concurrent metric scoring cache reuse with verified frozen replay\n', encoding='utf-8')
                command(['git', 'commit', '-F', str(message)], log)
            record = dict(status='COMMITTED', checks='PASS', commit=git('rev-parse', 'HEAD'),
                source_bindings=bindings(targets), runtime_source_bindings=bound,
                published_files=checked, inputs=gate_inputs, parent_gate=parent,
                predecessor_gate=predecessors, inherited_cache=inherited,
                canonical_source=str(canonical), runtime_bundle=str(runtime), byte_preserving_copy=True,
                training_updates=0, policy_selection_updates=0, scientific_result=False, time=time.time())
            write(receipt_path, record)
        command(['git', 'push', 'origin', 'main'], log)
        remote = git('ls-remote', 'origin', 'refs/heads/main').split()
        if len(remote) != 2 or remote[0] != record['commit']:
            raise RuntimeError('Normal push did not verify the checked commit')
        record.update(status='PUSHED', remote_commit=remote[0], verified_time=time.time())
        write(receipt_path, record)
        return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--registration')
    p.add_argument('--completion')
    p.add_argument('--controller-pid', type=int)
    p.add_argument('--controller-start-ticks')
    a = p.parse_args()
    if (a.controller_pid is None) != (a.controller_start_ticks is None):
        p.error('Both controller identity arguments are required together')
    main(a.root, registration=a.registration, completion=a.completion,
         controller_pid=a.controller_pid, controller_ticks=a.controller_start_ticks)
