"""Publish the separately frozen final cache assembler after scientific delivery.

This entry point performs CPU repository checks. It neither launches scoring
nor claims that CPU tests qualify reconstructed images or metric models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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



def load_r4_publisher(root):
    out = root / 'outputs/METRIC-CONCURRENT-R4-20261002'
    path = out / 'extension_source_publication.json'
    record = read(path); published(record)
    verify(record.get('runtime_source_bindings'), 'published R4 runtime')
    verify(record.get('published_files'), 'published R4 files')
    verify(record.get('inputs'), 'published R4 inputs')
    source = out / 'runtime/publish_extension.py'
    if record['runtime_source_bindings'].get(str(source)) != sha(source):
        raise RuntimeError('The original R4 publisher is not bound by its publication')
    spec = importlib.util.spec_from_file_location('_published_r4_publisher', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, record, path


def final_cache_gate(root, out, runtime, processes, controller_pid, controller_ticks):
    """Validate completed old science and the new CPU-only publication owner."""
    base, r4_publication, r4_publication_path = load_r4_publisher(root)
    oldout = root / 'outputs/METRIC-CONCURRENT-R4-20261002'
    oldruntime = oldout / 'runtime'
    oldreg, olddone = oldout / 'concurrent_registration.json', oldout / 'partial_completion.json'
    if read(olddone).get('sources_complete') != 100 or len(read(olddone).get('outputs', {})) != 100:
        raise RuntimeError('All 100 R4 source caches must be sealed before final assembly')
    parent = base.parent_gate(root, processes)
    oldbound, concurrent = base.concurrent_gate(oldout, oldruntime, oldreg, olddone, processes)
    previous = base.predecessor_gate(root, read(oldout / 'runtime_registration.json'), processes)
    inherited = base.inherited_cache_gate(root, read(oldreg), read(olddone))
    if oldbound != r4_publication['runtime_source_bindings']:
        raise RuntimeError('R4 publication and replay source inventory differ')
    bound = bindings(source_files(runtime))
    paths = [out / n for n in ('runtime_registration.json', 'handoff.json', 'controller_launch.json', 'r4_retirement.json')]
    reg, handoff, launch, retirement = map(read, paths)
    original = bindings(source_files(root / 'experiments/unified-metrics-20261002'))
    if (reg.get('status') != 'REGISTERED_FINAL_CACHE_GUARDIAN'
            or reg.get('runtime_source_bindings') != bound
            or reg.get('original_metric_source_bindings') != original
            or reg.get('training_updates') != 0 or reg.get('policy_selection_updates') != 0
            or reg.get('previous_runtime_registration_sha256') != sha(oldout / 'runtime_registration.json')
            or reg.get('handoff_sha256') != sha(paths[1])
            or handoff.get('new_source_bindings') != bound
            or handoff.get('previous_launch_sha256') != sha(oldout / 'controller_launch.json')
            or handoff.get('previous_runtime_registration_sha256') != sha(oldout / 'runtime_registration.json')
            or launch.get('status') != 'DETACHED_CPU_GUARDIAN'
            or launch.get('source_bindings') != bound
            or launch.get('runtime_registration_sha256') != sha(paths[0])):
        raise RuntimeError('Final cache guardian source/handoff registration differs')
    oldowner = read(oldout / 'controller_launch.json')
    def owner(value): return (value.get('pid'), str(value.get('start_ticks')))
    if (retirement.get('status') != 'R4_IDLE_METRIC_OWNER_RETIRED'
            or retirement.get('original_scientific_owner_untouched') is not True
            or owner(retirement.get('oldowner', {})) != owner(oldowner)
            or owner(handoff.get('previous_owner', {})) != owner(oldowner)):
        raise RuntimeError('The retired R4 owner differs from its recorded idle handoff')
    proof = retirement.get('proof_bindings'); verify(proof, 'R4 retirement proof')
    required = [oldout / n for n in ('concurrent_registration.json', 'partial_completion.json',
        'controller_launch.json', 'controller_status.json', 'runtime_registration.json', 'admission.json', 'partial_launch.json')]
    if not {str(p) for p in required}.issubset(proof):
        raise RuntimeError('The R4 retirement did not bind its full cache and worker evidence')
    oldstatus = read(oldout / 'controller_status.json')
    if oldstatus.get('status') != 'WAITING_FOR_ORIGINAL_COMPLETION' or oldstatus.get('worker_pid') is not None:
        raise RuntimeError('Only the idle fully scored R4 metric owner can be retired')
    require_exited(oldowner, processes, 'R4 metric owner')
    if (type(controller_pid) is not int or controller_pid <= 1
            or owner(launch) != (controller_pid, str(controller_ticks))):
        raise RuntimeError('The registered CPU publication guardian is required')
    actual = processes.get(controller_pid)
    if (not actual or actual.get('unreadable') or actual.get('state') == 'Z'
            or actual.get('start_ticks') != str(controller_ticks)
            or (runtime / 'controller.py').as_posix() not in actual.get('command', '').replace('\\', '/')):
        raise RuntimeError('The publication guardian process identity differs')
    history = Path(launch.get('history_path', '')).resolve()
    if (not history.is_relative_to((out / 'launch_history').resolve())
            or not history.is_file() or sha(history) != sha(paths[2])):
        raise RuntimeError('The guardian launch must have an unchanged immutable history copy')
    marker = runtime.as_posix() + '/'
    live = [pid for pid, item in processes.items() if pid not in (os.getpid(), controller_pid)
            and item.get('state') != 'Z' and marker in item.get('command', '').replace('\\', '/')]
    if live: raise RuntimeError('Unexpected final cache workers remain active: ' + str(live))
    inputs = {**parent['inputs'], **concurrent['inputs'], **previous['inputs'], **inherited['inputs'],
              **r4_publication['inputs'], **proof, **bindings([paths[0], paths[1], paths[3], history]),
              str(r4_publication_path): sha(r4_publication_path)}
    evidence = dict(status='COMPLETED_REAL_CACHE_AND_SCIENCE_READY', inputs=inputs,
        r4_publication_commit=r4_publication['commit'], cache_sources_complete=100,
        parent_gate=parent, r4_runtime_source_bindings=oldbound,
        guardian=launch, original_sources_unchanged=True, training_updates=0,
        policy_selection_updates=0, scientific_result=False,
        qualification_claim='Repository CPU checks only; real replay/metric evidence is preserved from R4')
    return bound, evidence


def main(root, *, controller_pid=None, controller_ticks=None):
    root = Path(root).resolve(); out = root / 'outputs/METRIC-FINAL-CACHE-20261002'
    runtime = out / 'runtime'; canonical = root / 'experiments/metric-final-cache-20261002'
    result = root / 'results/metric_final_cache_20261002'
    bound, evidence = final_cache_gate(root, out, runtime, process_snapshot(), controller_pid, controller_ticks)
    gpu_output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    if controller_pid in {int(x.strip()) for x in gpu_output.splitlines() if x.strip().isdigit()}:
        raise RuntimeError('The final publication guardian must be CPU-only')
    gate_inputs = evidence['inputs']
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
        command(['git', 'fetch', 'origin'], log)
        command(['git', 'cat-file', '-e', evidence['r4_publication_commit'] + '^{commit}'], log)
        command(['git', 'merge-base', '--is-ancestor', evidence['r4_publication_commit'], 'origin/main'], log)
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
                canonical_source_bindings=bindings(targets), cache_gate=evidence,
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
            _, again = final_cache_gate(root, out, runtime, process_snapshot(), controller_pid, controller_ticks)
            if again['inputs'] != gate_inputs: raise RuntimeError('Final cache publication evidence changed')
            if (names('diff', '--cached', '--name-only') - allowed or names('diff', '--name-only')
                    or sha(root / 'release_manifest.json') != manifest_sha):
                raise RuntimeError('Publication scope changed during checks')
            for p, h in {**checked, str(root / 'release_manifest.json'): manifest_sha}.items():
                if hashlib.sha256(command(['git', 'show', ':' + Path(p).relative_to(root).as_posix()], capture=True)).hexdigest() != h:
                    raise RuntimeError('Staged bytes differ from checked inputs')
            command(['git', 'fetch', 'origin'], log)
            command(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'], log)
            command(['git', 'merge-base', '--is-ancestor', evidence['parent_gate']['parent_commit'], 'origin/main'], log)
            if names('diff', '--cached', '--name-only'):
                message = out / 'commit_message.txt'
                message.write_text('Register verified cached metric assembly and fresh M2 replay\n', encoding='utf-8')
                command(['git', 'commit', '-F', str(message)], log)
            record = dict(status='COMMITTED', checks='PASS', commit=git('rev-parse', 'HEAD'),
                source_bindings=bindings(targets), runtime_source_bindings=bound,
                published_files=checked, inputs=gate_inputs, cache_gate=evidence,
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
    p.add_argument('--controller-pid', type=int, required=True)
    p.add_argument('--controller-start-ticks', required=True)
    a = p.parse_args()
    main(a.root, controller_pid=a.controller_pid, controller_ticks=a.controller_start_ticks)
