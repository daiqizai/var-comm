"""Checked normal publication of historical evaluation source and results.

Only this bundle's canonical source, lightweight historical results/report and
the repository release manifest can be staged. Oversized CSVs use the original
byte-preserving archive helper; its main/publication entry is never called.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from history_common import read, write, sha, verify, source_bindings
from history_inheritance import study_paths, origin_queue
from history_controller import (NAME, parent_gate, published, validate_queue, lock,
                                process_snapshot, gpu_pids, load_module, NotReady)

LIMIT = 8_000_000
ORIGINS = {'git@github.com:daiqizai/var-comm.git', 'https://github.com/daiqizai/var-comm.git'}


def copy_exact(source, target):
    source, target = Path(source), Path(target)
    if source.is_symlink() or target.is_symlink():
        raise RuntimeError('Publication symlinks are not allowed')
    if target.exists():
        if sha(source) != sha(target):
            raise RuntimeError('Existing publication bytes differ: '+str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def bind(paths):
    return {str(p.resolve()): sha(p) for p in sorted(map(Path, paths))}


def no_other_writers(processes=None):
    processes = process_snapshot() if processes is None else processes
    markers = ('/history_publish.py', '/numeric_delivery.py', '/publish_when_complete.py',
               '/publish.py', '/publish_extension.py', '/publish_source.py')
    for pid, item in processes.items():
        if pid == os.getpid() or item.get('state') == 'Z':
            continue
        command = item.get('command', '').replace('\\', '/')
        if (any(x in command for x in markers)
                or re.search(r'(?:^|\s)(?:\S*/)?git\s+(add|commit|push|reset|checkout|merge|rebase)(?:\s|$)', command)):
            raise NotReady('Another publication process is active: '+str(pid))


def archive_helper(root, queue):
    path = Path(root)/'experiments/unified-metrics-20261002/publish.py'
    expected = queue['shared_metric_bindings'].get(str(path.resolve()))
    if expected != sha(path):
        raise RuntimeError('Original archive helper hash differs')
    return load_module(path, '_historical_csv_archive_helper')


def selected_result_files(result, archives, analysis_outputs):
    """Allow only measured, sealed figures in addition to lightweight tables."""
    result=Path(result); omitted=set(archives['tables']); files=[]
    for path in sorted(result.rglob('*')):
        if not path.is_file() or path.relative_to(result).as_posix() in omitted:
            continue
        if path.is_symlink() or path.stat().st_size>LIMIT:
            raise RuntimeError('Unexpected or oversized selected result artifact: '+str(path))
        if path.suffix in ('.png','.pdf'):
            if (path.parent != result/'figures' or analysis_outputs.get(str(path)) != sha(path)):
                raise RuntimeError('Unsealed or misplaced scientific figure')
            with path.open('rb') as stream:
                magic=stream.read(8)
            if not (path.suffix=='.png' and magic==b'\x89PNG\r\n\x1a\n' or path.suffix=='.pdf' and magic.startswith(b'%PDF-')):
                raise RuntimeError('Scientific figure format differs')
        elif path.suffix not in {'.csv','.json','.md','.svg','.txt'}:
            raise RuntimeError('Unexpected selected result format: '+str(path))
        files.append(path)
    return files


def result_evidence(root, queue_path, queue):
    root = Path(root); out = root/'outputs'/NAME; result = root/'results/historical_metrics_r3_20261003'
    source = read(out/'source_publication.json'); published(source)
    if source.get('runtime_source_bindings') != queue['source_bindings']:
        raise RuntimeError('Results were not produced by the published runtime')
    verify(source['source_bindings']); verify(source['published_files']); verify(source['inputs'])
    inputs = {str(out/'source_publication.json'): sha(out/'source_publication.json')}
    copies = []
    model_id = None
    for job in queue['jobs']:
        study = job['study']
        folder, study_result = study_paths(root, queue, study)
        original_queue_path, original_queue = origin_queue(root, queue, queue_path, study)
        p = folder/'completion.json'; done = read(p)
        if (done.get('status') != 'HISTORICAL_STUDY_METRICS_COMPLETE' or done.get('study') != study
                or done.get('parity_passed') is not True or done.get('synthetic') is not False
                or done.get('training_updates') != 0 or done.get('policy_selection_updates') != 0
                or done.get('source_bindings') != original_queue['source_bindings']):
            raise RuntimeError('Historical result completion/parity/scope differs: '+study)
        verify(done['inputs']); verify(done['outputs'])
        regpath = study_result/'registration.json'; reg = read(regpath)
        if (reg.get('queue_registration_sha256') != sha(original_queue_path)
                or reg.get('source_bindings') != original_queue['source_bindings']
                or reg.get('source_count') != done.get('sources') or reg.get('frame_count') != done.get('frames')):
            raise RuntimeError('Historical score registration differs: '+study)
        if model_id is None:
            model_id = reg['evaluator_identity']
        if model_id != reg['evaluator_identity']:
            raise RuntimeError('Historical studies used different metric models')
        qpath = folder/'first_source_qualification.json'; qualification = read(qpath)
        if (qualification.get('status') != 'REAL_FIRST_SOURCE_PARITY_PASS'
                or qualification.get('source_index') != 0 or qualification.get('parity_passed') is not True
                or qualification.get('synthetic') is not False
                or qualification.get('registration_sha256') != sha(regpath)
                or qualification.get('original_values_preserved') is not True
                or sha(qualification['checkpoint']) != qualification['checkpoint_sha256']):
            raise RuntimeError('Missing real first-source qualification: '+study)
        batch = folder/'metric_batch_qualification.json'
        if sha(batch) != reg['batch_qualification_sha256']:
            raise RuntimeError('Historical metric batch qualification changed')
        for path in (p, qpath, regpath, batch):
            inputs[str(path)] = sha(path)
        inputs.update(done['inputs']); inputs.update(done['outputs'])
        proof_folder = result/'provenance'/('inherited_R2' if study in queue.get('inherited_studies', {}) else 'current_R3')/study
        copies += [(p, proof_folder/'completion.json'),
                    (qpath, proof_folder/'first_source_qualification.json'),
                    (batch, proof_folder/'metric_batch_qualification.json')]
        if study in queue.get('inherited_studies', {}):
            # Delivery copies retain exact R2 bytes and namespaced provenance.
            # They are never treated as R3 score registrations/completions.
            for name in ('metrics_per_frame.csv', 'source_baseline.csv', 'registration.json', 'model_metadata.json'):
                original = study_result/name
                if done['outputs'].get(str(original)) != sha(original):
                    raise RuntimeError('Inherited delivery artifact is not bound to R2 completion')
                copies.append((original, result/'inherited_R2'/study/name))
    analysis_path = result/'analysis_completion.json'; analysis = read(analysis_path)
    if analysis.get('status') != 'COMPLETE' or analysis.get('synthetic') is not False:
        raise RuntimeError('Historical paired analysis is incomplete')
    verify(analysis['inputs']); verify(analysis['outputs'])
    if analysis['inputs'].get(str(queue_path)) != sha(queue_path):
        raise RuntimeError('Historical analysis uses another queue')
    inputs.update(analysis['inputs']); inputs.update(analysis['outputs'])
    inputs[str(analysis_path)] = sha(analysis_path)
    for src, dst in copies:
        copy_exact(src, dst)
    return inputs


def prepare_files(root, runtime, queue_path, queue, phase, gate):
    root, runtime = Path(root), Path(runtime)
    out = root/'outputs'/NAME; result = root/'results/historical_metrics_r3_20261003'
    canonical = root/'experiments/historical-metrics-r3-20261003'
    sources = [Path(p) for p in queue['source_bindings']]
    names = {p.name for p in sources}
    if any(p.parent != runtime or p.is_symlink() or p.suffix not in ('.py', '.md') for p in sources):
        raise RuntimeError('Only flat reviewed Python/Markdown runtime may be published')
    if canonical.exists() and set(source_bindings(canonical)) - {str(canonical/n) for n in names}:
        raise RuntimeError('Canonical historical source contains unrelated files')
    targets = [canonical/p.name for p in sources]
    for src, dst in zip(sources, targets):
        copy_exact(src, dst)
    proof = result/'provenance'; proof.mkdir(parents=True, exist_ok=True)
    copies = [(queue_path, proof/'queue_registration.json'),
              (Path(queue['coverage_manifest']['path']), proof/'coverage_manifest.json'),
              (out/'cpu_checks.json', proof/'cpu_checks.json')]
    for study, item in queue.get('inherited_studies', {}).items():
        copies.append((Path(item['receipt_path']), proof/'inherited_R2'/study/'admission.json'))
    for src, dst in copies:
        copy_exact(src, dst)
    parent = proof/'parent_completion_gate.json'
    if parent.exists() and read(parent) != gate:
        raise RuntimeError('Original completion proof changed')
    write(parent, gate)
    inputs = {**gate['inputs'], **queue['source_bindings'], **queue['shared_metric_bindings'],
              **queue['input_proof_bindings'], str(queue_path): sha(queue_path),
              queue['coverage_manifest']['path']: queue['coverage_manifest']['sha256'],
              str(out/'cpu_checks.json'): sha(out/'cpu_checks.json')}
    ownproof = [dst for _, dst in copies]+[parent]
    if phase == 'source':
        return targets+ownproof, inputs, bind(targets)
    inputs.update(result_evidence(root, queue_path, queue))
    helper = archive_helper(root, queue)
    archive = helper.archive_tables(result=result, limit=LIMIT)
    # Only prose in the shard indexes changes; original table hashes/bytes stay.
    for entry in archive['tables'].values():
        index = result/entry['index']
        index.write_text(index.read_text().replace('publish.py --restore-tables',
            'history_publish.py --root ROOT --restore-tables'), encoding='utf-8')
    report = root/'reports/historical_metrics_r3_20261003.md'
    if not report.is_file() or report.stat().st_size > LIMIT:
        raise RuntimeError('Missing lightweight historical report')
    files = selected_result_files(result, archive, read(result/'analysis_completion.json')['outputs'])
    return targets+files+[report], inputs, bind(targets)


class Repository:
    def __init__(self, root, log):
        self.root, self.log = Path(root), log

    def command(self, args, *, capture=False):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        process = subprocess.run(list(map(str, args)), cwd=self.root, env=env,
            stdout=subprocess.PIPE if capture else self.log,
            stderr=subprocess.PIPE if capture else subprocess.STDOUT)
        self.log.flush()
        if process.returncode:
            raise RuntimeError('Checked publication command failed: '+repr(args))
        return process.stdout if capture else b''

    def git(self, *args):
        return self.command(['git', *args], capture=True).decode().strip()

    def names(self, *args):
        return set(filter(None, self.command(['git', *args, '-z'], capture=True).decode().split('\0')))


def publish(root, queue_path, phase, runtime=None):
    root = Path(root).resolve(); runtime = Path(runtime or Path(__file__).parent).resolve()
    queue_path = Path(queue_path).resolve(); queue = validate_queue(root, runtime, queue_path)
    if phase not in ('source', 'results'):
        raise ValueError(phase)
    out = root/'outputs'/NAME; out.mkdir(parents=True, exist_ok=True)
    own = lock(out/'publication.lock'); repo_lock = None
    try:
        gate = parent_gate(root)
        if gpu_pids():
            raise NotReady('Publication waits for all GPU computation owners to exit')
        no_other_writers()
        cpu = read(out/'cpu_checks.json')
        if cpu.get('status') != 'PASS' or cpu.get('source_bindings') != queue['source_bindings']:
            raise RuntimeError('Frozen bundle CPU checks are missing')
        repo_lock = lock(root/'.git/m2-recovery-publication.lock')
        with (out/(phase+'_publication_checks.log')).open('a', encoding='utf-8') as log:
            repo = Repository(root, log)
            if (Path(repo.git('rev-parse', '--show-toplevel')).resolve() != root
                    or repo.git('branch', '--show-current') != 'main'
                    or repo.git('remote', 'get-url', 'origin') not in ORIGINS):
                raise RuntimeError('Authorized single main repository required')
            repo.command(['git', 'fetch', 'origin'])
            for commit in gate['commits']:
                repo.command(['git', 'cat-file', '-e', commit+'^{commit}'])
                repo.command(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'])
            path = out/(phase+'_publication.json')
            previous = read(path) if path.exists() else None
            if previous:
                if previous.get('runtime_source_bindings') != queue['source_bindings'] or previous.get('checks') != 'PASS':
                    raise RuntimeError('Previous historical publication differs')
                verify(previous['published_files']); verify(previous['inputs'])
                if previous['status'] == 'PUSHED':
                    published(previous)
                    repo.command(['git', 'merge-base', '--is-ancestor', previous['commit'], 'origin/main'])
                    return previous
                if previous['status'] != 'COMMITTED' or repo.git('rev-parse', 'HEAD') != previous['commit']:
                    raise RuntimeError('Cannot retry a different checked historical commit')
                if repo.names('diff', '--name-only') or repo.names('diff', '--cached', '--name-only'):
                    raise RuntimeError('Tracked changes after checked commit must be preserved')
                for p, digest in previous['published_files'].items():
                    if hashlib.sha256(repo.command(['git', 'show', 'HEAD:'+Path(p).relative_to(root).as_posix()], capture=True)).hexdigest() != digest:
                        raise RuntimeError('Checked commit bytes changed')
                record = previous
            else:
                prefixes = ['experiments/historical-metrics-r3-20261003/', 'results/historical_metrics_r3_20261003/']
                dirty = repo.names('diff', '--name-only') | repo.names('diff', '--cached', '--name-only')
                pending_path = out/(phase+'_publication_pending.json')
                pending = read(pending_path) if pending_path.exists() else None
                unrelated = {p for p in dirty if not any(p.startswith(prefix) for prefix in prefixes)
                             and p != 'reports/historical_metrics_r3_20261003.md'}
                if unrelated:
                    if (unrelated != {'release_manifest.json'} or not pending
                            or pending.get('queue_registration_sha256') != sha(queue_path)
                            or pending.get('base_commit') != repo.git('rev-parse', 'HEAD')
                            or pending.get('manifest_sha256') != sha(root/'release_manifest.json')):
                        raise RuntimeError('Unrelated tracked edits must be preserved')
                    verify(pending['published_files']); verify(pending['inputs'])
                files, inputs, canonical = prepare_files(root, runtime, queue_path, queue, phase, gate)
                if any(p.is_symlink() or p.stat().st_size > LIMIT for p in files):
                    raise RuntimeError('Unexpected binary/symlink/oversized publication artifact')
                intended = {p.relative_to(root).as_posix() for p in files}
                allowed = intended|{'release_manifest.json'}
                if repo.names('diff', '--cached', '--name-only')-allowed:
                    raise RuntimeError('Unrelated staged paths must be preserved')
                # An oversized original CSV must never already be tracked.
                if any((root/p).suffix == '.csv' and (root/p).exists() and (root/p).stat().st_size > LIMIT
                       for p in repo.names('ls-files') if p.startswith('results/historical_metrics_r3_20261003/')):
                    raise RuntimeError('Oversized CSV is already tracked; refusing automatic removal')
                checked = bind(files)
                for start in range(0, len(files), 100):
                    repo.command(['git', 'add', '--', *[p.relative_to(root).as_posix() for p in files[start:start+100]]])
                repo.command([sys.executable, 'tools/update_repository_manifest.py'])
                repo.command(['git', 'add', '--', 'release_manifest.json'])
                manifest_sha = sha(root/'release_manifest.json')
                write(pending_path, dict(base_commit=repo.git('rev-parse', 'HEAD'),
                    queue_registration_sha256=sha(queue_path), manifest_sha256=manifest_sha,
                    published_files=checked, inputs=inputs))
                for script in ('tools/verify_repository.py', 'tools/run_cpu_checks.py'):
                    repo.command([sys.executable, script])
                for test in sorted((root/'experiments/historical-metrics-r3-20261003').glob('test_*.py')):
                    repo.command([sys.executable, test])
                verify(checked); verify(inputs)
                if parent_gate(root) != gate or gpu_pids():
                    raise RuntimeError('Parent completion or GPU ownership changed during publication')
                if repo.names('diff', '--cached', '--name-only')-allowed or repo.names('diff', '--name-only'):
                    raise RuntimeError('Repository changed during publication checks')
                if sha(root/'release_manifest.json') != manifest_sha:
                    raise RuntimeError('Release manifest changed during checks')
                for p, digest in {**checked, str(root/'release_manifest.json'): manifest_sha}.items():
                    staged = repo.command(['git', 'show', ':'+Path(p).relative_to(root).as_posix()], capture=True)
                    if hashlib.sha256(staged).hexdigest() != digest:
                        raise RuntimeError('Staged bytes differ from checked publication')
                repo.command(['git', 'fetch', 'origin'])
                repo.command(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'])
                no_other_writers()
                if repo.names('diff', '--cached', '--name-only'):
                    message = out/(phase+'_commit_message.txt')
                    message.write_text(('Register completed-method historical metric replay' if phase == 'source' else
                        'Publish independent metrics for completed historical communication methods')+'\n', encoding='utf-8')
                    repo.command(['git', 'commit', '-F', message])
                record = dict(status='COMMITTED', phase=phase, checks='PASS', commit=repo.git('rev-parse', 'HEAD'),
                    runtime_source_bindings=queue['source_bindings'], source_bindings=canonical,
                    published_files=checked, inputs=inputs, queue_registration_sha256=sha(queue_path), time=time.time())
                write(path, record)
            repo.command(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'])
            no_other_writers(); repo.command(['git', 'push', 'origin', 'main'])
            remote = repo.git('ls-remote', 'origin', 'refs/heads/main').split()
            if len(remote) != 2 or remote[0] != record['commit']:
                raise RuntimeError('Normal publication push did not verify exact remote SHA')
            record.update(status='PUSHED', remote_commit=remote[0], verified_time=time.time()); write(path, record)
            return record
    except BaseException as error:
        write(out/(phase+'_publication_failure.json'), dict(status='FAILED', error=str(error), time=time.time()))
        raise
    finally:
        if repo_lock is not None:
            repo_lock.close()
        own.close()


def restore(root):
    root = Path(root).resolve(); result = root/'results/historical_metrics_r3_20261003'
    queue = read(result/'provenance/queue_registration.json')
    helper = root/'experiments/unified-metrics-20261002/publish.py'
    matches = [digest for path, digest in queue['shared_metric_bindings'].items()
               if path.replace('\\', '/').endswith('/experiments/unified-metrics-20261002/publish.py')]
    if matches != [sha(helper)]:
        raise RuntimeError('Published original archive implementation differs')
    return load_module(helper, '_historical_csv_restore').restore_tables(result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--queue-registration')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--phase', choices=('source', 'results')); group.add_argument('--restore-tables', action='store_true')
    args = parser.parse_args()
    if args.restore_tables:
        restore(args.root)
    elif not args.queue_registration:
        parser.error('--queue-registration is required for publication')
    else:
        publish(args.root, args.queue_registration, args.phase)


if __name__ == '__main__':
    main()
