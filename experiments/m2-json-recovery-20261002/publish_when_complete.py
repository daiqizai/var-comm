"""Publish immutable M2 JSON recovery evidence after all authorized work exits.

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




class NotReady(RuntimeError):
    """The original scientific/publication owner has not finished yet."""


NAME = 'M2-JSON-RECOVERY-20261002'
PARENT = 'SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
METRICS = 'UNIFIED-METRICS-20261002'
FINAL = 'METRIC-FINAL-CACHE-20261002'
R4 = 'METRIC-CONCURRENT-R4-20261002'
ARTIFACT_SUFFIXES = {'.json', '.md', '.csv', '.txt', '.log'}


def read_ready(path):
    if not path.is_file(): raise NotReady('Waiting for ' + str(path))
    return read(path)


def no_publishers(processes, own_pid=None):
    own_pid = os.getpid() if own_pid is None else own_pid
    markers = ('/publish.py', '/publish_extension.py', '/publish_source.py', '/publish_when_complete.py')
    blocked = [pid for pid, item in processes.items() if pid != own_pid and item.get('state') != 'Z'
               and (any(m in item.get('command', '').replace('\\', '/') for m in markers)
                    or re.search(r'(^|\s)git\s+(add|commit|push|reset|checkout|merge|rebase)(\s|$)', item.get('command', '')))]
    if blocked: raise NotReady('Other repository publishers/writers are active: ' + str(blocked))


def completion_gate(root, processes):
    parent, metric, final, r4 = [root / 'outputs' / n for n in (PARENT, METRICS, FINAL, R4)]
    pc, mc = read_ready(parent / 'completion.json'), read_ready(metric / 'completion.json')
    if pc.get('status') != 'AUTHORIZED_TWO_METHODS_COMPLETE': raise NotReady('Waiting for complete M2 delivery')
    if mc.get('status') != 'UNIFIED_METRICS_COMPLETE': raise NotReady('Waiting for complete unified metric delivery')
    for name, record in (('M2', pc), ('metrics', mc)):
        if (record.get('synthetic') is not False or record.get('training_updates') != 0 or record.get('stop') is not True):
            raise RuntimeError('Unexpected final ' + name + ' scientific scope')
        published(record.get('publication', {}))
        pub = record['publication']; verify(pub.get('source_bindings'), name + ' published source')
        for field in ('published_files', 'inputs'):
            if field in pub: verify(pub[field], name + ' publication ' + field)
    paths = [parent / 'completion.json', metric / 'completion.json']
    for folder in (parent, metric):
        state, launch = read_ready(folder / 'supervisor_status.json'), read_ready(folder / 'supervisor_launch.json')
        if state.get('status') != 'COMPLETE' or state.get('stopped_after_authorized_scope') is not True:
            raise NotReady('A scientific/metric supervisor has not completed its scope')
        if state.get('pid') != launch.get('pid'): raise RuntimeError('Final supervisor identity differs')
        try: require_exited(launch, processes, folder.name + ' supervisor')
        except RuntimeError as e:
            if 'has not exited' in str(e): raise NotReady(str(e)) from e
            raise
        paths += [folder / 'supervisor_status.json', folder / 'supervisor_launch.json']
    fc = read_ready(final / 'controller_completion.json'); fl = read_ready(final / 'controller_launch.json')
    if (fc.get('status') != 'COMPLETE' or fc.get('training_updates') != 0 or fc.get('policy_selection_updates') != 0
            or fc.get('stopped_after_authorized_scope') is not True
            or fc.get('metrics_completion_sha256') != sha(metric / 'completion.json')
            or fc.get('pid') != fl.get('pid') or str(fc.get('start_ticks')) != str(fl.get('start_ticks'))):
        raise RuntimeError('Final cache guardian completion differs')
    verify(fc.get('source_bindings'), 'final cache guardian source')
    try: require_exited(fc, processes, 'final cache guardian')
    except RuntimeError as e:
        if 'has not exited' in str(e): raise NotReady(str(e)) from e
        raise
    paths += [final / 'controller_completion.json', final / 'controller_launch.json']
    for name in ('m2_controller_launch.json', 'controller_launch.json', 'partial_launch.json'):
        path = r4 / name; launch = read_ready(path)
        try: require_exited(launch, processes, 'R4 ' + name)
        except RuntimeError as e:
            if 'has not exited' in str(e): raise NotReady(str(e)) from e
            raise
        paths.append(path)
    markers = ('/experiments/scale-causal-partial-residual-20261002/', '/experiments/unified-metrics-20261002/',
               '/experiments/metric-speed-20261002/', '/experiments/metric-final-cache-20261002/',
               '/outputs/METRIC-CONCURRENT-R4-20261002/runtime/', '/outputs/METRIC-FINAL-CACHE-20261002/runtime/')
    live = [pid for pid, item in processes.items() if item.get('state') != 'Z'
            and any(m in item.get('command', '').replace('\\', '/') for m in markers)]
    if live: raise NotReady('Original scientific or metric workers are still active: ' + str(live))
    no_publishers(processes)
    return dict(status='ALL_AUTHORIZED_RESULTS_PUBLISHED_AND_OWNERS_EXITED', inputs=bindings(paths),
                commits=[pc['publication']['commit'], mc['publication']['commit']], scientific_result=False)


def recovery_gate(root, runtime, manifest_path):
    record = read(manifest_path)
    if (record.get('status') != 'RECOVERY_READY_FOR_PUBLICATION'
            or record.get('training_updates') != 0 or record.get('policy_selection_updates') != 0
            or record.get('allowed_json_type') != 'numpy.bool_'
            or record.get('gate_derivation') != 'first_200_from_frozen_screen'):
        raise RuntimeError('The immutable recovery publication manifest has an unexpected scope')
    bound = bindings(source_files(runtime))
    if record.get('source_bindings') != bound: raise RuntimeError('The frozen recovery runtime inventory changed')
    verify(record.get('proof_bindings'), 'immutable recovery proof')
    artifacts = record.get('artifacts')
    verify(artifacts, 'reviewed recovery artifacts')
    for path in artifacts:
        p = Path(path)
        if (p.is_symlink() or p.suffix not in ARTIFACT_SUFFIXES or p.stat().st_size > MAX_RECEIPT_BYTES
                or not p.resolve().is_relative_to(root.resolve())):
            raise RuntimeError('Unexpected or oversized recovery proof artifact: ' + path)
    return dict(status='IMMUTABLE_RECOVERY_EVIDENCE_READY', source_bindings=bound,
                inputs={**bound, **record['proof_bindings'], **artifacts, str(manifest_path): sha(manifest_path)},
                artifacts=artifacts, scope=dict(allowed_json_type='numpy.bool_', gate_derivation='first_200_from_frozen_screen',
                training_updates=0,policy_selection_updates=0), scientific_result=False)


def report_text():
    return '\n'.join([
        '# M2 序列化恢复记录', '',
        '本记录在 M2 原结果和统一补充指标均完成、推送且全部原工作进程退出后发布。原实验源码、策略与已保存缓存保留原字节。', '',
        '恢复范围由固定来源与证明文件登记：', '',
        '- JSON 兼容层只接受精确的 `numpy.bool_` 类型并转换成 Python `bool`；其他未知类型仍由原序列化错误处理。',
        '- gate 首 200 源的记录从冻结 screen 缓存派生，保存原流程会产生的对应记录；不新增图像、噪声或模型训练。',
        '- 原失败证据和缓存保留。固定证明文件见 `proofs/`；逐进程操作回执继续保存在原输出目录。', '',
        '本次提交登记工程恢复过程。它不重新选择策略，也不提供新的科学结果或模型资格结论。所有指标仍以原实验和统一指标报告为准。', ''])


def acquire_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True); handle = path.open('a')
    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException: handle.close(); raise
    return handle


def main(root, *, runtime=None, manifest=None, poll_seconds=30, once=False):
    root = Path(root).resolve(); out = root / 'outputs' / NAME
    runtime = Path(runtime or Path(__file__).parent).resolve()
    manifest = Path(manifest or out / 'recovery_publication_manifest.json').resolve()
    out.mkdir(parents=True,exist_ok=True)
    own_lock = acquire_lock(out / 'publication.lock')
    repo_lock = None
    try:
        recovery = recovery_gate(root,runtime,manifest)
        while True:
            verify(recovery['inputs'],'frozen recovery inputs')
            try:
                completed = completion_gate(root,process_snapshot()); break
            except NotReady as e:
                write(out/'publication_status.json',dict(status='WAITING_FOR_ALL_FINAL_PUBLICATIONS_AND_EXIT',reason=str(e),pid=os.getpid(),time=time.time()))
                if once: return None
                time.sleep(poll_seconds)
        canonical = root/'experiments/m2-json-recovery-20261002'
        result = root/'results/m2_json_recovery_20261002'
        gate_inputs = {**completed['inputs'],**recovery['inputs']}
        repo_lock = acquire_lock(root/'.git/m2-recovery-publication.lock')
        def command(args,log=None,capture=False):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            proc=subprocess.run([str(x) for x in args],cwd=root,env=env,
                stdout=subprocess.PIPE if capture else log,stderr=subprocess.PIPE if capture else subprocess.STDOUT)
            if log: log.flush()
            if proc.returncode: raise RuntimeError('Recovery publication command failed: '+repr(args))
            return proc.stdout if capture else b''
        def git(*args):return command(['git',*args],capture=True).decode().strip()
        def names(*args):return set(filter(None,command(['git',*args,'-z'],capture=True).decode().split('\0')))
        if (Path(git('rev-parse','--show-toplevel')).resolve()!=root
                or git('branch','--show-current')!='main' or git('remote','get-url','origin') not in ORIGINS):
            raise RuntimeError('The authorized original main repository is required')
        receipt=out/'publication.json';previous=read(receipt) if receipt.exists() else None
        with (out/'publication_checks.log').open('a',encoding='utf-8') as log:
            no_publishers(process_snapshot());command(['git','fetch','origin'],log)
            for commit in completed['commits']:
                command(['git','cat-file','-e',commit+'^{commit}'],log)
                command(['git','merge-base','--is-ancestor',commit,'origin/main'],log)
            if previous:
                if previous.get('runtime_source_bindings')!=recovery['source_bindings'] or previous.get('checks')!='PASS':
                    raise RuntimeError('Prior recovery publication differs')
                verify(previous['published_files'],'prior recovery publication');verify(previous['inputs'],'prior publication gates')
                command(['git','cat-file','-e',previous['commit']+'^{commit}'],log)
                if previous.get('status')=='PUSHED':
                    published(previous);command(['git','merge-base','--is-ancestor',previous['commit'],'origin/main'],log)
                    return previous
                if previous.get('status')!='COMMITTED' or git('rev-parse','HEAD')!=previous['commit']:
                    raise RuntimeError('Cannot retry a different recovery commit')
                if names('diff','--name-only') or names('diff','--cached','--name-only'):
                    raise RuntimeError('Tracked files changed after the checked recovery commit')
                for path,digest in previous['published_files'].items():
                    if hashlib.sha256(command(['git','show','HEAD:'+Path(path).relative_to(root).as_posix()],capture=True)).hexdigest()!=digest:
                        raise RuntimeError('Recovery commit bytes differ from checked publication')
                command(['git','merge-base','--is-ancestor','origin/main','HEAD'],log);record=previous
            else:
                sources=source_files(runtime);targets=[canonical/p.relative_to(runtime) for p in sources]
                if canonical.exists() and {p.relative_to(canonical) for p in source_files(canonical)}-{p.relative_to(runtime) for p in sources}:
                    raise RuntimeError('Canonical recovery source contains unrelated files')
                proof_sources=[Path(p) for p in sorted(recovery['artifacts'])]
                proof_targets=[result/'proofs'/f'{i:03d}_{p.name}' for i,p in enumerate(proof_sources)]
                for source,target in zip(sources+proof_sources,targets+proof_targets):
                    if target.exists() and sha(target)!=sha(source): raise RuntimeError('Existing recovery publication bytes differ: '+str(target))
                report=result/'RECOVERY_REPORT.md';provenance=result/'provenance.json';manifest_copy=result/'recovery_publication_manifest.json'
                intended={p.relative_to(root).as_posix() for p in targets+proof_targets+[report,provenance,manifest_copy]}
                allowed=intended|{'release_manifest.json'}
                dirty=names('diff','--name-only')|names('diff','--cached','--name-only')
                pending_path=out/'publication_checks_pending.json';pending=read(pending_path) if pending_path.exists() else None
                if dirty-intended:
                    if dirty-intended!={'release_manifest.json'} or not pending:raise RuntimeError('Unrelated tracked edits must be preserved')
                    if (pending.get('runtime_source_bindings')!=recovery['source_bindings'] or pending.get('inputs')!=gate_inputs
                            or pending.get('base_commit')!=git('rev-parse','HEAD')
                            or pending.get('manifest_sha256')!=sha(root/'release_manifest.json')
                            or pending.get('manifest_sha256')!=hashlib.sha256(command(['git','show',':release_manifest.json'],capture=True)).hexdigest()):
                        raise RuntimeError('Dirty manifest is not this unchanged pending recovery publication')
                    verify(pending['published_files'],'pending recovery publication')
                for source,target in zip(sources+proof_sources,targets+proof_targets):
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                result.mkdir(parents=True,exist_ok=True);shutil.copyfile(manifest,manifest_copy)
                report.write_text(report_text(),encoding='utf-8')
                write(provenance,dict(schema_version=1,completion_gate=completed,recovery=recovery,
                    runtime_source_bindings=recovery['source_bindings'],canonical_source_bindings=bindings(targets),
                    proof_copies={str(src):dict(published_path=str(dst),sha256=sha(dst)) for src,dst in zip(proof_sources,proof_targets)},
                    original_failure_and_cache_preserved=True,byte_preserving_copy=True,
                    training_updates=0,policy_selection_updates=0,scientific_result=False))
                if provenance.stat().st_size>MAX_RECEIPT_BYTES:raise RuntimeError('Recovery provenance is too large')
                files=targets+proof_targets+[report,provenance,manifest_copy];checked=bindings(files)
                if names('diff','--cached','--name-only')-allowed:raise RuntimeError('Unrelated staged paths must be preserved')
                no_publishers(process_snapshot())
                command(['git','add','--',*sorted(intended)],log)
                command([sys.executable,'tools/update_repository_manifest.py'],log)
                command(['git','add','--','release_manifest.json'],log)
                manifest_sha=sha(root/'release_manifest.json')
                write(pending_path,dict(runtime_source_bindings=recovery['source_bindings'],inputs=gate_inputs,
                    base_commit=git('rev-parse','HEAD'),published_files=checked,manifest_sha256=manifest_sha))
                command([sys.executable,'tools/verify_repository.py'],log)
                command([sys.executable,'tools/run_cpu_checks.py'],log)
                tests=sorted(canonical.glob('test_*.py'))
                if not tests:raise RuntimeError('Recovery publisher CPU tests are missing')
                for test in tests:command([sys.executable,str(test)],log)
                verify(gate_inputs,'final recovery inputs');verify(checked,'checked recovery files')
                if completion_gate(root,process_snapshot())['inputs']!=completed['inputs']:
                    raise RuntimeError('Original final evidence changed during recovery publication')
                if (names('diff','--cached','--name-only')-allowed or names('diff','--name-only')
                        or sha(root/'release_manifest.json')!=manifest_sha):raise RuntimeError('Publication scope changed during checks')
                for path,digest in {**checked,str(root/'release_manifest.json'):manifest_sha}.items():
                    if hashlib.sha256(command(['git','show',':'+Path(path).relative_to(root).as_posix()],capture=True)).hexdigest()!=digest:
                        raise RuntimeError('Staged recovery bytes differ from checks')
                command(['git','fetch','origin'],log);command(['git','merge-base','--is-ancestor','origin/main','HEAD'],log)
                no_publishers(process_snapshot())
                if names('diff','--cached','--name-only'):
                    message=out/'commit_message.txt';message.write_text('Document exact M2 boolean serialization recovery and cached gate derivation\n',encoding='utf-8')
                    command(['git','commit','-F',message],log)
                record=dict(status='COMMITTED',checks='PASS',commit=git('rev-parse','HEAD'),
                    runtime_source_bindings=recovery['source_bindings'],source_bindings=bindings(targets),
                    published_files=checked,inputs=gate_inputs,completion_gate=completed,
                    training_updates=0,policy_selection_updates=0,scientific_result=False,time=time.time())
                write(receipt,record)
            no_publishers(process_snapshot());command(['git','push','origin','main'],log)
            remote=git('ls-remote','origin','refs/heads/main').split()
            if len(remote)!=2 or remote[0]!=record['commit']:raise RuntimeError('Normal recovery push did not verify its commit')
            record.update(status='PUSHED',remote_commit=remote[0],verified_time=time.time());write(receipt,record)
            write(out/'publication_status.json',dict(status='COMPLETE',publication=record,stop=True,pid=os.getpid()))
            return record
    except BaseException as e:
        write(out/'publication_failure.json',dict(status='FAILED',error=str(e),pid=os.getpid(),time=time.time()))
        raise
    finally:
        if repo_lock:repo_lock.close()
        own_lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--runtime');p.add_argument('--recovery-manifest');p.add_argument('--poll-seconds',type=int,default=30)
    p.add_argument('--once',action='store_true');a=p.parse_args()
    if not 1<=a.poll_seconds<=60:p.error('poll-seconds must be between 1 and 60')
    result=main(a.root,runtime=a.runtime,manifest=a.recovery_manifest,poll_seconds=a.poll_seconds,once=a.once)
    if result is None:sys.exit(75)
