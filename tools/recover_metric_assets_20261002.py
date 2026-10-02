"""Bounded network recovery around unchanged metric preparation/supervision.

Keep this operational helper outside the frozen experiment source directory.
It never writes download parts, model files, or scientific completion receipts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def transient(error):
    """Whitelist the original downloader's transport errors, never integrity errors."""
    text=str(error).strip()
    if text=='Incomplete HTTP asset response':return True
    if re.fullmatch(r'HTTP Error (408|429|500|502|503|504):[^\r\n]*',text):return True
    if re.fullmatch(r'IncompleteRead\(\d+ bytes read(?:, \d+ more expected)?\)',text):return True
    # URLError wraps timeout, reset, DNS, and network availability exceptions.
    if text.startswith('<urlopen error ') and text.endswith('>'):text=text[15:-1]
    if text in ('timed out','Remote end closed connection without response'):return True
    return bool(re.fullmatch(r'(?:\[Errno (?:-3|32|101|103|104|110|111|113)\] )?'
        r'(?:Temporary failure in name resolution|Broken pipe|Network is unreachable|'
        r'Software caused connection abort|Connection aborted|Connection reset by peer|'
        r'Connection timed out|Connection refused|No route to host)',text))


def process_identity(pid,proc=Path('/proc')):
    try:
        fields=(Path(proc)/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=int(pid),start_ticks=fields[19],state=fields[0])
    except FileNotFoundError:return None


def named_processes(source,proc=Path('/proc')):
    scripts={str((Path(source)/name).resolve()):name for name in ('prepare_assets.py','supervisor.py')}
    found=[]
    for folder in Path(proc).iterdir():
        if not folder.name.isdecimal() or int(folder.name)==os.getpid():continue
        try:
            identity=process_identity(int(folder.name),proc)
            if identity is None or identity['state']=='Z':continue
            arguments=(folder/'cmdline').read_bytes().split(b'\0')
            for raw in arguments:
                arg=raw.decode(errors='replace')
                if not arg.endswith(('/prepare_assets.py','/supervisor.py')) and arg not in ('prepare_assets.py','supervisor.py'):continue
                path=Path(arg)
                if not path.is_absolute():path=Path(os.readlink(folder/'cwd'))/path
                name=scripts.get(str(path.resolve()))
                if name:
                    found.append(dict(identity,script=name));break
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            # Do not suppress inability to inspect this user's known processes.
            if folder.stat().st_uid==os.getuid():raise
    return found


def frozen_sources(source,out):
    expected=read(out/'supervisor_registration.json')['source_bindings']
    actual={str(p.resolve()):sha(p) for p in sorted(source.iterdir()) if p.suffix in ('.py','.md')}
    if not expected or actual!=expected:raise RuntimeError('Frozen metric source inventory differs')
    return expected


def ready_assets(out):
    path=out/'assets_complete.json'
    if not path.exists():return None
    value=read(path)
    if (value.get('status')!='ASSETS_READY' or value.get('training_updates')!=0 or
            value.get('manifest_sha256')!=sha(out/'modelmanifest.json') or
            value.get('environment_freeze_sha256')!=sha(out/'environment_freeze.txt') or
            not Path(value.get('environment','')).is_file()):
        raise RuntimeError('Existing asset completion/manifest/environment is invalid')
    return value


def require_retryable_failure(out,attempts):
    """A helper restart cannot turn an earlier integrity/program failure into a retry."""
    path=out/'assets_status.json'
    if not path.exists():raise RuntimeError('No original asset failure receipt to recover')
    failure=read(path)
    if failure.get('status')!='FAILED' or not transient(failure.get('error','')):
        raise RuntimeError('Only an explicit transient network failure may restart preparation: '+str(failure))
    if attempts:
        previous=attempts[-1]
        if failure.get('time',0)<previous['launch']['launched_at']:
            raise RuntimeError('Previous preparation failure receipt is stale')
        if previous.get('returncode')==0 or (isinstance(previous.get('returncode'),int) and previous['returncode']<0):
            raise RuntimeError('Previous preparation ended without a verified retryable exit')
        previous['failure']=failure
    return failure


def archive_receipts(out,folder):
    folder.mkdir(parents=True,exist_ok=False)
    entries={}
    # Small JSON evidence only; never copy or alter model/download contents.
    for path in sorted(out.glob('*.json')):
        target=folder/path.name;shutil.copyfile(path,target);entries[str(target)]=sha(target)
    return entries


def start(script,python,root,log):
    launched_at=time.time()
    with log.open('ab') as stream:
        child=subprocess.Popen([str(python),'-u',str(script)],cwd=root,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    identity=process_identity(child.pid)
    if identity is None:raise RuntimeError('Cannot record launched process identity')
    return child,dict(identity,script=str(script),python=str(python),log=str(log),launched_at=launched_at)


def recover(root,python,max_attempts=8,initial_delay=60.,max_delay=900.):
    import fcntl
    root=Path(root).resolve();python=Path(python).absolute()
    source=root/'experiments/unified-metrics-20261002'
    out=root/'outputs/UNIFIED-METRICS-20261002';work=out/'assets_recovery'
    if not python.is_file():raise RuntimeError('The original Python interpreter is missing')
    if (type(max_attempts) is not int or not 1<=max_attempts<=64 or
            not all(math.isfinite(x) for x in (initial_delay,max_delay)) or not 0<initial_delay<=max_delay):
        raise ValueError('Positive bounded retry/backoff settings required')
    work.mkdir(parents=True,exist_ok=True)
    with (work/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bound=frozen_sources(source,out)
        binding=dict(source_bindings=bound,helper_sha256=sha(__file__),python=str(python),root=str(root),
                     max_attempts=max_attempts,initial_delay=initial_delay,max_delay=max_delay)
        statepath=work/'state.json'
        state=read(statepath) if statepath.exists() else dict(binding=binding,attempts=[],created_at=time.time())
        if state.get('binding')!=binding:raise RuntimeError('Recovery registration changed; preserve and inspect old evidence')
        def status(value,**extra):
            state.update(status=value,pid=os.getpid(),updated_at=time.time(),**extra);write(statepath,state)
        def verify():
            if frozen_sources(source,out)!=bound:raise RuntimeError('Frozen source changed during recovery')
        try:
            active=named_processes(source)
            if active:
                if ready_assets(out) and all(p['script']=='supervisor.py' for p in active):
                    status('SUPERVISOR_ALREADY_RUNNING',active_processes=active);return
                raise RuntimeError('Previous named process is still alive: '+json.dumps(active))
            status('RECOVERY_STARTED')
            while ready_assets(out) is None:
                verify()
                if len(state['attempts'])>=max_attempts:raise RuntimeError('Bounded network retries exhausted')
                if named_processes(source):raise RuntimeError('Concurrent asset/supervisor process detected')
                require_retryable_failure(out,state['attempts'])
                attempt=len(state['attempts'])+1
                folder=work/f'attempt_{attempt:02d}_{time.time_ns()}';evidence=archive_receipts(out,folder)
                launched_before=time.time()
                child,launch=start(source/'prepare_assets.py',python,root,folder/'prepare_assets.log')
                entry=dict(attempt=attempt,launch=launch,previous_receipts=evidence)
                state['attempts'].append(entry);write(folder/'launch.json',launch)
                write(out/'assets_launch.json',dict(launch,started=launch['launched_at'],
                    source_sha256=sha(source/'prepare_assets.py'),recovery_owner=os.getpid()))
                status('PREPARER_RUNNING',current_attempt=attempt)
                code=child.wait();entry.update(returncode=code,finished_at=time.time())
                verify()
                if named_processes(source):raise RuntimeError('Named process remains after preparer exit')
                if code==0:
                    if ready_assets(out) is None:raise RuntimeError('Preparer exited successfully without ASSETS_READY')
                    status('ASSETS_READY');break
                failure_path=out/'assets_status.json';failure=read(failure_path)
                entry['failure']=failure;write(folder/'failed_assets_status.json',failure)
                if (failure.get('status')!='FAILED' or failure.get('time',0)<launched_before or
                        not transient(failure.get('error','')) or code<0):
                    raise RuntimeError('Non-transient or unverified preparation failure: '+str(failure.get('error',code)))
                delay=min(max_delay,initial_delay*2**(attempt-1))
                if attempt>=max_attempts:raise RuntimeError('Bounded network retries exhausted')
                status('WAITING_TO_RETRY',last_network_error=failure['error'],retry_after=time.time()+delay)
                time.sleep(delay)
            verify();assets=ready_assets(out)
            active=named_processes(source)
            if active:
                if all(p['script']=='supervisor.py' for p in active):
                    status('SUPERVISOR_ALREADY_RUNNING',active_processes=active);return
                raise RuntimeError('Preparer still active before supervisor restart')
            folder=work/f'supervisor_{time.time_ns()}';archive_receipts(out,folder)
            child,launch=start(source/'supervisor.py',python,root,folder/'supervisor.log')
            write(folder/'launch.json',launch);state['supervisor_launch']=launch
            write(out/'supervisor_launch.json',dict(launch,started=launch['launched_at'],
                source_bindings=bound,recovery_owner=os.getpid()))
            status('SUPERVISOR_STARTING',assets_complete_sha256=sha(out/'assets_complete.json'),
                   manifest_sha256=assets['manifest_sha256'])
            # Detect immediate setup failure; supervision then belongs to the
            # unchanged supervisor, whose own lock remains authoritative.
            time.sleep(2)
            if child.poll() is not None:raise RuntimeError('Metric supervisor exited immediately: '+str(child.returncode))
            actual=process_identity(child.pid)
            if actual is None or actual['state']=='Z' or actual['start_ticks']!=launch['start_ticks']:
                raise RuntimeError('Restarted supervisor identity changed')
            verify();status('SUPERVISOR_RESTARTED')
        except BaseException as error:
            status('FAILED',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--python',required=True,help='Same interpreter path used for the original preparer and supervisor')
    parser.add_argument('--max-attempts',type=int,default=8)
    parser.add_argument('--initial-delay',type=float,default=60.)
    parser.add_argument('--max-delay',type=float,default=900.)
    args=parser.parse_args()
    recover(args.root,args.python,args.max_attempts,args.initial_delay,args.max_delay)
