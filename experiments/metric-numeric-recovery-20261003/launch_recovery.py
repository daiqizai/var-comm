"""Exact-entry startup and launch proof for frozen metric numeric recovery."""
from __future__ import annotations
import argparse
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

ROOT=Path('/home/liulu/projects/VAR_COMM')
NAME='METRIC-NUMERIC-RECOVERY-20261003'
ENV='METRIC_NUMERIC_RECOVERY_MANIFEST'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def sources(here):
    files=sorted(p for p in Path(here).resolve().iterdir() if p.suffix in ('.py','.md'))
    if not files or any(p.is_symlink() or not p.is_file() for p in files):raise RuntimeError('Invalid recovery source inventory')
    return {str(p):sha(p) for p in files}
def verify(bound):
    if not isinstance(bound,dict) or not bound:raise RuntimeError('Missing numeric recovery bindings')
    for path,digest in bound.items():
        if not re.fullmatch('[0-9a-f]{64}',str(digest)) or sha(path)!=digest:raise RuntimeError('Numeric recovery input changed: '+path)
def layout(root=ROOT):
    root=Path(root).resolve();out=root/'outputs'/NAME;final=root/'outputs/METRIC-FINAL-CACHE-20261002'
    return dict(root=root,out=out,runtime=out/'runtime',controller=final/'runtime/controller.py',
        final_runner=final/'runtime/final_runner.py',cache=root/'outputs/METRIC-CONCURRENT-R4-20261002',
        original=root/'experiments/unified-metrics-20261002')
def role(argv,paths):
    if argv and Path(argv[0]).resolve()==paths['controller']:
        if argv!=[str(paths['controller']),'--root',str(paths['root'])]:raise RuntimeError('Unexpected frozen guardian arguments')
        return 'controller'
    if argv and Path(argv[0]).resolve()==paths['final_runner']:
        if argv!=[str(paths['final_runner']),'--root',str(paths['root']),'--cache-dir',str(paths['cache'])]:raise RuntimeError('Unexpected frozen final runner arguments')
        return 'final_runner'
    return None
def process(pid,proc=Path('/proc')):
    try:raw=(Path(proc)/str(int(pid))/'stat').read_text()
    except (FileNotFoundError,ProcessLookupError):return None
    fields=raw[raw.rfind(') ')+2:].split()
    if len(fields)<20 or not fields[19].isdigit():raise RuntimeError('Cannot establish recovery process identity')
    return dict(pid=int(pid),start_ticks=fields[19],state=fields[0])
def identity():
    me=process(os.getpid())
    if me is None:raise RuntimeError('Recovery process disappeared')
    return {k:me[k] for k in ('pid','start_ticks')}


def validate(path,argv,here,*,root=ROOT):
    paths=layout(root);path=Path(path).resolve();here=Path(here).resolve();scope=role(argv,paths)
    if scope is None or path!=paths['out']/'manifest.json' or here!=paths['runtime']:raise RuntimeError('Unexpected recovery execution/manifest location')
    value=read(path);own=sources(here)
    if (value.get('status')!='REGISTERED_METRIC_NUMERIC_RECOVERY' or value.get('root')!=str(paths['root'])
            or value.get('controller_entry')!=str(paths['controller']) or value.get('final_runner_entry')!=str(paths['final_runner'])
            or value.get('cache_dir')!=str(paths['cache']) or value.get('runtime_source_bindings')!=own
            or value.get('training_updates')!=0 or value.get('policy_selection_updates')!=0
            or value.get('inference_functions_changed') is not False or value.get('metric_values_changed') is not False):
        raise RuntimeError('Numeric recovery manifest scope/source differs')
    frozen=value.get('frozen_source_bindings');failure=value.get('previous_failure_bindings')
    verify(own);verify(frozen);verify(failure)
    for folder in (paths['original'],paths['cache']/'runtime',paths['controller'].parent):
        if any(frozen.get(p)!=h for p,h in sources(folder).items()):raise RuntimeError('Full original/R4/final sources are not pinned')
    assets=paths['root']/'outputs/UNIFIED-METRICS-20261002/assets_complete.json'
    if frozen.get(str(assets))!=sha(assets) or value.get('python_executable')!=read(assets)['environment']:
        raise RuntimeError('Original Python environment is not pinned')
    qp=Path(value.get('qualification_path','')).resolve()
    if qp!=paths['out']/'numeric_snr_qualification.json' or sha(qp)!=value.get('qualification_sha256'):
        raise RuntimeError('Numeric parser real qualification path/hash differs')
    q=read(qp)
    if (q.get('status')!='REAL_NUMERIC_SNR_COMPATIBILITY_PASS' or q.get('training_updates')!=0 or q.get('policy_selection_updates')!=0
            or any(q.get(k) is not True for k in ('raw_strings_unchanged','source_row_hashes_unchanged','row_ids_unchanged',
                'actual_grid_equal','integer_grid_valid','other_studies_unchanged'))):
        raise RuntimeError('Actual CSV parser compatibility has not passed')
    verify(q.get('source_bindings'));verify(q.get('table_bindings'))
    required={str(here/'numeric_snr.py'):own.get(str(here/'numeric_snr.py')),
              str(paths['original']/'replay.py'):frozen.get(str(paths['original']/'replay.py'))}
    if any(not h or q['source_bindings'].get(p)!=h for p,h in required.items()):raise RuntimeError('Numeric qualification used different parser/replay')
    bound={str(path):sha(path),**own,**frozen,**failure,str(qp):sha(qp),**q['source_bindings'],**q['table_bindings']}
    verify(bound)
    return value,paths,scope,bound


def recorder(manifest,paths,scope,bound):
    who=identity();destination=paths['out']/'executions'/f'{who["pid"]}_{who["start_ticks"]}.json'
    base=dict(**who,role=scope,inputs=bound,manifest_sha256=sha(paths['out']/'manifest.json'),
        runtime_source_bindings=manifest['runtime_source_bindings'],training_updates=0,policy_selection_updates=0,
        inference_functions_changed=False,metric_values_changed=False,scientific_completion_claimed=False,
        environment=dict(manifest_path=os.environ.get(ENV),pythonpath=os.environ.get('PYTHONPATH','')))
    events=[]
    def record(status,**details):
        events.append(dict(status=status,time=time.time(),**details))
        write(destination,dict(base,status=status,events=list(events),time=time.time()))
    record('STARTUP_VALIDATED');atexit.register(lambda:record('PROCESS_EXIT_RECORDED'))
    return record


def install_parent(manifest,paths,bound,record):
    original=subprocess.Popen
    if getattr(original,'_numeric_recovery',False):raise RuntimeError('Parent recovery already installed')
    expected=[manifest['python_executable'],'-u',str(paths['final_runner']),'--root',str(paths['root']),'--cache-dir',str(paths['cache'])]
    def popen(*args,**kwargs):
        command=args[0] if args else kwargs.get('args')
        if type(command) in (list,tuple) and list(command)==expected:
            if kwargs.get('shell',False) or type(kwargs.get('env')) is not dict:raise RuntimeError('Unexpected final runner process launch')
            verify(bound);env=dict(kwargs['env']);env[ENV]=str(paths['out']/'manifest.json')
            env['PYTHONPATH']=str(paths['runtime'])+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            kwargs=dict(kwargs,env=env)
            record('FINAL_RUNNER_ENVIRONMENT_INJECTED',command=expected)
        elif type(command) in (list,tuple) and str(paths['final_runner']) in command:
            raise RuntimeError('Unregistered final runner launch arguments')
        return original(*args,**kwargs)
    popen._numeric_recovery=True;subprocess.Popen=popen
    return original


def adapt_imports(module,manifest,paths,bound,record):
    original=module.imports
    if (Path(module.__file__).resolve()!=paths['cache']/'runtime/concurrent_runner.py'
            or Path(original.__code__.co_filename).resolve()!=Path(module.__file__).resolve()
            or getattr(original,'_numeric_recovery',False)):
        raise RuntimeError('Unexpected frozen imports function')
    def imports(root):
        if Path(root).resolve()!=paths['root']:raise RuntimeError('Recovery root changed')
        verify(bound);answer=original(root);replay=answer[1]
        spec=importlib.util.spec_from_file_location('_bound_numeric_snr_recovery',paths['runtime']/'numeric_snr.py')
        adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
        receipt=adapter.install(replay,root)
        create=replay.create_engine
        if getattr(create,'_numeric_recovery',False):raise RuntimeError('Replay creation hook already installed')
        def create_engine(*a,**k):
            verify(bound);engine=create(*a,**k)
            for path in bound:
                if engine.bind(path)!=bound[path]:raise RuntimeError('Engine recovery artifact changed')
            original_manifest=engine.manifest
            def engine_manifest(self):
                value=original_manifest();value['numeric_snr_compatibility']=dict(receipt=receipt,
                    manifest_path=str(paths['out']/'manifest.json'),manifest_sha256=bound[str(paths['out']/'manifest.json')],
                    inputs=bound,raw_csv_unchanged=True,original_source_bindings_preserved=True)
                return value
            engine.manifest=types.MethodType(engine_manifest,engine)
            record('RECOVERED_ENGINE_BOUND',studies=list(engine.rows))
            return engine
        create_engine._numeric_recovery=True;replay.create_engine=create_engine
        record('NUMERIC_SNR_HOOK_INSTALLED',compatibility=receipt)
        return answer
    imports._numeric_recovery=True;module.imports=imports


class ImportAdapter(importlib.abc.MetaPathFinder):
    def __init__(self,manifest,paths,bound,record):self.arguments=manifest,paths,bound,record;self.used=False
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='concurrent_runner':return None
        if self.used:raise RuntimeError('Concurrent imports hook loaded twice')
        manifest,paths,bound,record=self.arguments
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or Path(spec.origin).resolve()!=paths['cache']/'runtime/concurrent_runner.py':
            raise RuntimeError('Concurrent module origin differs')
        original=spec.loader;self.used=True
        class Loader:
            def create_module(self,spec):return original.create_module(spec)
            def exec_module(self,module):
                original.exec_module(module);adapt_imports(module,manifest,paths,bound,record)
        spec.loader=Loader();return spec


def install(path,argv,here,*,root=ROOT):
    manifest,paths,scope,bound=validate(path,argv,here,root=root);record=recorder(manifest,paths,scope,bound)
    if scope=='controller':install_parent(manifest,paths,bound,record)
    else:
        if 'concurrent_runner' in sys.modules:raise RuntimeError('Frozen concurrent module imported before recovery')
        sys.meta_path.insert(0,ImportAdapter(manifest,paths,bound,record))
    record('PARENT_LAUNCH_HOOK_READY' if scope=='controller' else 'CHILD_IMPORT_HOOK_READY')


def launch(path,*,root=ROOT):
    paths=layout(root);here=Path(__file__).resolve().parent
    argv=[str(paths['controller']),'--root',str(paths['root'])]
    manifest,paths,_,bound=validate(path,argv,here,root=root)
    old=manifest.get('previous_controller',{})
    if type(old.get('pid')) is not int or old['pid']<=1 or not str(old.get('start_ticks','')).isdigit():raise RuntimeError('Previous guardian identity missing')
    actual=process(old['pid'])
    if actual and actual['state']!='Z' and actual['start_ticks']==str(old['start_ticks']):raise RuntimeError('Previous guardian remains alive')
    current=read(paths['controller'].parent.parent/'controller_launch.json')
    if current.get('pid')!=old['pid'] or str(current.get('start_ticks'))!=str(old['start_ticks']):
        prior_path=paths['out']/'launch.json'
        if not prior_path.exists():raise RuntimeError('Expected failed guardian lineage differs')
        prior=read(prior_path)
        if (prior.get('previous_controller')!=old or prior.get('manifest_sha256')!=sha(path)
                or prior.get('runtime_source_bindings')!=manifest['runtime_source_bindings']
                or current.get('pid')!=prior.get('pid') or str(current.get('start_ticks'))!=str(prior.get('start_ticks'))):
            raise RuntimeError('Previous numeric recovery launch differs')
        prior_actual=process(prior['pid'])
        if prior_actual and prior_actual['state']!='Z' and prior_actual['start_ticks']==str(prior['start_ticks']):
            raise RuntimeError('Previous recovery guardian remains alive')
    state=read(paths['controller'].parent.parent/'controller_status.json')
    if state.get('status')!='FAILED' or state.get('pid')!=current['pid']:raise RuntimeError('Only the failed registered guardian can resume')
    if state.get('worker_pid') is not None:
        worker=process(state['worker_pid'])
        if worker and worker['state']!='Z' and worker['start_ticks']==str(state.get('worker_start_ticks')):raise RuntimeError('Prior final worker remains alive')
    import fcntl
    with (paths['out']/'recovery_launch.lock').open('a') as lockfile:
        fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        history=paths['out']/'launch.json'
        if history.exists():
            prior=read(history);worker=process(prior['pid'])
            if worker and worker['state']!='Z' and worker['start_ticks']==str(prior['start_ticks']):raise RuntimeError('Recovery guardian already running')
        verify(bound);env=dict(os.environ);env[ENV]=str(Path(path).resolve())
        env['PYTHONPATH']=str(paths['runtime'])+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        command=[manifest['python_executable'],'-u',*argv]
        with (paths['out']/'controller_recovery.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=paths['root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        who=process(child.pid)
        if who is None:raise RuntimeError('Launched guardian identity unavailable; inspect log')
        value=dict(status='FROZEN_GUARDIAN_RECOVERY_LAUNCHED',pid=who['pid'],start_ticks=who['start_ticks'],command=command,
            previous_controller=old,manifest_sha256=sha(path),runtime_source_bindings=manifest['runtime_source_bindings'],inputs=bound,time=time.time())
        immutable=paths['out']/'launch_history'/f'{who["pid"]}_{who["start_ticks"]}.json'
        write(immutable,value);write(history,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    answer=launch(parser.parse_args().manifest)
    print(json.dumps({k:answer[k] for k in ('status','pid','start_ticks')}))
