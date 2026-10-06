"""One bounded actualcal sequence; no GPU, later population or blind retry.

35 CPU fixture tests may run while the original proxy finishes. New paid calls
remain impossible until both original CPU and prescreen owners have exited,
the immutable request is assembled, and same-ledger registration succeeds.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import main_raw64_actual_driver as d

read,sha,save,require,combine=d.read,d.sha,d.save,d.require,d.combine
STATUS='MAIN_RAW64_ACTUALCAL_SEQUENCE_NORMALLY_COMPLETE_R3'


def recovery_hash(materials):
    """Authenticate the actual failed attempt; never rewrite or retry it."""
    spec=materials['recovery']; pins={v['path']:v['sha256'] for v in spec.values()}
    d.u.verify(pins)
    archive=read(spec['archive']['path']);diag=read(spec['diagnosis']['path'])
    require(archive['status']=='FAILED_ACTUALCAL_QUALIFICATION_ORIGINALS_PRESERVED' and archive['originals_retained'] is True and archive['original_owner_success'] is False,'Original failed qualification must be preserved')
    require(len(archive['items'])==20 and len({x['original'] for x in archive['items']})==20 and len({x['archive'] for x in archive['items']})==20,'Exact original20 archive required')
    for item in archive['items']:
        for key in ('original','archive'):
            p=item[key]; require(Path(p).is_absolute() and Path(p).stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Changed original or archive: '+p)
            pins=combine(pins,{p:item['sha256']})
    require(diag['status']=='ACTUALCAL_PRESTART_HASH_CACHE_FAILURE_DIAGNOSED' and diag['archive_sha256']==sha(spec['archive']['path']) and diag['tests_run']==31 and diag['tests_passed']==30 and diag['tests_failed']==1 and diag['failed_test']=='test_hash_cache_revalidates_changed_file','Actual31 failed-test diagnosis required')
    require(diag['actual_calibration_registered'] is False and diag['actual_calibration_calls']==diag['development_calls']==diag['holdout_calls']==0 and diag['original_scientific_protocol_changed'] is False,'Failed attempt performed scientific work')
    failed=[x['original'] for x in archive['items'] if Path(x['original']).name=='failure.json']; require(len(failed)==1,'Unique original failure required')
    old=Path(failed[0]).parent; endpath=str(old/'qualification_exit.json'); identpath=str(old/'identity.json')
    require(pins.get(endpath)==diag['qualification_exit_sha256'] and pins.get(failed[0])==diag['failure_sha256'],'Failed wait/failure bytes differ')
    end=read(endpath);owner=read(identpath);failure=read(failed[0])
    require(end['process_waited'] is True and end['exit_code']==1 and end['capture_or_monitor_error'] is None and pins.get(end['log'])==end['log_sha256'],'Original qualification did not reap failure normally')
    require(failure['original_owner_success'] is False and failure['retry_allowed'] is False and set(archive['processes_exited'])=={owner['pid'],end['identity']['pid']},'Original failure/process inventory differs')
    d.gone(owner);d.gone(end['identity'])
    require(not (old/'completion.json').exists(),'Original failed sequence cannot claim success')
    ap=str(Path(spec['archive']['path']).parent/'archive_preparation_failure.json')
    require(sha(ap)==archive['archive_preparation_failure_sha256'],'Archive preparation failure must remain preserved')
    return combine(pins,{ap:sha(ap)})


def recovery_receipts(materials):
    """Preserve the real R2 successful tests and failed metadata preparation."""
    spec=materials['receipt_recovery'];pins={v['path']:v['sha256'] for v in spec.values()};d.u.verify(pins)
    archive=read(spec['archive']['path']);diag=read(spec['diagnosis']['path'])
    require(archive['status']=='FAILED_R2_PREPARE_EVIDENCE_PRESERVED' and archive['originals_unchanged'] is True,'Original R2 prepare evidence required')
    entries=archive['entries'];require(len(entries)==21 and len({x['source'] for x in entries})==len({x['copy'] for x in entries})==21,'Exact original21 archive required')
    for item in entries:
        for key in ('source','copy'):
            p=item[key];require(Path(p).is_absolute() and Path(p).stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Changed R2 original/copy: '+p);pins=combine(pins,{p:item['sha256']})
    require(diag['status']=='ORIGINAL_RECEIPT_SCHEMA_MISMATCH_DIAGNOSED' and diag['failure_archive_sha256']==sha(spec['archive']['path']) and diag['actual_qualification_tests']==34 and diag['actual_qualification_passed'] is True,'Actual successful34 and failed prepare diagnosis required')
    require(diag['registration_created'] is False and diag['calibration_calls']==0 and diag['original_owner_success'] is False and diag['processes_exited'] is True,'R2 failure scope differs')
    fp=archive['source_failure'];require(pins.get(fp)==archive['source_failure_sha256'],'Original R2 failure pin differs')
    old=Path(fp).parent;endpath=str(old/'qualification_exit.json');identitypath=str(old/'identity.json')
    require(endpath in pins and identitypath in pins,'Original R2 process evidence missing')
    end=read(endpath);owner=read(identitypath);failure=read(fp)
    require(end['process_waited'] is True and end['exit_code']==0 and end['capture_or_monitor_error'] is None and pins.get(end['log'])==end['log_sha256'],'R2 tests must have normal wait0 and closed log')
    require(failure['original_owner_success'] is False and failure['retry_allowed'] is False and 'budget_after' in failure['traceback'],'Original precise prepare failure required')
    qs=[x['source'] for x in entries if Path(x['source']).name=='completion.json'];require(len(qs)==1,'Only original qualification completion expected')
    q=read(qs[0]);require(q['status']==d.QSTATUS and q['tests_run']==34 and q['test_modules']==d.TEST_MODULES and q['process_waited'] is True and q['exit_code']==0 and q['GPU_used'] is False and q['new_packet_decodes']==0 and q['log']==end['log'] and q['outputs'].get(endpath)==sha(endpath) and q['outputs'].get(end['log'])==end['log_sha256'],'R2 actual qualification receipt differs')
    text=Path(end['log']).read_text();require('Ran 34 tests' in text and text.rstrip().endswith('OK') and 'skipped=' not in text,'Original complete34 log required')
    d.gone(owner);d.gone(end['identity']);require(not (old/'completion.json').exists(),'Failed R2 sequence cannot claim normal success')
    return pins


def recovery(materials):
    return combine(recovery_hash(materials),recovery_receipts(materials))


def test_metadata(materials):
    path=materials['test_metadata'];meta=read(path)
    require(meta['status']=='ACTUALCAL_R3_REAL_PREPARE_CONTRACT_TEST_ASSETS' and len(meta['files'])==47,'Exact real prepare metadata fixture required')
    pins={path:sha(path),meta['materials']['path']:meta['materials']['sha256']}
    require(len({x['remote'] for x in meta['files']})==47,'Unique real fixture paths required')
    for item in meta['files']:
        require(item['path']==item['remote'],'Actual qualification uses original remote metadata, not local aliases')
        pins=combine(pins,{item['path']:item['sha256']})
    d.u.verify(pins);return pins


def source_inventory(materials):
    rootcfg=read(materials['root_config']);rootreg=read(rootcfg['registration']);oldreg=read(materials['legacy_batch']['registration'])
    require(rootreg['config_sha256']==sha(materials['root_config']),'Original root config changed')
    sources=combine(rootreg['source_bindings'],oldreg['source_bindings'],{p:sha(p) for p in materials['source_paths']})
    for mod in (d.u,d.c,d.budget):
        p=str(Path(mod.__file__).absolute());sources=combine(sources,{p:sha(p)})
    for name in ['main_raw64_actual_driver']+d.TEST_MODULES:
        p=str(Path(d.__file__).absolute().with_name(name+'.py'));sources=combine(sources,{p:sha(p)})
    original_runner=str(Path(d.u.__file__).absolute());original_ledger=str(Path(original_runner).with_name('main_raw64_ledger.py'))
    require(rootreg['source_bindings'].get(original_runner)==sha(original_runner) and rootreg['source_bindings'].get(original_ledger)==sha(original_ledger),'Import u/ledger from their ORIGINAL root registered runtime; no copied substitute')
    d.u.verify(sources)
    return sources,rootcfg,original_ledger


def environment(materials,ledger_module):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(materials['pythonpath']),MAIN_RAW64_ROOT_LEDGER_MODULE=ledger_module)
    env.update({k:'2' for k in d.u.ENV_THREADS})
    env['ACTUALCAL_R3_TEST_METADATA']=materials['test_metadata'];return env


def check(cfg):
    d.u.check_stop(cfg)


def call(e,name,argv,env,affinity,cfg,*,actual_owner=False):
    """Always reap every spawned process; owner errors request STOP first."""
    logpath=e/(name+'.log');exitpath=e/(name+'_exit.json');error=None;identity=None
    with logpath.open('xb') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,preexec_fn=lambda:d.u.set_resources(affinity))
        try:
            save(e/(name+'_pending.json'),dict(pid=child.pid,argv=argv,identity_verified=False))
            identity=d.u.capture(child,argv);save(e/(name+'_launch.json'),dict(identity=identity,expected_argv=argv))
            while child.poll() is None:
                check(cfg);time.sleep(.2)
        except BaseException:
            error=traceback.format_exc()
            if actual_owner:
                try:d.u.request_stop(cfg,'Actualcal sequence owner supervision failed')
                except BaseException:error+='\nSTOP_WRITE_FAILED\n'+traceback.format_exc()
        finally:
            # Further signals cannot interrupt the one required wait.
            previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
            for s in previous:signal.signal(s,signal.SIG_IGN)
            try:
                rc=child.wait();log.flush();os.fsync(log.fileno())
            finally:
                for s,h in previous.items():signal.signal(s,h)
    receipt=dict(identity=identity,expected_argv=argv,process_waited=True,exit_code=rc,log=str(logpath),log_sha256=sha(logpath),capture_or_monitor_error=error)
    save(exitpath,receipt)
    require(error is None and rc==0,'Original child did not close normally: '+name)
    return str(exitpath)


def wait_predecessors(m,cfg):
    p=m['prescreen'];paths=[m['root_completion'],m['root_owner_exit'],p['completion'],p['science'],p['worker_exit'],p['sequence_completion'],p['owner_exit'],p['owner_identity'],p['sequence_identity']]
    failure_paths=[Path(cfg['root'])/'cpu_sequence_v1/failure.json',Path(m['root_completion']).parent/'failure.json',Path(p['completion']).parent/'failure.json',Path(p['sequence_completion']).parent/'failure.json']
    while True:
        check(cfg);require(not any(x.exists() for x in failure_paths),'A predecessor failed; preserve it and stop without retry')
        if all(Path(x).is_file() for x in paths):
            identities=[read(m['root_owner_exit'])['identity'],read(p['owner_identity']),read(p['sequence_identity']),read(p['worker_exit'])['identity']]
            try:
                for who in identities:d.gone(who)
            except ValueError:
                time.sleep(.2);continue
            return
        time.sleep(1)


def qualification(m,e,sources,cfg,ledger_module,affinity,recovery_pins=None):
    qp=Path(m['prepared_qualification']);require(not qp.parent.exists(),'Fresh actual CPU qualification directory required')
    qp.parent.mkdir(parents=True)
    argv=[cfg['python'],'-B','-m','unittest',*d.TEST_MODULES]
    endpath=call(e,'qualification',argv,environment(m,ledger_module),affinity,cfg)
    end=read(endpath);text=Path(end['log']).read_text()
    require(('Ran '+str(d.TEST_COUNT)+' tests') in text and text.rstrip().endswith('OK') and 'skipped=' not in text,'Complete actual35 CPU tests without skips required')
    d.u.verify(sources)
    q=dict(status=d.QSTATUS,tests_run=d.TEST_COUNT,test_modules=d.TEST_MODULES,python=cfg['python'],pythonpath=m['pythonpath'],process_waited=True,exit_code=0,GPU_used=False,new_packet_decodes=0,model_imports=False,source_bindings=sources,input_bindings={m['root_config']:sha(m['root_config']),cfg['registration']:sha(cfg['registration']),m['legacy_batch']['registration']:sha(m['legacy_batch']['registration'])},outputs={endpath:sha(endpath),end['log']:end['log_sha256']},log=end['log'])
    q['input_bindings']=combine(q['input_bindings'],recovery_pins or {})
    save(qp,q)
    # Consume the exact receipt before entering any actual stage.
    d.qualification(dict(prepared_qualification=str(qp),python=cfg['python'],source_bindings=sources),{str(qp):sha(qp)})
    return str(qp)


@contextmanager
def sequence_lock(cfg):
    import fcntl
    with (Path(cfg['root'])/'actualcal_sequence.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def run(materials_path,execution_dir,qualification_affinity):
    m=read(materials_path);recovery_pins=combine(recovery(m),test_metadata(m));sources,cfg,ledger_module=source_inventory(m)
    require(sys.platform.startswith('linux') and str(Path(sys.executable).absolute())==cfg['python'],'Original exact LDPC interpreter required')
    e=Path(execution_dir).absolute();root=Path(cfg['root']).absolute();require(root in e.parents and not e.exists(),'Fresh sequence directory below study root required')
    require(len(qualification_affinity)==len(set(qualification_affinity))==2 and not set(qualification_affinity)&set(sum(m['resources']['affinities'],[])),'Independent two-core CPU-only fixture affinity required')
    require(str(Path(__file__).absolute()) in m['source_paths'],'Sequence helper must join immutable source union')
    check(cfg);d.u.set_resources(sorted(set(sum(m['resources']['affinities'],[])+qualification_affinity)))
    with sequence_lock(cfg):
        e.mkdir(parents=True);save(e/'registration.json',dict(status='MAIN_RAW64_ACTUALCAL_SEQUENCE_REGISTERED_V1',materials_path=str(Path(materials_path).absolute()),materials_sha256=sha(materials_path),source_bindings=sources,deadline_unix=cfg['deadline_unix'],scope=['CPU_interface_qualification','wait_normal_predecessors','actual_calibration_registration','actual_calibration_CPU_only'],automatic_visual=False,automatic_development=False))
        save(e/'recovery_evidence.json',dict(status='ORIGINAL_QUALIFICATION_AND_PREPARE_FAILURES_PRESERVED_R3',input_bindings=recovery_pins,original_owner_success=False,hash_policy='FRESH_BYTES_NO_CACHE',legacy_science_budget_field='budget'))
        save(e/'identity.json',d.u.process(os.getpid()))
        previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        def stop(signum,frame):
            for s in previous:signal.signal(s,signal.SIG_IGN)
            raise InterruptedError('Sequence interrupted')
        for s in previous:signal.signal(s,stop)
        try:
            q=qualification(m,e,sources,cfg,ledger_module,qualification_affinity,recovery_pins)
            wait_predecessors(m,cfg);d.u.verify(sources)
            request=str(e/'request.json');prepared=d.build_request(materials_path,request)
            require(read(request)['source_bindings']==sources,'Prepared and qualified source union differ')
            # Separate explicit child registration; it cannot decode or launch.
            register_exit=call(e,'registration',[cfg['python'],'-B',str(Path(d.__file__).absolute()),'--stage','register','--request',request],environment(m,ledger_module),qualification_affinity,cfg)
            config=str(Path(m['execution_dir'])/'config.json');ctx=d.load_registered(config)
            require(ctx['cfg']['registration']==str(Path(m['execution_dir'])/'registration.json'),'Unexpected actual registration path')
            argv=[cfg['python'],'-B',str(Path(d.__file__).absolute()),'--config',config,'--stage','run']
            endpath=call(e,'owner',argv,environment(m,ledger_module),sum(m['resources']['affinities'],[]),cfg,actual_owner=True)
            normal=str(Path(m['execution_dir'])/'completion.json');end=read(endpath)
            spec=dict(config=config,registration=ctx['cfg']['registration'],completion=normal,owner_exit=endpath,bindings={p:sha(p) for p in (config,ctx['cfg']['registration'],normal,endpath,end['log'])})
            closed=d.closed_calibration(spec);d.u.verify(sources)
            closure_path=e/'normal_closed_observation.json';save(closure_path,dict(status='MAIN_RAW64_ACTUAL_NORMAL_INVENTORY_VERIFIED_V1',spec=spec,owner_completion_sha256=sha(normal),science_completion_sha256=sha(Path(m['out'])/'completion.json'),point_count=len(closed['points']),source_count=len(closed['records']),frame_count=closed['done']['frame_count'],new_packet_calls=closed['done']['new_packet_calls'],reused_frames=closed['done']['reused_frames'],budget_after=closed['budget_before'],normal_owner_success=True,all_input_output_hashes_verified=True,automatic_successor_started=False))
            done=dict(status=STATUS,original_owner_success=True,owner_completion_sha256=sha(normal),owner_exit_sha256=sha(endpath),qualification_sha256=sha(q),request_sha256=prepared['sha256'],normal_closed_observation_sha256=sha(closure_path),budget_after=closed['budget_before'],outputs={p:sha(p) for p in (q,request,register_exit,endpath,str(closure_path),normal,str(Path(m['out'])/'completion.json'))},automatic_visual=False,automatic_development=False,GPU_used=False)
            save(e/'completion.json',done);return done
        except BaseException:
            save(e/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),original_owner_success=False,retry_allowed=False));raise
        finally:
            for s,h in previous.items():signal.signal(s,h)


def main():
    p=argparse.ArgumentParser();p.add_argument('--materials',required=True);p.add_argument('--execution-dir',required=True);p.add_argument('--qualification-affinity',nargs=2,type=int,required=True);a=p.parse_args()
    print(run(a.materials,a.execution_dir,a.qualification_affinity)['status'],flush=True)


if __name__=='__main__':main()
