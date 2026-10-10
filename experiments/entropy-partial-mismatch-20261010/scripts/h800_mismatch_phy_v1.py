"""Finite CPU owner for the frozen raw one-bin configuration mismatch diagnostic.

Only prepare/run are explicit entry points. No automatic successor or retry.
This registers5400 logical/4500 physical frames, at most9000 packet decodes,
and finite later visual/metric caps without admitting those later stages.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import h800_ep_pilot_phy_v1 as base
import h800_mismatch_input_plan_v1 as inputs
import h800_mismatch_phy_core_v1 as core

g=base.g;gate=base.gate;plan=inputs.frozen
SCHEMA='H800_RAW100_ONE_BIN_MISMATCH_PHY_V1'
PASS='PASS_H800_RAW100_ONE_BIN_MISMATCH_PHY_ONLY'
OLD='/home/liulu/projects/VAR_COMM/'
RESTORATIONS=(('mismatch_raw100_seed_v1',inputs.RESTORE_SHA),
    ('mismatch_raw_dependencies_seed_v1','819d7f895f60c71bba0d3748ab3bed0fb188aea351409ebfb61c6796a879e689'),
    ('mismatch_raw_action_seed_v1','ef642ca8c0312bc33d2a9767ac84e70d86677e28fa37e088494ccbee8b12a6f2'))
SOURCE_PINS={'h800_ep_pilot_phy_v1.py':'61683057d883ae6ecd317d7ab998930e9e367faDE54ca3ef97d791431f90aaed'.lower(),
    'mismatch_plan.py':'c15e7975334bd40eb46fe35cdaa65a7e4532b63514540eb98cb05bf36a9ba0bf',
    'h800_mismatch_input_plan_v1.py':'cd76c5dda42635be8fd21d32d2e2c9a660506970e753ecafb372c4f629ed895b'}
TOOLS=base.TOOLS+('h800_ep_pilot_phy_v1.py','mismatch_plan.py','h800_mismatch_input_plan_v1.py','h800_mismatch_phy_core_v1.py')


def restoration_map():
    mapped={};pins=[]
    for name,digest in RESTORATIONS:
        path=gate.RT/'incoming'/name/'hydration_completion.json';receipt=g.checked_json(path,digest)
        g.require(receipt['model_calls']==receipt['packet_calls']==receipt['metric_calls']==receipt['source_array_loads']==0,
            'Restoration must contain only original bytes')
        for row in receipt['mapping']:
            g.require(row['original_path'] not in mapped,'Ambiguous restored original path');mapped[row['original_path']]=row
        pins.append(dict(path=str(path),sha256=digest))
    return mapped,pins


def original_file(original,digest,mapped):
    g.require(original.startswith(OLD),'Explicit original project path required')
    candidates=([mapped[original]['actual_path']] if original in mapped else [])+[
        str(gate.R2/original[len(OLD):]),str(gate.RT/'assets/frozen_namespace_v1/VAR_COMM'/original[len(OLD):])]
    for value in candidates:
        path=g.inside(value)
        if path.is_file() and not path.is_symlink() and g.sha(path)==digest:
            return dict(original_path=original,path=str(path),sha256=digest)
    raise RuntimeError('Missing exact original dependency: '+original)


def original_inputs():
    mapped,restore=restoration_map();origin=OLD+'outputs/MAIN-RAW64-20261007/'
    cp=original_file(origin+'unified500_raw_parallel_execution_r2/config.json',inputs.CPU_CONFIG_SHA,mapped)
    dp=original_file(origin+'unified500_raw_parallel_execution_r2/completion.json',inputs.CPU_COMPLETE_SHA,mapped)
    cfg=core.readpin(cp);done=core.readpin(dp)
    fixture=Path(__file__).parents[1]/'tests/fixtures/common500_v1'
    original=g.checked_json(fixture/'plan.json',plan.ORIGINAL_PLAN_SHA)
    manifest=g.checked_json(fixture/'manifest.json',plan.SOURCE_MANIFEST_SHA)
    source_rows,schedule=plan.validate_inputs(original,manifest)
    g.require(done['status']=='MAIN_RAW64_UNIFIED500_RAW_NORMALLY_COMPLETE_V1' and done['all_waited'] is True and
        done['source_count']==500 and done['config_sha256']==inputs.CPU_CONFIG_SHA,'Original closure metadata differs')
    deps=[original_file(row['original_path'],row['sha256'],mapped) for row in inputs.requirements(cfg)]
    resolved={row['original_path']:row['path'] for row in deps}
    policies=g.checked_json(resolved[cfg['selected']],inputs.POLICY_SHA)
    g.require(policies['status']=='POLICIES_FROZEN_ON_CALIBRATION1000' and policies['holdout_used'] is False and
        {(r['snr_db'],r['family']):r['candidate_id'] for r in policies['winners']}==
        {key:row['candidate_id'] for key,row in schedule.items()},'Frozen raw winners changed')
    records=[]
    for row in source_rows:
        record=dict(row)
        for name in ('archive','checkpoint'):
            value=original_file(row[name],row[name+'_sha256'],mapped)
            record['actual_'+name]={k:value[k] for k in ('path','sha256')}
        checkpoint=core.readpin(record['actual_checkpoint'])
        g.require(all(checkpoint[k]==record[k] for k in ('source_index','source_id','tokens_sha256','preprocessing_id')) and
            checkpoint['archive']==record['archive'] and checkpoint['outputs'][record['archive']]==record['archive_sha256'] and
            checkpoint['encoder_tokens_verified'] is True and checkpoint['asset_readback_exact'] is True,
            'Original source checkpoint/order differs')
        records.append(record)
    action=OLD+'outputs/CONTENT-REAL-64QAM-20261006/prepared_v1/main_physical_runtime_r2/main_action_space.py'
    ap=original_file(action,cfg['source_bindings'][action],mapped);deps.append(ap)
    header=[original_file(path,h,mapped) for path,h in cfg['source_bindings'].items() if path.startswith(OLD+'src/var_comm/')]
    g.require(len(header)==25 and all(Path(r['path']).is_relative_to(gate.R2/'src/var_comm') for r in header),
        'Exact original25 source package files required')
    actual={str(p) for p in (gate.R2/'src/var_comm').rglob('*') if p.is_file() and p.suffix in ('.py','.cpp')}
    g.require(actual=={r['path'] for r in header},'Paid-header import source closure changed');deps+=header
    noise=fixture/'raw64_unified500_plan.py';g.require(g.sha(noise)==plan.ORIGINAL_NOISE_CODE_SHA,'Original common500 noise source changed')
    deps.append(dict(original_path=str(noise),path=str(noise),sha256=plan.ORIGINAL_NOISE_CODE_SHA))
    roles={name:resolved[path] for name,path in cfg['adapter_config'].items()}
    roles.update({name:resolved[cfg[name]] for name in ('adapter_module','receiver_module','original_receiver_module','aliases','profiles')})
    roles.update(action_module=ap['path'],noise_module=str(noise))
    ledger=Path(__file__).parents[2]/'wcl-evidence-closure-20261009/scripts/t2_ledger.py'
    ledger_sha='372355ac0f2b32274da89e3d1790f9c748eaa42e97bafb0b99ce140b8594200f'
    g.require(g.sha(ledger)==ledger_sha,'Existing atomic ledger implementation changed')
    deps.append(dict(original_path=str(ledger),path=str(ledger),sha256=ledger_sha))
    cat=g.json.loads(Path(roles['catalogue']).read_bytes())
    g.require(len(cat['profiles'])==433 and len(cat['aliases'])==523,'Original raw receive catalogue changed')
    binding=dict(files=sorted(deps,key=lambda r:r['original_path']),roles=roles,project_root=str(gate.R2),
        catalogue_sha256=core.digest(cat['profiles']),aliases_sha256=core.digest(cat['aliases']),
        qualification_scope='Original raw LDPC qualification and unchanged raw433 code/catalogue; EP457 is CPU environment evidence only, not a raw433 qualification')
    closure=dict(restorations=restore,original_CPU_config=cp,original_CPU_completion=dp,
        original_plan=core.pin(fixture/'plan.json'),source_manifest=core.pin(fixture/'manifest.json'),
        original_policy=dict(path=resolved[cfg['selected']],sha256=inputs.POLICY_SHA),
        source_population='posthoc_first100_subset_of_original_common500_holdout',historical_reuse_admitted=0)
    return records,list(plan.logical_frames(source_rows,schedule)),binding,closure


def cpu_identity(binding):
    inherited,environment=base.phy_binding()
    # Independent new raw identity; no equality claim to old host numerical output.
    identity=dict(schema='H800_RAW433_ACTUAL_CPU_IDENTITY_V1',raw_source_binding_sha256=core.digest(binding),
        effective_catalogue_sha256=binding['catalogue_sha256'],aliases_sha256=binding['aliases_sha256'],
        CPU_python=inherited['CPU_python'],admitted_CPU_environment=inherited['admitted_CPU_environment'],
        actual_host_native_binding=inherited['actual_host_native_binding'],CUDA_visible=False,
        reused_environment_evidence=inherited['qualification'],historical_event_admission=False)
    return identity,environment


def registration(path,digest):
    r=g.checked_json(g.inside(path),digest);e=base.execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    records,rows,binding,closure=original_inputs();identity,environment=cpu_identity(binding)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and
        r['complete_scientific_caps']==inputs.future_caps() and r['records']==records and r['frames']==rows and
        r['raw_runtime_binding']==binding and r['input_closure']==closure and r['CPU_identity']==identity and
        r['CPU_python']==str(gate.CPU_PYTHON) and r['root']==str(gate.R2), 'Frozen mismatch request changed')
    g.require(r['GPU_execution_admitted'] is r['metrics_execution_admitted'] is r['automatic_successor'] is False and
        r['historical_reuse_admitted']==0,'Only new finite CPU frames admitted; no historical reuse')
    engine().check_tools(r['tool_bindings']);return r,environment


def prepare(a):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh mismatch registration required')
    records,rows,binding,closure=original_inputs();identity,_=cpu_identity(binding)
    e=base.execution(a.deadline_unix,a.max_seconds);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',execution=e,complete_scientific_caps=inputs.future_caps(),
        records=records,frames=rows,raw_runtime_binding=binding,input_closure=closure,CPU_identity=identity,
        CPU_python=str(gate.CPU_PYTHON),root=str(gate.R2),historical_reuse_admitted=0,GPU_execution_admitted=False,
        metrics_execution_admitted=False,automatic_successor=False,preparation_packet_decodes=0,preparation_model_calls=0,
        tool_bindings={name:g.sha(Path(__file__).with_name(name)) for name in (*TOOLS,Path(__file__).name)})
    path=out/'request.json';g.write(path,r);return core.pin(path)


def run(a):
    g.require(g.sys.platform.startswith('linux'),'Actual Linux owner required')
    path=g.inside(a.request);r,environment=registration(path,a.request_sha256);out=path.parent/'run';e=r['execution']
    shared=g.helper();started=time.monotonic();child=None
    with shared.owner_lock(g.inside(g.BASE/'controls/h800_raw_mismatch_phy_v1/CPU.owner.lock')):
        g.require(not out.exists(),'Mismatch already claimed; no automatic repeat');out.mkdir()
        g.write(out/'intent.json',dict(owner_pid=os.getpid(),request_sha256=a.request_sha256,execution=e))
        try:
            resources=shared.cpu_snapshot();g.require(resources['effective_cpu_cores']>=2 and
                len(resources['allowed_cpus'])>=2 and resources['available_memory_bytes']>=2*(1<<30),'Two CPU cores and memory required')
            affinity=resources['allowed_cpus'][:2];g.write(out/'prelaunch_resources.json',resources)
            env=gate.controlled_cpu_environment(out,environment);g.write(out/'controlled_environment.json',env)
            argv=[r['CPU_python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),
                '--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            seconds=min(e['max_seconds']-(time.monotonic()-started),e['deadline_unix']-time.time());g.require(seconds>0,'Deadline expired')
            with (out/'child.log').open('x') as log:
                child=subprocess.Popen(argv,env=env,cwd=r['root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(affinity,seconds,2))
                g.write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,
                    argv=argv,cpu_affinity=affinity,started_unix=time.time()))
                waited=shared.wait_owned(child,seconds)
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Mismatch child failed; no automatic retry')
            done=core.readpin(core.pin(out/'worker_completion.json'));result=done['results']
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and result['logical_count']==5400 and
                result['physical_count']==4500 and result['packet_ledger']['unresolved']==0 and
                result['packet_ledger']['total']<=9000,'Mismatch CPU budget did not actually close')
            registration(path,a.request_sha256)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Mismatch final deadline exceeded')
            result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=core.pin(out/'worker_completion.json'),
                actual_wait=waited,actual_children_waited=True,worker_exit_codes=[child.returncode],automatic_successor=False)
            g.write(out/'completion.json',result);return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            g.write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
                actual_child_waited=child is not None and child.poll() is not None,child_exit_code=None if child is None else child.returncode));raise


def worker(a):
    path=g.inside(a.request);r,_=registration(path,a.request_sha256);out=path.parent/'run';e=r['execution']
    launch=engine().worker_identity(out,a.request_sha256,a.owner_pid)
    g.require(launch['argv'][0]==r['CPU_python'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
        len(os.sched_getaffinity(0))==2,'Actual hidden-CUDA two-core child required')
    stopped=False;started=time.monotonic();last=0.;boundaries=0;ledger=None
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),
            'Mismatch stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Mismatch deadline exceeded')
    def boundary():
        nonlocal last,boundaries
        guard();boundaries+=1
        if time.monotonic()-last>=10:
            last=time.monotonic()
            with (out/'progress.jsonl').open('a') as f:f.write(g.json.dumps(dict(logical_boundaries=boundaries,
                packet_ledger=None if ledger is None else ledger.snapshot(),elapsed_seconds=time.monotonic()-started))+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256))
        # t2_ledger is the existing atomic finite callback ledger, separate new DB.
        sys_path=str(Path(__file__).parents[2]/'wcl-evidence-closure-20261009/scripts')
        if sys_path not in g.sys.path:g.sys.path.insert(0,sys_path)
        from t2_ledger import Ledger
        ledger=Ledger(out/'packet_ledger.sqlite',a.request_sha256,9000)
        runtime=core.create_runtime(r['raw_runtime_binding'],ledger,guard)
        import numpy as np
        actual_identity=dict(r['CPU_identity'],torch_version=runtime.torch.__version__,numpy_version=np.__version__,
            backend_identity=runtime.backend.identity,raw64_backend_identity=runtime.backend.identity64,actual_host=os.uname().nodename,
            flags=dict(threads=runtime.torch.get_num_threads(),interop=runtime.torch.get_num_interop_threads(),
                TF32=runtime.torch.backends.cuda.matmul.allow_tf32,deterministic=runtime.torch.are_deterministic_algorithms_enabled()))
        g.write(out/'actual_runtime_identity.json',actual_identity)
        results=core.execute(runtime,r['frames'],r['records'],ledger,boundary,out/'frames',actual_identity)
        ledger.close();ledger=None;guard();registration(path,a.request_sha256)
        g.require(not runtime.torch.cuda.is_initialized(),'Mismatch initialized CUDA')
        done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=results,
            actual_runtime_identity=core.pin(out/'actual_runtime_identity.json'),CUDA_initialized=False,neural_model_calls=0,
            quality_scores=0,historical_reuse_admitted=0,automatic_successor=False)
        g.write(out/'worker_completion.json',done);return done
    except BaseException as error:
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,
            error=repr(error),traceback=traceback.format_exc(),packet_ledger=None if ledger is None else ledger.snapshot()));raise
    finally:
        if ledger is not None:ledger.close()


def engine():
    for name,digest in SOURCE_PINS.items():g.require(g.sha(Path(__file__).with_name(name))==digest,'Changed frozen owner dependency: '+name)
    old=base.engine();ns=dict(vars(old));ns.update(__file__=__file__,TOOLS=TOOLS)
    for name in ('check_tools','worker_identity'):ns[name]=gate.source.host.clone_function(ns[name],ns)
    return base.source.types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--out',required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();engine();result=prepare(a) if a.command=='prepare' else run(a) if a.command=='run' else worker(a)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
