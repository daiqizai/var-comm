"""Two finite CPU steps: actual nine-test qualification, then exact request.

No registration, GPU construction, worker launch, ranking or ledger access.
The original worker's register/worker entry points remain separate.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

import clean_missing_r1 as c

DEADLINE=1791651239.4938014
ENV_SHA='bb467db2f62f242b639243baaeae8249c5b02301a83b5c66a375bceecab4aedd'
SOURCES={'clean_missing_r1.py':'e9d03dc6a17a8ccf487424db3a9f9ff4e099e1d63fb2fe3d8b103b7b69d5e271',
         'test_clean_missing_r1.py':'e4472566733d87207f5454f0f68b1bd5b843b2a30465d6b7a98aee45d015afd0'}

def checked(path,pins):
    path=str(Path(path).absolute());c.require(pins.get(path)==c.sha(path),'Original pin missing or changed '+path)
    return c.read(path)

def base(root,runtime,visual_config,catalogue,catalogue_sha):
    root=Path(root).absolute();runtime=Path(runtime).absolute();visual_config=str(Path(visual_config).absolute())
    sources={str(runtime/n):h for n,h in SOURCES.items()};c.verify(sources)
    c.require(str(Path(c.__file__).absolute())==str(runtime/'clean_missing_r1.py'),'Import actual uploaded worker runtime')
    v=c.read(visual_config);regpath=v['registration'];vr=c.read(regpath)
    c.require(v['root']==str(root) and vr['config_sha256']==c.sha(visual_config)
              and vr['input_bindings'].get(visual_config)==c.sha(visual_config),'Original MAIN visual config seal differs')
    prior=c.merge(vr['source_bindings'],vr['input_bindings'])
    owner=checked(v['visual_owner_config'],prior)
    c.require(owner['worker_config']==visual_config and owner['python']==v['python']
              and owner['worker_argv'][0]==v['python'],'Original UM/owner identity differs')
    oldroot=root/'outputs/CONTENT-REAL-64QAM-20261006';oldcfg=str(oldroot/'H/execution_v1/source_config.json')
    oc=c.read(oldcfg);oreg=c.read(oc['registration']);oprior=c.merge(oreg['source_bindings'],oreg['input_bindings'])
    checked(oldcfg,oprior)
    c.require(oc['root']==str(root) and oc['uep_runtime']==v['uep_runtime'] and oc['native_runtime']==v['native_runtime'],
              'Original H source/native context differs')
    guard=str(Path(v['runtime_dir'])/'original_preflight_environment_r1.py')
    c.require(prior.get(guard)==ENV_SHA==c.sha(guard),'Exact successful environment guard required')
    for p in (v['source_driver_module'],v['quality_driver_module'],v['static_closure_module']):
        c.require(prior.get(p)==c.sha(p),'Original source absent from visual registration '+p)
    c.require(c.sha(catalogue)==catalogue_sha,'Actual new catalogue pin differs')
    cp=c.read(catalogue);c.require(len(cp['profiles'])==433 and cp['legacy_profile_count']==270,'Actual433 catalogue required')
    work=root/'outputs/MAIN-RAW64-20261007/prepared/clean_missing_cpu_qualification_v1'
    fixtures=dict(old_clean_source0=str(Path(oc['out'])/'clean-quality200/sources/0000.json'),
        new_catalogue=str(Path(catalogue).absolute()),original_source_driver=v['source_driver_module'],
        original_raw_cp=str(Path(oc['S1'])/'export-assets/source_checkpoints/0000.json'))
    inputs={p:c.sha(p) for p in [visual_config,regpath,v['visual_owner_config'],oldcfg,oc['registration'],str(catalogue),str(Path(__file__).absolute())]}
    return dict(root=root,runtime=runtime,v=v,vr=vr,prior=prior,owner=owner,oc=oc,oldreg=oreg,oldprior=oprior,
        oldroot=oldroot,oldcfg=oldcfg,guard=guard,work=work,fixtures=fixtures,sources=sources,inputs=inputs)

def qualify(x):
    work=x['work'];c.require(not work.exists(),'Fresh qualification only; preserve every failed attempt');work.mkdir(parents=True)
    child=None;log=work/'tests.log'
    try:
        c.require(sys.platform.startswith('linux'),'Actual qualification requires original Linux UM environment')
        assets=work/'test_assets.json';c.save(assets,x['fixtures'])
        inputs=c.merge(x['inputs'],{str(assets):c.sha(assets)},{p:c.sha(p) for p in x['fixtures'].values()})
        argv=[x['v']['python'],'-B','-m','unittest','-v','test_clean_missing_r1']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(x['runtime']),MAIN64_CLEAN_TEST_ASSETS=str(assets))
        for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='2'
        def resources():os.sched_setaffinity(0,{16,17});os.setpriority(os.PRIO_PROCESS,0,15)
        c.save(work/'request.json',dict(argv=argv,source_bindings=x['sources'],input_bindings=inputs,
            expected_tests=9,model_calls=0,packet_decodes=0,affinity=[16,17],nice=15))
        with log.open('x',encoding='utf-8') as f:
            child=subprocess.Popen(argv,cwd=x['runtime'],env=env,stdout=f,stderr=subprocess.STDOUT,preexec_fn=resources)
            # Capture the live process where possible, but always wait even on a fast exit.
            try:identity=c.proc(child.pid)
            except FileNotFoundError:identity={'pid':child.pid,'exited_before_identity_capture':True}
            c.save(work/'launch.json',dict(identity=identity,argv=argv,python=x['v']['python']))
            code=child.wait();f.flush();os.fsync(f.fileno())
        c.save(work/'exit.json',dict(pid=child.pid,exit_code=code,process_waited=True,log_sha256=c.sha(log)))
        c.require(code==0 and 'Ran 9 tests' in log.read_text() and log.read_text().rstrip().endswith('OK'),'Actual nine-test qualification failed')
        c.verify(x['sources']);c.verify(inputs)
        outputs={str(p):c.sha(p) for p in (log,work/'request.json',work/'launch.json',work/'exit.json')}
        result=dict(status='MAIN_RAW64_CLEAN_MISSING_CPU_TESTS_PASS',tests_run=9,test_modules=['test_clean_missing_r1'],
            python=x['v']['python'],exit_code=0,process_waited=True,new_model_calls=0,new_packet_decodes=0,
            source_bindings=x['sources'],input_bindings=inputs,outputs=outputs,log=str(log))
        c.save(work/'completion.json',result);return result
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate();child.wait()
        c.save(work/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),
            child_pid=None if child is None else child.pid,new_model_calls=0,new_packet_decodes=0));raise

def model_inputs(x):
    """Read original provenance only; never import a model or read data shards."""
    v=x['v'];root=x['root'];prior=x['prior'];inputs={};models={}
    def take(p,h=None):
        p=str(p);actual=c.sha(p);c.require(h is None or h==actual,'Frozen model/input changed '+p)
        inputs[p]=actual;return c.read(p) if Path(p).suffix=='.json' else None
    cal=take(x['oc']['calibration_registration'],x['oldprior'].get(x['oc']['calibration_registration']))
    identity=cal['identity'];num=take(v['native_qualification'],prior.get(v['native_qualification']))
    c.require(num['frozen_identity']==identity,'Original numerical/model identity differs')
    pathsfile=root/'configs/next_scale_prior_diagnostic.yaml'
    c.require(str(pathsfile) in prior,'Original model paths not sealed');take(pathsfile,prior[str(pathsfile)])
    # The bound original file uses four plain scalar lines in its paths block.
    # Accept that exact grammar; do not import the historical runtime to parse it.
    text=pathsfile.read_text();block=re.search(r'(?m)^paths:\s*\n((?:[ \t]+[^\n]*\n)+)',text)
    c.require(block is not None,'Original plain paths block required')
    paths={}
    for key in ('vae_checkpoint','vae_checkpoint_sha256','var_checkpoint','var_checkpoint_sha256'):
        values=re.findall(r'(?m)^  '+key+r': ([^\s#\x27\"]+)[ \t]*$',block.group(1))
        c.require(len(values)==1,'Original plain model path/hash field differs '+key);paths[key]=values[0]
    for name in ('vae_checkpoint','var_checkpoint'):
        p=str(Path(paths[name]).resolve());h=paths[name+'_sha256'];take(p,h);models[p]=h
    gate=take(identity['decoder_gate'],identity['decoder_gate_sha256']);selection=gate['selection']
    take(selection['checkpoint'],selection['checkpoint_sha256']);models[selection['checkpoint']]=selection['checkpoint_sha256']
    stats=str(root/'outputs/RX-POSTERIOR-STEP1-20260929/calibration_statistics.pt')
    take(stats,identity['calibration_statistics_sha256']);models[stats]=inputs[stats]
    quality=identity['quality'];qc=quality['config']
    for key in ('dino_checkpoint','alexnet_checkpoint'):
        take(qc[key],qc[key+'_sha256']);models[qc[key]]=qc[key+'_sha256']
    linear={p:h for p,h in prior.items() if h==quality['linear_weights_sha256'] and Path(p).suffix=='.pth'}
    c.require(linear,'Original LPIPS linear weight not bound')
    for p,h in linear.items():take(p,h);models[p]=h
    # The unchanged Native factory reads this provenance before loading replay.
    supervisor=str(root/'outputs/UNIFIED-METRICS-20261002/supervisor_registration.json')
    supervisor_data=take(supervisor,prior.get(supervisor))
    replay=str(root/'experiments/unified-metrics-20261002/replay.py')
    c.require(supervisor_data['source_bindings'].get(replay)==prior.get(replay)==c.sha(replay),
              'Original replay loader lacks matching sealed provenance')
    return inputs,models

def request(x):
    v=x['v'];oc=x['oc'];root=x['root'];outroot=root/'outputs/MAIN-RAW64-20261007'
    qpath=str(x['work']/'completion.json');c.require(not (x['work']/'failure.json').exists(),'Qualification failure remains')
    inputs,models=model_inputs(x);inputs=c.merge(x['inputs'],inputs,{qpath:c.sha(qpath)})
    closure=c.module(v['static_closure_module'],'clean_missing_request_static_closure')
    graph=closure.collect_bindings(root,oc['native_runtime'],v['var_source'],v['dino_source'],oc['uep_runtime'])
    for p,h in x['vr']['native_source_bindings'].items():
        c.require(graph.get(p)==h,'Previously sealed native source changed or disappeared '+p)
    sources=c.merge(x['oldreg']['source_bindings'],graph,x['sources'],{x['guard']:ENV_SHA,
        v['static_closure_module']:c.sha(v['static_closure_module']),str(Path(__file__).absolute()):c.sha(__file__)})
    cleanroot=Path(oc['out'])/'clean-quality200';stage=Path(x['oldcfg']).parent/'stages/clean_quality/workers/clean-quality200'
    direct=[cleanroot/'completion.json',stage/'launch.json',stage/'exit_receipt.json',stage/'worker.log',
        Path(oc['S1'])/'export-assets/completion.json',oc['source200'],x['owner']['nvidia_smi']]
    inputs=c.merge(inputs,{str(p):c.sha(p) for p in direct})
    r=dict(schema=c.SCHEMA,root=str(root),execution_dir=str(outroot/'clean_missing_execution_v1'),out=str(outroot/'clean_missing_v1'),
        states=[list(a) for a in c.STATES],source_count=200,images=200,new_packet_decodes=0,selection=False,development_used=False,holdout_used=False,
        python=v['python'],source_driver_module=v['source_driver_module'],quality_module=v['quality_driver_module'],
        closure_module=v['static_closure_module'],environment_module=x['guard'],old_source_config=x['oldcfg'],
        old_clean_completion=str(cleanroot/'completion.json'),old_clean_launch=str(stage/'launch.json'),
        old_clean_exit=str(stage/'exit_receipt.json'),old_clean_log=str(stage/'worker.log'),native_qualification=v['native_qualification'],
        new_profiles=x['fixtures']['new_catalogue'],prepared_qualification=qpath,var_source=v['var_source'],dino_source=v['dino_source'],
        visual_lock=x['owner']['visual_lock_path'],nvidia_smi=x['owner']['nvidia_smi'],
        resources=dict(gpu_device=0,threads=6,interop_threads=2,affinity=[4,5,6,7,8,9],nice=15),max_seconds=3600,
        deadline_unix=DEADLINE,stop_files=[str(outroot/'STOP'),str(x['oldroot']/'STOP')],
        source_bindings=sources,input_bindings=inputs,native_source_bindings=graph,model_bindings=models)
    ctx=c.inspect(r);r['input_bindings']=ctx['inputs'];r['source_bindings']=ctx['sources']
    target=outroot/'prepared/clean_missing_request_v1.json';c.save(target,r)
    inventory=target.with_name('clean_missing_record_paths_v1.json');c.save(inventory,dict(source_ids=ctx['source_ids'],records=ctx['records'],request_sha256=c.sha(target)))
    return dict(status='CLEAN_MISSING_REQUEST_READY_NOT_REGISTERED',request=str(target),request_sha256=c.sha(target),
        record_paths=str(inventory),sources=len(r['source_bindings']),inputs=len(r['input_bindings']),images=200,new_packet_decodes=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['qualify','request'],required=True)
    for n in ('root','runtime','visual-config','catalogue','catalogue-sha'):p.add_argument('--'+n,required=True)
    a=p.parse_args();x=base(a.root,a.runtime,a.visual_config,a.catalogue,a.catalogue_sha)
    result=qualify(x) if a.stage=='qualify' else request(x)
    print(json.dumps({'status':result['status']} if a.stage=='qualify' else result,sort_keys=True))
