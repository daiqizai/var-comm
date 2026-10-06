"""One separately registered H18 metric GPU job; no PHY, selection or inference of missing images."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import h_development_render_driver as u

DONE='H_DEVELOPMENT_METRICS_COMPLETE'
SCOPE='H18_UNIFIED_METRICS_ONLY'
STOP=False
METRIC_PATHS=('suite_module','validation_module','replay_module','metrics_registration','modelmanifest',
    'models_qualification','independent_binding','convnext_weights')
REQUIRED=('owner_module','wait_module','cpu_driver_module','render_driver_module','metric_adapter_module',
    'metric_driver_module','metric_assets','visual_owner_config','budget_registration','protocol',*METRIC_PATHS)
read,sha,require,bind,merge,verify,save,module=u.read,u.sha,u.require,u.bind,u.merge,u.verify,u.save,u.module


def stop(*_):
    global STOP
    STOP=True


def normal_render_closed(spec):
    """Verify the historical exact graph; do not enumerate newly added Python retroactively."""
    require(set(spec)=={'config','owner_config','registration','launch','completion'},'Exact normal render batch required')
    cfg,reg=read(spec['config']),read(spec['registration'])
    bound=merge(reg['source_bindings'],reg['input_bindings']);verify(bound)
    require(cfg['registration']==spec['registration'] and cfg['visual_owner_config']==spec['owner_config'],
        'Render configuration lineage differs')
    for p in (spec['config'],spec['owner_config'],cfg['owner_module'],cfg['wait_module'],cfg['cpu_driver_module'],cfg['rx_adapter_module']):
        require(bound.get(p)==sha(p),'Actual render dependency unbound: '+p)
    owner=module(cfg['owner_module'],'metric_normal_owner');wait=module(cfg['wait_module'],'metric_normal_wait')
    cpu=module(cfg['cpu_driver_module'],'metric_normal_cpu');rx=module(cfg['rx_adapter_module'],'metric_normal_rx')
    expected=dict(status=u.DONE,source_count=100,frame_count=5400,H_policy_snr_points=18,MAIN_frames=0,
        new_packet_decodes=0,policy_selection=False,development_used=True,holdout_used=False,
        images_scored=True,source_decode_complete=True,arithmetic_source_decode_complete=True,unified_neural_metrics_run=False)
    closed=cpu.closed_batch(owner,wait,dict(config=spec['owner_config'],registration=spec['registration'],
        launch=spec['launch'],completion=spec['completion']),cfg['owner_module'],expected,owner.raw_process_state)
    done=closed['done'];old=closed['owner']
    require(len(old['stages'])==1 and old['stages'][0]['id']=='render' and len(old['stages'][0]['jobs'])==1
        and old['stages'][0]['jobs'][0]['id']=='development_render'
        and old['stages'][0]['jobs'][0]['argv'][1:]==['-B',cfg['visual_driver_module'],'--config',spec['config']],
        'Wrong completed rendering entry')
    graph=read(cfg['visual_source_closure'])
    require(bound.get(cfg['visual_source_closure'])==sha(cfg['visual_source_closure'])
        and graph['status']=='EXACT_SOURCE_CLOSURE_MATCH'
        and graph['source_bindings']==done['visual_source_bindings'],'Historical executed visual closure differs')
    verify(graph['source_bindings']);verify(done['outputs'])
    group=rx.verify_cpu_closed(cfg['cpu_batch'],cpu,owner,wait)
    before=owner.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
    u.assert_budget(before,group['before'],group)
    require(before==done['budget_before']==done['budget_after']==closed['owner_done']['budget'],
        'Closed visual and current paid budgets differ')
    require(done['source_ids']==group['source_ids'],'Rendered population changed')
    inputs=merge(bound,closed['bindings'],group['bindings'],done['outputs'],bind(spec.values()))
    return dict(owner=owner,wait=wait,cpu=cpu,closed=closed,render_cfg=cfg,group=group,old=old,before=before,
        bindings=inputs,population=group['population'],manifest=group['context']['manifest'],
        assets=group['context']['assets'],schedule=group['schedule'],source_ids=group['source_ids'])


def metric_assets(path):
    value=read(path)
    require(value['status']=='H_ORIGINAL_METRIC_ASSETS_BOUND' and set(value['paths'])==set(METRIC_PATHS),
        'Explicit original metric asset manifest required')
    bound=merge(value['source_bindings'],value['input_bindings']);verify(bound)
    for p in value['paths'].values():require(bound.get(p)==sha(p),'Metric asset path is not sealed: '+p)
    r=read(value['paths']['metrics_registration']);q=read(value['paths']['models_qualification'])
    require(r['synthetic'] is False and q['status']=='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS'
        and r['modelmanifest_sha256']==q['modelmanifest_sha256']==sha(value['paths']['modelmanifest']),
        'Original real metric qualification differs')
    flags=r['numerical_runtime']
    require(flags['threads']==6 and flags['interop_threads']==2 and flags['deterministic'] is True
        and not flags['matmul_tf32'] and not flags['cudnn_tf32'] and flags['precision']=='highest',
        'Original metric FP32 flags differ')
    independent=read(value['paths']['independent_binding'])
    require(independent['status']=='FROZEN_IDENTITY_CLARIFICATION_BEFORE_ANY_H_C_VALIDATION'
        and independent['used_for_selection'] is False
        and independent['reference_implementation_sha256']==sha(value['paths']['validation_module'])
        and independent['weights_sha256']==sha(value['paths']['convnext_weights']), 'Independent classifier binding differs')
    return value,bound,flags


def load_registered(path):
    cfg=read(path);reg=read(cfg['registration'])
    require(cfg['schema']=='H_DEVELOPMENT_METRICS_CONFIG_V1' and reg['status']=='H_EXECUTION_REVISION_REGISTERED'
        and reg['branch']=='H' and reg['source_stage_scope']==SCOPE and reg['allowed_stage_ids']==['development'],
        'Independent metric-only registration required')
    bound=merge(reg['source_bindings'],reg['input_bindings']);verify(bound)
    for p in (str(Path(path).absolute()),str(Path(__file__).absolute()),*[cfg[k] for k in REQUIRED]):
        require(bound.get(p)==sha(p),'Metric implementation/input unbound: '+p)
    require(cfg['metric_driver_module']==str(Path(__file__).absolute()),'Registered metric entry differs')
    for p in cfg['render_batch'].values():require(bound.get(p)==sha(p),'Render predecessor input unbound')
    ctx=normal_render_closed(cfg['render_batch']);prior=ctx['render_cfg']
    for k in ('root','H_out','owner_module','wait_module','cpu_driver_module','ledger','budget_registration','phase_limits','stop_file','protocol'):
        require(cfg[k]==prior[k],'Original execution input changed: '+k)
    require(cfg['render_driver_module']==prior['visual_driver_module'],'Actual completed renderer differs')
    require(ctx['before']==reg['budget_before'],'Registration/current budget differs')
    ma,mb,flags=metric_assets(cfg['metric_assets'])
    require(all(cfg[k]==ma['paths'][k] for k in METRIC_PATHS)
        and all(bound.get(p)==s for p,s in mb.items()),'Metric dependency closure not in registration')
    require(read(cfg['independent_binding'])['original_H_protocol_sha256']==sha(cfg['protocol']),
        'Independent metric binding is not the original H protocol')
    require(Path(cfg['H_out']) in Path(cfg['out']).parents and cfg['out']!=prior['out']
        and Path(cfg['out']) not in Path(prior['out']).parents and Path(prior['out']) not in Path(cfg['out']).parents,
        'Metric output overlaps source images')
    require(type(cfg['max_seconds']) is int and 0<cfg['max_seconds']<=86400
        and cfg['overall_deadline_unix']==u.DEADLINE,'Original finite deadline required')
    require(not Path(cfg['stop_file']).exists() and not (Path(cfg['out'])/'STOP').exists(),'STOP blocks metrics')
    adapter=module(cfg['metric_adapter_module'],'h18_registered_metric_adapter')
    require(sha(cfg['replay_module'])==adapter.REPLAY_SHA and sha(cfg['convnext_weights'])==adapter.CONVNEXT_SHA,
        'Original mismatch/classifier identity differs')
    ctx.update(cfg=cfg,reg=reg,bound=bound,adapter=adapter,metric_flags=flags)
    return ctx


def gpu_admission(ctx,path):
    c=ctx['cfg'];a=ctx['owner'];oc=read(c['visual_owner_config']);a.validate_config(oc,ctx['reg'],sha(c['visual_owner_config']))
    require(len(oc['stages'])==1 and oc['stages'][0]['id']=='development'
        and oc['stages'][0]['resource']=='gpu' and len(oc['stages'][0]['jobs'])==1,'One metric GPU job required')
    job=oc['stages'][0]['jobs'][0];argv=job['argv']
    require(job['id']=='development_metrics' and argv[1:]==['-B',str(Path(__file__).absolute()),'--config',str(Path(path).absolute())]
        and job['out']==c['out'],'Current GPU command differs')
    base=Path(oc['owner_out']);ip=base/'owner_identity.json';identity=read(ip)
    require(int(identity['pid'])==os.getppid() and a.same_identity(identity,a.identity(os.getppid()))
        and identity['config_sha256']==sha(c['visual_owner_config']) and identity['registration_sha256']==sha(c['registration'])
        and not (base/'failure.json').exists(),'Live exclusive metric parent absent')
    lp=base/'stages/development/workers/development_metrics/launch.json';started=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-started<10 and a.same_identity(identity,a.identity(os.getppid())),'Metric worker not sealed');time.sleep(.1)
    launch=read(lp);current=a.identity(os.getpid());gp=base/'stages/development/gpu_admission.json';gate=read(gp)
    require(a.same_identity(current,launch['identity']) and launch['argv']==argv and launch['resource']=='gpu'
        and launch['registration_sha256']==sha(c['registration']) and launch['threads']==oc['gpu_threads']==6
        and set(launch['affinity'])==set(oc['gpu_affinity'])==set(os.sched_getaffinity(0))
        and os.environ.get('CUDA_VISIBLE_DEVICES')==str(oc['gpu_device']), 'Metric worker resource/identity differs')
    require(gate['status']=='GPU_IDLE_CONFIRMED' and gate['registration_sha256']==sha(c['registration']), 'Exclusive GPU admission absent')
    return dict(owner_identity=identity,worker_identity=current,bindings=bind((ip,lp,gp)))


class GPUHealth:
    """Same original 86C/three-hot-check stop; foreign compute process never admitted."""
    def __init__(self):self.hot=0;self.next=0.;self.last=None
    def check(self,force=False):
        if not force and time.monotonic()<self.next:return
        pids=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        require(all(int(v.strip())==os.getpid() for v in pids.splitlines() if v.strip()),'Foreign GPU compute process appeared')
        values=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=temperature.gpu,clocks_event_reasons.hw_thermal_slowdown',
            '--format=csv,noheader,nounits'],text=True).strip().split(',')
        temperature=int(values[0]);active=values[1].strip().lower()=='active'
        self.hot=self.hot+1 if temperature>=86 or active else 0
        self.last=dict(temperature=temperature,hardware_thermal_slowdown=active,consecutive_hot=self.hot)
        require(self.hot<3,'Original sustained GPU thermal threshold');self.next=time.monotonic()+5


class GuardedBackend:
    def __init__(self,backend,health,reference_boundary=None):
        self.backend=backend;self.health=health;self.identity=backend.identity;self.reference_boundary=reference_boundary
    def prepare(self,*args):
        if self.reference_boundary is not None:self.reference_boundary()
        self.health.check();return self.backend.prepare(*args)
    def score(self,*args):self.health.check();return self.backend.score(*args)


def build_backend(ctx):
    cfg=ctx['cfg'];adapter=ctx['adapter']
    for p in (str(Path(cfg['suite_module']).parent),str(Path(cfg['replay_module']).parent),str(Path(cfg['root'])/'src')):sys.path.insert(0,p)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    suite_module=module(cfg['suite_module'],'h18_frozen_metric_suite');import torch
    suite=suite_module.load_suite(Path(cfg['root']),'cuda:0')
    require(suite[6]==ctx['metric_flags'],'Loaded original metric flags differ')
    require(all(ctx['bound'].get(p)==s for p,s in suite[5].items()),'Loaded original metric dependencies not preregistered')
    validation=module(cfg['validation_module'],'h18_frozen_convnext')
    policy=ctx['group']['cfg']['finalized']
    require(ctx['bound'].get(policy)==sha(policy),'Frozen calibration policy not bound')
    classifier=validation.ConvNeXtValidation(cfg['convnext_weights'],adapter.CONVNEXT_SHA,device='cuda:0',
        stage='development',policy_path=policy,policy_sha256=sha(policy))
    return adapter.SuiteBackend(suite,classifier,torch,suite_module.numeric_flags)


def write_source(out,index,sid,rows,cost,inputs,regsha,evaluator):
    out=Path(out);rp=out/'sources'/f'{index:04d}.json';cp=out/'source_checkpoints'/f'{index:04d}.json'
    require(len(rows)==54 and all(r['source_index']==index and r['source_id']==sid for r in rows),'Wrong source metric coverage')
    save(rp,rows);outputs=bind((rp,))
    save(cp,dict(status='H_DEVELOPMENT_METRIC_SOURCE_COMPLETE',source_index=index,source_id=sid,frame_count=54,
        registration_sha256=regsha,metric_evaluator_identity=evaluator,outputs=outputs,input_bindings=inputs,cost=cost,new_packet_decodes=0))
    return merge(outputs,bind((cp,)))


def claim_output(out,regsha):
    out=Path(out)
    if out.exists():require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'Prior metric attempt preserved; no retry')
    else:out.mkdir(parents=True)
    save(out/'attempt.json',dict(status='H_DEVELOPMENT_METRICS_ATTEMPT',registration_sha256=regsha,
        pid=os.getpid(),started_unix=time.time(),new_packet_decodes=0))


def run(path):
    ctx=load_registered(path);cfg=ctx['cfg'];out=Path(cfg['out']);regsha=sha(cfg['registration']);claim_output(out,regsha)
    started=time.monotonic();wall=time.time();adapter=ctx['adapter'];health=GPUHealth()
    try:
        supervision=gpu_admission(ctx,path)
        def boundary():
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP at metric source boundary')
            elapsed=time.monotonic()-started
            require(elapsed<cfg['max_seconds'] and max(time.time(),wall+elapsed)<u.DEADLINE,'Metric wall limit')
            require(ctx['owner'].same_identity(supervision['owner_identity'],ctx['owner'].identity(os.getppid())),'Exclusive metric owner disappeared')
            health.check(force=True)
        boundary();backend=build_backend(ctx);guarded=GuardedBackend(backend,health,boundary)
        replay=module(cfg['replay_module'],'h18_frozen_mismatch');mismatch=adapter.frozen_mismatch(replay,ctx['source_ids'])
        def load(i):return adapter.load_source(i,ctx['population'],ctx['manifest'],ctx['assets'],ctx['closed']['done'],
            ctx['render_cfg']['out'],ctx['schedule'])
        # Only targets survive this pass. No reconstruction archive from one
        # source is retained while loading the next, and no new population is read.
        sources=[]
        for i in range(100):
            boundary();item=load(i)
            sources.append({k:item[k] for k in ('source_index','source_id','target','reference_sha256')});del item
        references=adapter.reference_table(sources,guarded,mismatch);del sources
        meta=out/'metric_identity.json';save(meta,dict(metadata=backend.metadata,evaluator_identity=backend.identity,
            mismatch=mismatch,metric_batch_size=1,selection=False,bootstrap_not_run=True))
        outputs=bind((meta,));allrows=[];costs=[]
        for i,sid in enumerate(ctx['source_ids']):
            boundary();item=load(i);rows,cost=adapter.score_source(item,guarded,references,mismatch,ctx['schedule'])
            trace=Path(ctx['group']['cfg']['out'])/f'worker_{i%2}'/'traces'/f'{i:04d}.json'
            require(ctx['group']['done']['outputs'].get(str(trace))==sha(trace),'Post-score wire trace unsealed')
            rows=adapter.attach_wire_accounting(rows,read(trace));inputs=merge(item['input_bindings'],bind((trace,)))
            outputs.update(write_source(out,i,sid,rows,cost,inputs,regsha,backend.identity));allrows.extend(rows);costs.append(cost)
            u.progress(out/'status.json',dict(status='RUNNING',completed_sources=i+1,total_sources=100,completed_frames=len(allrows),
                elapsed_seconds=time.monotonic()-started,registration_sha256=regsha));del item
        require(len(allrows)==5400,'Incomplete metric population');boundary();backend.check()
        after=ctx['owner'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
        u.assert_budget(after,ctx['before'],ctx['group'])
        fp=out/'frame_metrics.json';save(fp,allrows);outputs.update(bind((fp,)))
        csvpath=out/'frame_metrics.csv';fields=sorted(k for k,v in allrows[0].items() if v is None or isinstance(v,(str,int,float,bool)))
        with csvpath.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in allrows)
        outputs.update(bind((csvpath,)));costpath=out/'metric_cost_counts.json'
        save(costpath,dict(frame_count=5400,unique_metric_pairs=sum(c['unique_metric_pairs'] for c in costs),
            exact_duplicate_metric_reuses=sum(c['exact_duplicate_metric_reuses'] for c in costs),metric_batch_size=1,
            online_latency_measured=False,new_packet_decodes=0));outputs.update(bind((costpath,)))
        verify(ctx['bound']);verify(outputs)
        done=dict(status=DONE,registration_sha256=regsha,config_sha256=sha(path),source_count=100,frame_count=5400,
            H_policy_snr_points=18,MAIN_frames=0,source_ids=ctx['source_ids'],noise_seeds=[6201,6202,6203],
            outputs=outputs,input_bindings=ctx['reg']['input_bindings'],source_bindings=ctx['reg']['source_bindings'],
            render_completion_sha256=sha(cfg['render_batch']['completion']),metric_evaluator_identity=backend.identity,
            metric_metadata=backend.metadata,numerical_runtime=ctx['metric_flags'],metric_batch_size=1,supervision=supervision,
            budget_before=ctx['before'],budget_after=after,new_packet_decodes=0,policy_selection=False,development_used=True,
            holdout_used=False,GPU_used=True,unified_neural_metrics_run=True,statistical_aggregation_run=False,
            MAIN_complete=False,P_complete=False,overall_development_complete=False,H_full_delivery_claimed=False,
            online_timing_measured=False,elapsed_seconds=time.monotonic()-started)
        save(out/'completion.json',done);return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,
            traceback=traceback.format_exc(),new_packet_decodes=0));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);print(run(args.config)['status'])


if __name__=='__main__':main()
