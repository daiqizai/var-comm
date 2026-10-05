"""Seal one CPU-only original-calibration1000 asset export; never launch it.

The prior initial200 selector must have its own normal closed owner receipt.
Rendering's historical failed owner remains a failure; its independent closeout
is retained transitively through the selected, sealed evidence. This operation
does not read tensors, encode sources, decode packets, or construct models.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

DEADLINE=1791564605.9549868
PHASES={'qualification':2000,'coarse':49152,'refine':12288,'initial_true200':20000,
        'whole_calibration':48000,'partial_calibration':36000,'development':13200,'engineering_reserve':19360}
COUNTS=dict.fromkeys(PHASES,0)
COUNTS.update(qualification=1608,coarse=49152,initial_true200=19200)
INPUT_NAMES=('protocol','calibration_registration','image_manifest','s1_completion','s1_registration',
             's1_assets_completion','h_source_registration','h_source_config','h_source_completion')


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def bind(paths):return {str(p):sha(p) for p in paths}
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Changed immutable binding: '+p)
def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting binding: '+p);result[p]=s
    return result
def pin(record):
    p=Path(record['path']);require(p.is_absolute() and sha(p)==record['sha256'],'Pinned input changed: '+str(p));return p
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def check_budget(value):
    require(value.get('created') is True and value.get('charged')==69960
            and value.get('phase_charged')==COUNTS and value.get('unresolved')==0
            and value.get('failed')==0 and value.get('development_remaining')==13200,
            'Original quiescent 69960-packet budget differs')


def verify_selection(owner,prior,paths,state_reader):
    """Prove a normal selector completion and all actual process exits."""
    cp,rp,lp=paths['selection_owner_config'],paths['selection_registration'],paths['selection_launch']
    cfg,reg,launch=read(cp),read(rp),read(lp)
    owner.validate_config(cfg,reg,sha(cp))
    require(cfg['registration']==str(rp) and reg['allowed_stage_ids']==['freeze']
            and cfg['phase_limits']==PHASES and cfg['overall_deadline_unix']==DEADLINE,
            'Wrong selection scope/budget/deadline')
    require(len(cfg['stages'])==1,'Expected one selector stage')
    stage=cfg['stages'][0]
    require(stage['id']=='freeze' and stage['resource']=='cpu' and len(stage['jobs'])==1,'Wrong selector job scope')
    job=stage['jobs'][0]
    require(job['id']=='initial_selection' and job['completion']==str(paths['selection_completion'])
            and job['argv'].count('--config')==1
            and job['argv'][job['argv'].index('--config')+1]==str(paths['selection_config']),
            'Selector configuration/scientific receipt differs')
    sc=read(paths['selection_config'])
    require(sc['registration']==str(rp) and sc['owner_config']==str(cp)
            and sc['out']==job['out'] and sc['root']==cfg['root'],'Selector config identity differs')
    base=Path(cfg['owner_out'])
    require(paths['selection_owner_completion']==base/'completion.json','Wrong owner completion path')
    for p in (Path(cfg['out'])/'STOP',base/'STOP',Path(job['out'])/'STOP'):
        require(not p.exists(),'STOP blocks new source batch: '+str(p))
    require(not (base/'failure.json').exists() and not (Path(job['out'])/'failure.json').exists(),
            'Original selector failure must not be ignored')
    identity=launch['identity'];expected=[launch['argv'][0],'-B',str(Path(owner.__file__).resolve()),'--config',str(cp)]
    require(identity['argv']==launch['argv']==expected and launch['registration_sha256']==sha(rp)
            and launch['owner_config_sha256']==sha(cp),'Selection owner launch differs')
    ip=base/'owner_identity.json';record=read(ip)
    require(all(record[k]==identity[k] for k in ('pid','uid','start_ticks','argv'))
            and record['registration_sha256']==sha(rp) and record['config_sha256']==sha(cp),
            'Selection owner identity differs')
    require(prior.exited(identity,state_reader),'Selection owner is live or unreaped')
    closed=prior.verify_batch(owner,cp,rp,identity['uid'],state_reader)
    done=read(paths['selection_completion'])
    expected_fields={'status':'H_INITIAL_TRUE200_WHOLE_SELECTION_COMPLETE','source_count':200,
        'measured_frames':9600,'candidate_count':16,'selected_count':8,'partial_reselected':False,
        'new_packet_decodes':0,'new_visual_inference':0,'GPU_used':False,'development_used':False,
        'holdout_used':False,'full1000_calibration_complete':False,'original_render_owner_success':False}
    for k,v in expected_fields.items():require(done.get(k)==v,'Selection scope differs: '+k)
    require(done['registration_sha256']==sha(rp),'Selection registration differs')
    check_budget(done['budget']);check_budget(read(paths['selection_owner_completion'])['budget'])
    verify(done['outputs']);verify(reg['source_bindings']);verify(reg['input_bindings'])
    require(reg['input_bindings'].get(str(paths['selection_config']))==sha(paths['selection_config']),
            'Selector configuration was not registered')
    bindings=merge(closed['bindings'],bind((cp,rp,lp,ip,paths['selection_config'],paths['selection_completion'])))
    return dict(owner_config=cfg,registration=reg,completion=done,
                closure=dict(closed,bindings=bindings,owner_identity=identity,owner_exited=True,
                             original_render_owner_success=False))


def collect_shard_bindings(plan):
    """Hash exactly original calibration shards/sidecars; never load arrays."""
    values={}
    for row in plan['records']:
        for name in ('latent','pixel'):
            p=Path(row[name+'_shard']);s=row[name+'_sha256']
            require(p.is_absolute(),'Original shard must be absolute')
            require(str(p) not in values or values[str(p)]==s,'Conflicting original shard SHA')
            values[str(p)]=s
    verify(values)
    for p in sorted({Path(row['latent_shard']).with_suffix('.json') for row in plan['records']}):
        require(read(p)['sha256']==values[str(p.with_suffix('.pt'))],'Latent sidecar does not seal original shard')
        values[str(p)]=sha(p)
    return values


def predecessor(request,state_reader=None):
    names=('stage_owner','closure_module','asset_module','selection_owner_config','selection_registration',
           'selection_launch','selection_owner_completion','selection_config','selection_completion',*INPUT_NAMES)
    paths={k:pin(request[k]) for k in names}
    owner=module(paths['stage_owner'],'full1000_registered_stage_owner')
    prior=module(paths['closure_module'],'full1000_closed_batch_audit')
    adapter=module(paths['asset_module'],'full1000_original_asset_adapter')
    selected=verify_selection(owner,prior,paths,state_reader or owner.raw_process_state)
    old=selected['owner_config'];hcfg=read(paths['h_source_config']);hreg=read(paths['h_source_registration'])
    require(hcfg['root']==old['root'] and hcfg['out']==old['out']
            and hcfg['registration']==str(paths['h_source_registration'])
            and hcfg['calibration_registration']==str(paths['calibration_registration'])
            and Path(hcfg['S1'])==paths['s1_completion'].parent,'Original source config differs')
    paths['s1_source_ids']=Path(hcfg['source200'])
    require(hreg['input_bindings'].get(str(paths['h_source_config']))==sha(paths['h_source_config'])
            and hreg['input_bindings'].get(str(paths['protocol']))==sha(paths['protocol'])
            and hreg['input_bindings'].get(str(paths['s1_source_ids']))==sha(paths['s1_source_ids']),
            'Source200/protocol/config not sealed by original H registration')
    sreg=read(paths['s1_registration']);s1=read(paths['s1_completion'])
    require(s1['status']=='S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION'
            and s1['registration_sha256']==sha(paths['s1_registration'])
            and s1['outputs'].get(str(paths['s1_assets_completion']))==sha(paths['s1_assets_completion'])
            and old['S1_gate']['path']==str(paths['s1_completion'])
            and old['S1_gate']['sha256']==sha(paths['s1_completion']), 'S1 original completion differs')
    for reg in (hreg,sreg):verify(reg['source_bindings']);verify(reg['input_bindings'])
    verify(s1['outputs'])
    require(sha(paths['image_manifest'])==adapter.PIXEL_MANIFEST_SHA,'Original pixel manifest changed')
    for name in ('image_manifest','calibration_registration','s1_source_ids'):
        require(sreg['input_bindings'].get(str(paths[name]))==sha(paths[name]),
                'Original S1 registration does not seal '+name)
    cal=read(paths['calibration_registration'])
    plan=adapter.build_population_plan(cal,read(paths['image_manifest']),str(paths['image_manifest']),
        read(paths['s1_source_ids'])['source_ids'],cal['identity'])
    shard_bindings=collect_shard_bindings(plan)
    # Constructor validates closed receipts/output SHAs but never reads source arrays.
    reuse=adapter.Source200ReuseReader(str(paths['s1_assets_completion']),str(paths['h_source_completion']),
        sha(paths['s1_registration']),sha(paths['h_source_registration']))
    budget=owner.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],PHASES,quiescent=True)
    check_budget(budget)
    require(sha(old['budget_registration'])==old['budget_registration_sha256'],'Budget registration changed')
    sources=merge(selected['registration']['source_bindings'],hreg['source_bindings'],sreg['source_bindings'])
    for name in ('stage_owner','closure_module'):
        require(sources.get(str(paths[name]))==sha(paths[name]),'Unbound original runtime: '+name)
    inputs=merge(selected['registration']['input_bindings'],hreg['input_bindings'],sreg['input_bindings'],
        selected['closure']['bindings'],selected['completion']['outputs'],s1['outputs'],reuse.input_bindings,
        shard_bindings,bind(paths.values()),bind((Path(old['budget_registration']),)))
    return dict(owner=owner,adapter=adapter,paths=paths,old=old,closure=selected['closure'],
                plan=plan,budget=budget,sources=sources,inputs=inputs)


def make_configs(request,ctx):
    execution,out=Path(request['execution_dir']),Path(request['assets_out'])
    old=ctx['old'];rp=execution/'execution_registration.json';cp=execution/'assets_config.json'
    cfg=dict(schema='H_FULL1000_ASSET_ADAPTER_CONFIG_V1',registration=str(rp),root=old['root'],out=str(out),
        stop_file=str(Path(old['out'])/'STOP'),max_seconds=request['max_seconds'],
        expected_visual_identity=ctx['plan']['visual_identity'])
    cfg.update({k:str(ctx['paths'][k]) for k in (*INPUT_NAMES,'s1_source_ids')})
    oc=copy.deepcopy(old);oc.update(owner_out=str(execution),registration=str(rp))
    job=dict(id='full1000_assets',out=str(out),cwd=old['root'],
        argv=[request['python'],'-B',str(ctx['paths']['asset_module']),'--config',str(cp)],
        completion=str(out/'completion.json'),accepted_statuses=[ctx['adapter'].DONE],
        receipt_expect=dict(source_count=1000,reused_codec_sources=200,pending_source_encoding_count=800,
            source1000_codec_complete=False,new_source_encoding=False,GPU_used=False,new_visual_inference=0,
            new_packet_decodes=0,development_used=False,holdout_used=False,full1000_calibration_complete=False))
    oc['stages']=[dict(id='source',resource='cpu',requires=[],max_seconds=request['max_seconds'],jobs=[job])]
    return cfg,oc


def register(request_path):
    request_path=Path(request_path).resolve();r=read(request_path)
    require(r['schema']=='H_FULL1000_ASSETS_REGISTRATION_REQUEST_V1','Wrong request schema')
    require(sys.platform.startswith('linux'),'Actual registration requires Linux process closure')
    require(time.time()<DEADLINE,'Original H deadline expired')
    ctx=predecessor(r);a=ctx['owner'];h=Path(ctx['old']['out'])
    qp=pin(r['prepared_qualification']);qual=read(qp)
    require(qual['status']=='H_FULL1000_ASSETS_CPU_QUALIFICATION_PASS' and qual['GPU_used'] is False
            and qual['new_packet_decodes']==0 and qual['results'] and all(x['exit_code']==0 for x in qual['results']),
            'CPU helper/adapter qualification did not pass')
    verify(qual['source_bindings'])
    for p in (Path(__file__).resolve(),ctx['paths']['asset_module']):
        require(qual['source_bindings'].get(str(p))==sha(p),'Qualification misses current source: '+str(p))
    execution,out=Path(r['execution_dir']),Path(r['assets_out'])
    for p in (execution,out):require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent H path required: '+str(p))
    require(execution!=out and execution not in out.parents and out not in execution.parents,'Outputs overlap')
    require(Path(r['python']).is_absolute() and Path(r['python']).is_file()
            and type(r['max_seconds']) is int and 0<r['max_seconds']<=3600,'Bad interpreter/finite CPU cap')
    cfg,oc=make_configs(r,ctx)
    sources=merge(ctx['sources'],qual['source_bindings'],bind((Path(__file__).resolve(),ctx['paths']['asset_module'])))
    inputs=merge(ctx['inputs'],bind((request_path,qp)))
    verify(sources);verify(inputs);execution.mkdir()
    try:
        closure=execution/'predecessor_closure.json';save(closure,dict(ctx['closure'],budget=ctx['budget']))
        inventory=execution/'calibration_asset_inventory.json'
        save(inventory,dict(status='ORIGINAL_CALIBRATION_METADATA_PINNED_NOT_EXPORTED',source_count=1000,
            source_ids=ctx['plan']['source_ids'],records=ctx['plan']['records'],
            reusable_source_count_pending_validation=200,pending_source_encoding_count=800,
            source1000_codec_complete=False,new_packet_decodes=0,new_visual_inference=0))
        cp,op,rp=execution/'assets_config.json',execution/'owner_config.json',execution/'execution_registration.json'
        save(cp,cfg);save(op,oc);inputs=merge(inputs,bind((closure,inventory,cp,op)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,
            source_stage_scope='FULL1000_CPU_ASSETS_ONLY',owner_config_sha256=sha(op),allowed_stage_ids=['source'],
            phase_limits=PHASES,source_bindings=sources,input_bindings=inputs,budget_before=ctx['budget'],
            budget_registration_sha256=oc['budget_registration_sha256'],scientific_protocol_sha256=sha(cfg['protocol']),
            predecessor_selection_completion_sha256=sha(ctx['paths']['selection_completion']),
            predecessor_owner_completion_sha256=sha(ctx['paths']['selection_owner_completion']),
            original_render_owner_success=False,source_count=1000,reused_codec_sources_pending_validation=200,
            pending_source_encoding_count=800,source1000_codec_complete=False,full1000_calibration_complete=False,
            GPU_jobs=0,new_packet_decodes=0,new_visual_inference=0,new_source_encoding=False,
            development_used=False,holdout_started=False,C_started=False,H_full_delivery_claimed=False,
            future_stage_automatic=False,scientific_protocol_modified=False,created_unix=time.time())
        a.validate_config(oc,reg,sha(op));verify(sources);verify(inputs)
        require(not (h/'STOP').exists() and not (execution/'STOP').exists() and time.time()<DEADLINE,
                'STOP/deadline changed during registration')
        require(a.budget_snapshot(oc['budget_path'],oc['budget_registration_sha256'],PHASES,quiescent=True)==ctx['budget'],
                'Shared budget changed before sealing')
        save(rp,reg)
        # Complete registered adapter gate, still without loading a tensor/model.
        ctx['adapter'].load_registered(cp)
        result=dict(status='H_FULL1000_CPU_ASSETS_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(rp),
            owner_config_sha256=sha(op),outputs=bind((closure,inventory,cp,op,rp)),workers_started=False,
            GPU_used=False,new_packet_decodes=0,new_visual_inference=0,source1000_codec_complete=False,
            future_stage_automatic=False)
        save(execution/'registration_completion.json',result);return result
    except BaseException as error:
        save(execution/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),
            traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);args=p.parse_args()
    result=register(args.request);print(json.dumps({'status':result['status'],'registration_sha256':result['registration_sha256']}))


if __name__=='__main__':main()
