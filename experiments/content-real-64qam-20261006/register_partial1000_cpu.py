"""Register only six frozen partial1000 CPU policies after whole1000 closes.

Read-only predecessor checks precede every new file. No PHY, GPU, policy search,
worker launch, quota transfer, or implicit recovery is performed by this tool.
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
REQUEST_NAMES=('whole_registrar_module','whole_request','whole_config','whole_registration',
    'whole_owner_config','whole_launch','whole_owner_completion','whole_completion','prepared_qualification')


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
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
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def check_closed_budget(before,owner_budget,science,t):
    """Whole can use fewer body calls after bad headers; never refund any call."""
    t.check_budget(before)
    ledger=science['ledger'];audit=science['ledger_audit'];counts=ledger['phase_charged']
    require(ledger['schema']=='PACKET_PRECHARGE_V1' and ledger['branch']=='H' and ledger['limits']==t.PHASES
        and ledger['total_cap']==200000 and ledger['unresolved']==0 and ledger['development_reserve_transferable'] is False,
        'Whole scientific ledger is not closed under original budget')
    require(set(counts)==set(t.PHASES) and all(counts[p]==before['phase_charged'][p] for p in counts if p!='whole_calibration')
        and 24000<=counts['whole_calibration']<=48000,'Only the original whole phase may have advanced')
    grouped=ledger['counts'];observed=dict.fromkeys(t.PHASES,0);header=body=0;keys=set()
    for row in grouped:
        k=(row['phase'],row['kind'],row['status']);n=row['count']
        require(k not in keys and row['phase'] in t.PHASES and row['kind'] in ('header','body')
            and row['status']=='COMPLETE' and type(n) is int and n>0,'Duplicate/unresolved/invalid ledger count')
        keys.add(k);observed[row['phase']]+=n
        if row['phase']=='whole_calibration':
            if row['kind']=='header':header+=n
            else:body+=n
    require(observed==counts and header==24000 and 0<=body<=24000 and header+body==counts['whole_calibration'],
        'Whole header/body payment count differs')
    require(ledger['charged']==sum(counts.values()) and ledger['phase_remaining']=={p:t.PHASES[p]-counts[p] for p in counts},
        'Whole ledger total/remainders differ')
    require(audit['status']=='FULL_CPU_LEDGER_TRACE_MATCH' and audit['all_complete'] is True
        and audit['paid_events']==science['logical_packet_events']==counts['whole_calibration'],
        'Whole trace audit payment differs')
    expected=dict(created=True,charged=ledger['charged'],phase_charged=counts,unresolved=0,failed=0,development_remaining=13200)
    require(owner_budget==expected,'Normal owner and scientific ledger snapshots disagree')
    return expected


def verify_whole_closed(a,w,t,paths,oldpaths,state_reader):
    oc,reg,cfg=read(paths['whole_owner_config']),read(paths['whole_registration']),read(paths['whole_config'])
    a.validate_config(oc,reg,sha(paths['whole_owner_config']))
    require(reg['allowed_stage_ids']==['whole_calibration','report'] and reg['source_stage_scope']=='WHOLE1000_ACTUAL_CPU_RECEIVE_ONLY'
        and oc['phase_limits']==t.PHASES and oc['overall_deadline_unix']==DEADLINE and cfg['phase']=='whole_calibration'
        and cfg['schema']=='H_FULL_PAYLOAD_CONFIG_V1' and cfg['workers']==2,'Wrong whole1000 predecessor scope')
    require(oc['registration']==cfg['registration']==str(paths['whole_registration']) and cfg['owner_config']==str(paths['whole_owner_config'])
        and cfg['root']==oc['root'] and cfg['cpu_affinities']==oc['cpu_affinities'] and cfg['phase_limits']==t.PHASES,
        'Whole config/owner relationship differs')
    base=Path(oc['owner_out']);out=Path(cfg['out'])
    require(paths['whole_owner_completion']==base/'completion.json' and paths['whole_completion']==out/'completion.json',
        'Whole completion paths differ')
    for p in (Path(oc['out'])/'STOP',base/'STOP',out/'STOP',out/'worker_0/STOP',out/'worker_1/STOP'):
        require(not p.exists(),'STOP blocks partial1000 registration: '+str(p))
    for p in (base/'failure.json',base/'registration_failure.json',out/'failure.json',out/'worker_0/failure.json',out/'worker_1/failure.json'):
        require(not p.exists(),'Failed whole stage cannot be promoted to success: '+str(p))
    require(len(oc['stages'])==2 and all(s['resource']=='cpu' for s in oc['stages'])
        and len(oc['stages'][0]['jobs'])==2 and len(oc['stages'][1]['jobs'])==1
        and oc['stages'][1]['requires']==['whole_calibration'],'Whole must close two CPU workers followed by merge')
    for stage in oc['stages']:
        for i,job in enumerate(stage['jobs']):
            worker=stage['id']=='whole_calibration';suffix=['--stage','worker','--worker-index',str(i)] if worker else ['--stage','merge']
            folder=out/f'worker_{i}' if worker else out
            require(job['id']==(f'whole_worker_{i}' if worker else 'whole_merge')
                and job['argv']==[job['argv'][0],'-B',str(oldpaths['driver_module']),'--config',str(paths['whole_config']),*suffix]
                and job['out']==str(folder) and job['completion']==str(folder/'completion.json'),'Whole actual job/config differs')
    launch=read(paths['whole_launch']);ident=launch['identity'];ip=base/'owner_identity.json';actual=read(ip)
    require(launch['argv']==ident['argv']==[launch['argv'][0],'-B',str(oldpaths['owner_module']),'--config',str(paths['whole_owner_config'])]
        and launch['registration_sha256']==sha(paths['whole_registration']) and launch['owner_config_sha256']==sha(paths['whole_owner_config']),
        'Whole owner launch differs')
    require(all(actual[k]==ident[k] for k in ('pid','uid','start_ticks','argv')) and actual['registration_sha256']==sha(paths['whole_registration'])
        and actual['config_sha256']==sha(paths['whole_owner_config']),'Whole owner identity differs')
    require(w.exited(ident,state_reader),'Whole owner still live or unreaped')
    closed=w.verify_batch(a,paths['whole_owner_config'],paths['whole_registration'],ident['uid'],state_reader)
    done=read(paths['whole_completion'])
    expected=dict(status='H_FULL1000_CPU_RECEIVE_COMPLETE',phase='whole_calibration',worker_index=None,workers=2,
        source_count=1000,frame_count=24000,maximum_packet_calls=48000,images_scored=False,source_decode_complete=False,
        arithmetic_source_decode_complete=False,GPU_used=False,development_used=False,holdout_used=False,
        policy_selection=False,full1000_calibration_complete=False,visual_stage_started=False,
        requires_both_cpu_groups_closed_before_visual_stage=True,initial_physical_results_reused=False)
    for k,v in expected.items():require(done.get(k)==v,'Whole merged science scope differs: '+k)
    require(done['registration_sha256']==sha(paths['whole_registration']) and done['driver_config_sha256']==sha(paths['whole_config'])
        and done['source_indices']==list(range(1000)) and done['selected_sha256']==sha(cfg['selected'])
        and done['partial_reference_sha256']==sha(cfg['partial_reference']) and done['source_completion_sha256']==sha(cfg['source_completion'])
        and done['budget_registration_sha256']==oc['budget_registration_sha256'],'Whole source/policy/config bindings differ')
    for i in (0,1):
        cp=out/f'worker_{i}/completion.json';d=read(cp)
        require(done['outputs'].get(str(cp))==sha(cp) and d['status']=='H_FULL1000_CPU_WORKER_COMPLETE'
            and d['phase']=='whole_calibration' and d['worker_index']==i and d['workers']==2 and d['source_count']==500
            and d['frame_count']==12000 and d['source_indices']==list(range(i,1000,2))
            and d['driver_config_sha256']==sha(paths['whole_config']) and d['registration_sha256']==sha(paths['whole_registration'])
            and d['selected_sha256']==done['selected_sha256'] and d['partial_reference_sha256']==done['partial_reference_sha256']
            and d['source_completion_sha256']==done['source_completion_sha256'] and d['source_decode_complete'] is False,
            'Whole worker coverage/config differs')
    current=check_closed_budget(reg['budget_before'],read(paths['whole_owner_completion'])['budget'],done,t)
    require(done['ledger']['registration_sha256']==oc['budget_registration_sha256'],'Whole ledger registration differs')
    for values in (reg['source_bindings'],reg['input_bindings'],done['source_bindings'],done['input_bindings'],done['outputs']):verify(values)
    bound=merge(reg['source_bindings'],reg['input_bindings'])
    for k in ('whole_config','whole_request','whole_registrar_module'):
        require(bound.get(str(paths[k]))==sha(paths[k]),'Whole execution did not bind '+k)
    closed['bindings']=merge(closed['bindings'],bind((ip,*[paths[k] for k in ('whole_config','whole_registration','whole_owner_config',
        'whole_launch','whole_owner_completion','whole_completion')])))
    closed.update(owner_identity=ident,owner_exited=True,whole_owner_success=True,original_render_owner_success=False)
    return dict(old=oc,cfg=cfg,reg=reg,done=done,closure=closed,budget=current)


def make_configs(r,ctx,oldpaths):
    old,cfg=ctx['old'],copy.deepcopy(ctx['cfg']);e=Path(r['execution_dir']);out=Path(r['payload_out']);cp=e/'payload_config.json'
    cfg.update(phase='partial_calibration',out=str(out),registration=str(e/'execution_registration.json'),owner_config=str(e/'owner_config.json'),
        engineering_contract=str(e/'engineering_contract.json'),max_worker_seconds=r['max_worker_seconds'],whole_cpu_completion=str(ctx['paths']['whole_completion']))
    cfg['predecessor_batches']['whole']=dict(config=str(ctx['paths']['whole_owner_config']),registration=str(ctx['paths']['whole_registration']),launch=str(ctx['paths']['whole_launch']))
    oc=copy.deepcopy(old);oc.update(owner_out=str(e),registration=cfg['registration']);jobs=[]
    common=dict(phase='partial_calibration',workers=2,images_scored=False,source_decode_complete=False,arithmetic_source_decode_complete=False,
        GPU_used=False,development_used=False,holdout_used=False,policy_selection=False,full1000_calibration_complete=False,maximum_packet_calls=36000,visual_stage_started=False)
    for i in (0,1):
        folder=out/f'worker_{i}';jobs.append(dict(id=f'partial_worker_{i}',argv=[r['python'],'-B',str(oldpaths['driver_module']),
            '--config',str(cp),'--stage','worker','--worker-index',str(i)],cwd=old['root'],out=str(folder),completion=str(folder/'completion.json'),
            accepted_statuses=['H_FULL1000_CPU_WORKER_COMPLETE'],receipt_expect=dict(common,worker_index=i,source_count=500,frame_count=9000)))
    oc['stages']=[dict(id='partial_calibration',resource='cpu',requires=[],max_seconds=r['max_worker_seconds'],jobs=jobs),
        dict(id='report',resource='cpu',requires=['partial_calibration'],max_seconds=r['merge_seconds'],jobs=[dict(id='partial_merge',
            argv=[r['python'],'-B',str(oldpaths['driver_module']),'--config',str(cp),'--stage','merge'],cwd=old['root'],out=str(out),completion=str(out/'completion.json'),
            accepted_statuses=['H_FULL1000_CPU_RECEIVE_COMPLETE'],receipt_expect=dict(common,worker_index=None,source_count=1000,frame_count=18000))])]
    return cfg,oc


def register(request_path):
    request_path=Path(request_path).absolute();r=read(request_path)
    require(r['schema']=='H_PARTIAL1000_CPU_REGISTRATION_REQUEST_V1','Wrong partial1000 request')
    require(sys.platform.startswith('linux'),'Actual registration requires Linux process evidence')
    require(time.time()<DEADLINE,'Original H deadline expired')
    paths={k:pin(r[k]) for k in REQUEST_NAMES};wr=read(paths['whole_registration'])
    verify(wr['source_bindings']);verify(wr['input_bindings']);bound=merge(wr['source_bindings'],wr['input_bindings'])
    for k in ('whole_registrar_module','whole_request','whole_config','whole_owner_config'):
        require(bound.get(str(paths[k]))==sha(paths[k]),'Unbound completed whole dependency: '+k)
    helper=module(paths['whole_registrar_module'],'partial_original_whole_registrar');oldreq=read(paths['whole_request'])
    require(oldreq['schema']=='H_WHOLE1000_CPU_REGISTRATION_REQUEST_V1','Wrong original whole request')
    oldpaths={k:pin(oldreq[k]) for k in helper.REQUEST_NAMES}
    for p in oldpaths.values():require(bound.get(str(p))==sha(p),'Whole request input not bound: '+str(p))
    a=module(oldpaths['owner_module'],'partial_original_stage_owner');w=module(oldpaths['wait_module'],'partial_original_wait')
    t=module(oldpaths['assets_registrar_module'],'partial_assets_registration_toolkit')
    prior_source=helper.verify_source_closed(a,w,t,oldpaths,a.raw_process_state)
    prior_selection=t.verify_selection(a,w,oldpaths,a.raw_process_state)
    ctx=verify_whole_closed(a,w,t,paths,oldpaths,a.raw_process_state);old,cfg=ctx['old'],ctx['cfg']
    require(old['root']==prior_source['old']['root']==prior_selection['owner_config']['root'] and old['out']==prior_source['old']['out'],
        'Whole/source/selection H workspaces differ')
    driver=module(oldpaths['driver_module'],'partial_frozen_payload_driver');prepared=driver.load_registered(paths['whole_config'])
    require(prepared['cfg']['phase']=='whole_calibration' and len(prepared['context']['schedule'])==14,'Wrong full public schedule')
    contract=copy.deepcopy(read(cfg['engineering_contract']))
    contract.update(execution_scope='ONLY partial_calibration; eight whole winners unchanged, no policy selection',
        original_engineering_contract=dict(path=cfg['engineering_contract'],sha256=sha(cfg['engineering_contract'])))
    require(contract['noise_stage']=='full_calibration' and contract['complete_public_receive_catalogue'] is True,
        'Original shared noise/paid catalogue contract differs')
    prepared['core'].prepare(read(cfg['protocol']),read(cfg['catalogue']),read(cfg['selected']),read(cfg['partial_reference']),
        read(cfg['shortlist']),prepared['context']['source_ids'],dict(contract,execution_registration_sha256='0'*64))
    q=read(paths['prepared_qualification']);verify(q['source_bindings'])
    require(q['status']=='H_PARTIAL1000_CPU_QUALIFICATION_PASS' and q['GPU_used'] is False and q['new_packet_decodes']==0
        and q['results'] and all(x['exit_code']==0 for x in q['results']),'Partial implementation not CPU qualified')
    for p in (Path(__file__).absolute(),oldpaths['driver_module'],oldpaths['core_module']):
        require(q['source_bindings'].get(str(p))==sha(p),'Qualification misses current source: '+str(p))
    sources=merge(wr['source_bindings'],q['source_bindings'],bind((Path(__file__).absolute(),)))
    inputs=merge(wr['input_bindings'],ctx['done']['outputs'],ctx['closure']['bindings'],prior_source['closure']['bindings'],
        prior_selection['closure']['bindings'],bind((request_path,*paths.values())))
    before=a.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],t.PHASES,quiescent=True)
    require(before==ctx['budget'],'Shared budget changed after whole merge/owner closed')
    execution,out,h=Path(r['execution_dir']),Path(r['payload_out']),Path(old['out'])
    for p in (execution,out):require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent H path required: '+str(p))
    require(execution!=out and execution not in out.parents and out not in execution.parents,'New output paths overlap')
    require(type(r['max_worker_seconds']) is int and 0<r['max_worker_seconds']<=21600
        and type(r['merge_seconds']) is int and 0<r['merge_seconds']<=r['max_worker_seconds']
        and Path(r['python']).is_absolute() and Path(r['python']).is_file(),'Invalid finite CPU cap/interpreter')
    require(r['python']==oldreq['python'],'Reuse the actually qualified whole CPU interpreter')
    require(set(sum(old['cpu_affinities'],[]))<=set(os.sched_getaffinity(0)),'Registered CPU affinity unavailable')
    ctx['paths']=paths;newcfg,oc=make_configs(r,ctx,oldpaths);verify(sources);verify(inputs);execution.mkdir()
    try:
        cp,op,rp=execution/'payload_config.json',execution/'owner_config.json',execution/'execution_registration.json'
        ep,gate=execution/'engineering_contract.json',execution/'predecessor_closure.json'
        save(gate,dict(status='WHOLE1000_SOURCE_AND_SELECTION_OWNERS_CLOSED',whole=ctx['closure'],source=prior_source['closure'],
            selection=prior_selection['closure'],budget=before,original_render_owner_success=False))
        save(ep,contract);save(cp,newcfg);save(op,oc);inputs=merge(inputs,bind((ep,gate,cp,op)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,
            source_stage_scope='PARTIAL1000_ACTUAL_CPU_RECEIVE_ONLY',allowed_stage_ids=['partial_calibration','report'],
            owner_config_sha256=sha(op),phase_limits=t.PHASES,budget_registration_sha256=old['budget_registration_sha256'],
            scientific_protocol_sha256=sha(cfg['protocol']),source_bindings=sources,input_bindings=inputs,budget_before=before,
            source_count=1000,frozen_partial_candidates=6,whole_candidates_reselected=False,noise_seeds=[6101,6102,6103],snrs_db=[13,19],
            frame_count=18000,maximum_packet_calls=36000,CPU_workers=2,threads_per_worker=2,nice=15,cpu_affinities=old['cpu_affinities'],
            GPU_jobs=0,noise_stage='full_calibration',full_public_catalogue=True,visual_stage_started=False,policy_selection=False,
            development_used=False,holdout_started=False,full1000_calibration_complete=False,H_full_delivery_claimed=False,C_started=False,
            scientific_protocol_modified=False,future_stage_automatic=False,created_unix=time.time())
        a.validate_config(oc,reg,sha(op));verify(sources);verify(inputs)
        require(not (h/'STOP').exists() and time.time()<DEADLINE,'STOP/deadline changed')
        verify_whole_closed(a,w,t,paths,oldpaths,a.raw_process_state)
        helper.verify_source_closed(a,w,t,oldpaths,a.raw_process_state);t.verify_selection(a,w,oldpaths,a.raw_process_state)
        require(a.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],t.PHASES,quiescent=True)==before,
            'Shared budget changed during partial registration')
        save(rp,reg);driver.load_registered(cp)  # Immutable metadata only, never build_runtime/run_frame.
        done=dict(status='H_PARTIAL1000_CPU_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(rp),owner_config_sha256=sha(op),
            outputs=bind((cp,op,rp,ep,gate)),source_count=1000,frozen_partial_candidates=6,frame_count=18000,maximum_packet_calls=36000,
            workers_started=False,new_packet_decodes=0,GPU_used=False,visual_stage_started=False,future_stage_automatic=False,
            required_launch_condition='Verify registration_completion outputs and absence of registration_failure.json; recheck all exits, STOP and exact closed budget before owner launch.')
        save(execution/'registration_completion.json',done);return done
    except BaseException as error:
        save(execution/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);a=p.parse_args()
    result=register(a.request);print(json.dumps({k:result[k] for k in ('status','registration_sha256')}))


if __name__=='__main__':main()
