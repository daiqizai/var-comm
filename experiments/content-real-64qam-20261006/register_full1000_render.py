"""Register one actual-RX GPU batch after both full1000 CPU groups close.

This tool hashes inputs and inspects immutable receipts/closed process evidence.
It never builds a model, decodes a packet, renders a frame, selects a policy or
starts a job. Existing science, source guards and all budget limits are retained.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

DEADLINE=1791564605.9549868
ARCHIVE_CAP=36*1024**3
REQUEST_NAMES=('partial_registrar_module','partial_request','partial_config','partial_registration','partial_owner_config',
    'partial_launch','partial_owner_completion','partial_completion','source_registrar_module','source_visual_closure',
    'rx_adapter_module','visual_driver_module','prepared_qualification')


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


def final_budget(t,partial_helper,whole_reg,whole_owner,whole_done,partial_reg,partial_owner,partial_done):
    """Retain actual whole calls; advance only the separately paid partial phase."""
    whole_closed=partial_helper.check_closed_budget(whole_reg['budget_before'],whole_owner['budget'],whole_done,t)
    require(partial_reg['budget_before']==whole_closed,'Partial registration did not start at the closed whole budget')
    ledger=partial_done['ledger'];counts=ledger['phase_charged'];rows=ledger['counts'];seen=set();observed=dict.fromkeys(t.PHASES,0)
    require(ledger['schema']=='PACKET_PRECHARGE_V1' and ledger['branch']=='H' and ledger['limits']==t.PHASES
        and ledger['total_cap']==200000 and ledger['unresolved']==0 and ledger['development_reserve_transferable'] is False,
        'Partial scientific ledger is unresolved or uses another budget')
    headers=bodies=0
    for row in rows:
        key=(row['phase'],row['kind'],row['status']);n=row['count']
        require(key not in seen and row['phase'] in t.PHASES and row['kind'] in ('header','body') and row['status']=='COMPLETE'
            and type(n) is int and n>0,'Invalid/duplicate/unresolved physical ledger count')
        seen.add(key);observed[row['phase']]+=n
        if row['phase']=='partial_calibration':
            if row['kind']=='header':headers+=n
            else:bodies+=n
    require(counts==observed and headers==18000 and 0<=bodies<=18000 and counts['partial_calibration']==headers+bodies,
        'Partial actual header/body coverage differs')
    require(all(counts[p]==whole_closed['phase_charged'][p] for p in t.PHASES if p!='partial_calibration')
        and ledger['charged']==sum(counts.values()) and ledger['phase_remaining']=={p:t.PHASES[p]-counts[p] for p in counts},
        'Another budget phase advanced or recorded totals differ')
    audit=partial_done['ledger_audit']
    require(audit['status']=='FULL_CPU_LEDGER_TRACE_MATCH' and audit['all_complete'] is True
        and audit['paid_events']==partial_done['logical_packet_events']==counts['partial_calibration'],
        'Partial actual trace audit differs')
    expected=dict(created=True,charged=ledger['charged'],phase_charged=counts,unresolved=0,failed=0,development_remaining=13200)
    require(partial_owner['budget']==expected,'Partial owner/ledger final budgets disagree')
    return expected


def inspect_predecessors(paths):
    """Read registered request chains so no visual asset path is guessed."""
    reg=read(paths['partial_registration']);verify(reg['source_bindings']);verify(reg['input_bindings'])
    bound=merge(reg['source_bindings'],reg['input_bindings'])
    require(reg['allowed_stage_ids']==['partial_calibration','report']
        and reg['source_stage_scope']=='PARTIAL1000_ACTUAL_CPU_RECEIVE_ONLY','Wrong partial predecessor revision')
    for k in ('partial_registrar_module','partial_request','partial_config','partial_owner_config','source_registrar_module','source_visual_closure'):
        require(bound.get(str(paths[k]))==sha(paths[k]),'Partial execution omitted historical input: '+k)
    ph=module(paths['partial_registrar_module'],'visual_partial_registration_gate');pr=read(paths['partial_request'])
    require(pr['schema']=='H_PARTIAL1000_CPU_REGISTRATION_REQUEST_V1','Wrong actual partial request')
    pp={k:pin(pr[k]) for k in ph.REQUEST_NAMES}
    for p in pp.values():require(bound.get(str(p))==sha(p),'Partial original request input unbound: '+str(p))
    wh=module(pp['whole_registrar_module'],'visual_whole_registration_gate');wr=read(pp['whole_request'])
    oldpaths={k:pin(wr[k]) for k in wh.REQUEST_NAMES}
    a=module(oldpaths['owner_module'],'visual_original_stage_owner');w=module(oldpaths['wait_module'],'visual_original_wait')
    t=module(oldpaths['assets_registrar_module'],'visual_original_assets_gate')
    source=wh.verify_source_closed(a,w,t,oldpaths,a.raw_process_state)
    selection=t.verify_selection(a,w,oldpaths,a.raw_process_state)
    whole=ph.verify_whole_closed(a,w,t,pp,oldpaths,a.raw_process_state)
    cfg=read(paths['partial_config']);oc=read(paths['partial_owner_config']);a.validate_config(oc,reg,sha(paths['partial_owner_config']))
    require(paths['partial_owner_completion']==Path(oc['owner_out'])/'completion.json'
        and paths['partial_completion']==Path(cfg['out'])/'completion.json','Actual partial completion paths differ')
    spec1=dict(config=str(pp['whole_config']),registration=str(pp['whole_registration']),owner_config=str(pp['whole_owner_config']),
        launch=str(pp['whole_launch']),completion=str(pp['whole_completion']))
    spec2=dict(config=str(paths['partial_config']),registration=str(paths['partial_registration']),owner_config=str(paths['partial_owner_config']),
        launch=str(paths['partial_launch']),completion=str(paths['partial_completion']))
    driver=module(oldpaths['driver_module'],'visual_frozen_physical_driver')
    rx=module(paths['rx_adapter_module'],'visual_prepared_actual_rx_gate')
    groups={}
    for phase,spec in (('whole_calibration',spec1),('partial_calibration',spec2)):
        groups[phase]=rx.verify_batch_closed(spec,phase,driver,a,w)
    first,second=groups['whole_calibration'],groups['partial_calibration']
    for k in ('root','runtime_dir','protocol','catalogue','shortlist','selected','partial_reference','source_completion','asset_completion',
              'asset_manifest','ledger','phase_limits','cpu_affinities','budget_registration','budget_registration_sha256'):
        require(first['cfg'][k]==second['cfg'][k],'Whole/partial physical inputs changed: '+k)
    require(first['context']['context']['schedule']==second['context']['context']['schedule']
        and first['context']['context']['source_ids']==second['context']['context']['source_ids'],
        'Whole/partial source IDs or public14-slot policies differ')
    require(first['done']['frame_count']+second['done']['frame_count']==42000,'Exactly42000 actual received frames required')
    before=final_budget(t,ph,whole['reg'],read(pp['whole_owner_completion']),first['done'],reg,read(paths['partial_owner_completion']),second['done'])
    require(a.budget_snapshot(oc['budget_path'],oc['budget_registration_sha256'],t.PHASES,quiescent=True)==before,
        'Ledger changed after final partial owner closed')
    bindings=merge(source['closure']['bindings'],selection['closure']['bindings'],whole['closure']['bindings'],
        *(v['closure']['bindings'] for v in groups.values()),bind((*pp.values(),*oldpaths.values(),*paths.values())))
    return dict(old=oc,reg=reg,cfg=cfg,bound=bound,source=source,selection=selection,whole=whole,groups=groups,
        cpu_batches={'whole_calibration':spec1,'partial_calibration':spec2},before=before,closure_bindings=bindings,
        owner=a,wait=w,toolkit=t,rx=rx,oldpaths=oldpaths)


def visual_inputs(ctx,paths):
    cfg=ctx['source']['source_cfg'];done=ctx['source']['source_done'];oldbound=ctx['bound']
    for k in ('static_closure_module','numerical_reference','calibration_registration','source_driver_module'):
        require(oldbound.get(cfg[k])==sha(cfg[k]),'Original visual dependency not sealed: '+k)
    numerical=read(cfg['numerical_reference']);cal=read(cfg['calibration_registration'])
    require(numerical['status']=='REAL_NATIVE_QUALIFICATION_PASS' and numerical['synthetic'] is False
        and numerical['frozen_identity']==cal['identity']==done['frozen_visual_identity'],
        'Original visual qualification/calibration/source identities differ')
    flags=numerical['numerical_runtime']
    require(flags==done['numerical_runtime'] and flags['threads']==ctx['old']['gpu_threads']==6
        and flags['deterministic'] is True and flags['matmul_tf32'] is False and flags['cudnn_tf32'] is False
        and flags['cudnn_benchmark'] is False and flags['precision']=='highest','Original complete FP32 numerical flags differ')
    sh=module(paths['source_registrar_module'],'visual_source_registration_toolkit')
    helper=module(cfg['static_closure_module'],'visual_complete_native_source_graph')
    reference=read(paths['source_visual_closure'])
    actual,delta=sh.static_source_closure(helper,ctx['old']['root'],cfg,reference,oldbound)
    return dict(cfg=cfg,reference=reference,actual=actual,delta=delta,helper=helper,flags=flags)


def make_configs(r,ctx,paths,visual):
    old=ctx['old'];source=visual['cfg'];physical=ctx['cfg'];e=Path(r['execution_dir']);rp=e/'execution_registration.json';cp=e/'render_config.json'
    cfg={k:copy.deepcopy(source[k]) for k in ('root','H_out','runtime_dir','uep_runtime','native_runtime','var_source','dino_source',
        'calibration_registration','numerical_reference','source_driver_module','static_closure_module','ledger','budget_registration','phase_limits','stop_file')}
    runtime=Path(source['runtime_dir'])
    cfg.update(schema='H_FULL1000_RX_CONFIG_V1',out=r['render_out'],registration=str(rp),visual_owner_config=str(e/'owner_config.json'),
        owner_module=str(ctx['oldpaths']['owner_module']),wait_module=str(ctx['oldpaths']['wait_module']),
        cpu_driver_module=str(ctx['oldpaths']['driver_module']),receiver_module=str(runtime/'h_payload_rx.py'),
        cache_module=str(runtime/'h_payload_render_driver.py'),rx_adapter_module=str(paths['rx_adapter_module']),
        visual_driver_module=str(paths['visual_driver_module']),protocol=physical['protocol'],cpu_batches=copy.deepcopy(ctx['cpu_batches']),
        max_seconds=r['max_seconds'],overall_deadline_unix=DEADLINE,max_archive_bytes=ARCHIVE_CAP)
    oc=copy.deepcopy(old);oc.update(owner_out=str(e),registration=str(rp))
    oc['stages']=[dict(id='render',resource='gpu',requires=[],max_seconds=r['max_seconds'],jobs=[dict(id='full1000_render',
        argv=[r['python'],'-B',str(paths['visual_driver_module']),'--config',str(cp)],cwd=old['root'],out=r['render_out'],
        completion=str(Path(r['render_out'])/'completion.json'),accepted_statuses=['H_FULL1000_RX_COMPLETE'],
        receipt_expect=dict(source_count=1000,frame_count=42000,images_scored=True,source_decode_complete=True,
            arithmetic_source_decode_complete=True,phase_frame_counts={'whole_calibration':24000,'partial_calibration':18000},
            new_packet_decodes=0,policy_selection=False,development_used=False,holdout_used=False))])]
    return cfg,oc


def register(request_path):
    request_path=Path(request_path).absolute();r=read(request_path)
    require(r['schema']=='H_FULL1000_RENDER_REGISTRATION_REQUEST_V1','Wrong full1000 render request')
    require(sys.platform.startswith('linux'),'Actual registration requires Linux process evidence')
    require(time.time()<DEADLINE,'Original H deadline expired')
    paths={k:pin(r[k]) for k in REQUEST_NAMES}
    q=read(paths['prepared_qualification']);verify(q['source_bindings'])
    require(q['status']=='H_FULL1000_RENDER_CPU_QUALIFICATION_PASS' and q['GPU_used'] is False and q['new_packet_decodes']==0
        and q['results'] and all(x['exit_code']==0 for x in q['results']),'Prepared full visual code lacks CPU qualification')
    for p in (Path(__file__).absolute(),paths['rx_adapter_module'],paths['visual_driver_module']):
        require(q['source_bindings'].get(str(p))==sha(p),'Qualification misses new executable source: '+str(p))
    ctx=inspect_predecessors(paths);visual=visual_inputs(ctx,paths);a,t=ctx['owner'],ctx['toolkit'];old=ctx['old'];before=ctx['before']
    sources=merge(ctx['reg']['source_bindings'],q['source_bindings'],visual['actual'],bind((Path(__file__).absolute(),)))
    inputs=merge(ctx['reg']['input_bindings'],ctx['closure_bindings'],*(v['done']['outputs'] for v in ctx['groups'].values()),
        bind((request_path,*paths.values())))
    verify(sources);verify(inputs)
    execution,out,h=Path(r['execution_dir']),Path(r['render_out']),Path(old['out'])
    for p in (execution,out):require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent H path required: '+str(p))
    require(execution!=out and execution not in out.parents and out not in execution.parents,'New execution/science paths overlap')
    require(Path(r['python']).is_absolute() and Path(r['python']).is_file() and type(r['max_seconds']) is int
        and 0<r['max_seconds']<=86400,'Original finite visual time cap/interpreter required')
    source_owner=ctx['source']['old'];source_python=source_owner['stages'][0]['jobs'][0]['argv'][0]
    require(r['python']==source_python,'Use the same actual original GPU environment')
    require(old['gpu_threads']==6 and set(old['gpu_affinity'])<=set(os.sched_getaffinity(0)),
        'Original six-thread GPU CPU affinity unavailable')
    require(shutil.disk_usage(h).free>=ARCHIVE_CAP+8*1024**3,'Insufficient space for all frames plus8GiB reserve')
    cfg,oc=make_configs(r,ctx,paths,visual)
    for key in ('receiver_module','cache_module','source_driver_module','calibration_registration','numerical_reference',
        'static_closure_module','owner_module','wait_module','cpu_driver_module'):
        require(merge(inputs,sources).get(cfg[key])==sha(cfg[key]),'Visual dependency unbound: '+key)
    execution.mkdir()
    try:
        cp,op,rp=execution/'render_config.json',execution/'owner_config.json',execution/'execution_registration.json'
        gate,graph=execution/'predecessor_closure.json',execution/'visual_source_closure.json'
        save(gate,dict(status='BOTH_FULL1000_CPU_GROUPS_AND_SOURCE_SELECTION_CLOSED',bindings=ctx['closure_bindings'],budget=before,
            phase_frames={'whole_calibration':24000,'partial_calibration':18000},new_packet_decodes=0,GPU_used=False,
            original_render_owner_success=False))
        save(graph,dict(status='EXACT_SOURCE_CLOSURE_MATCH',source_bindings=visual['actual'],inputs=visual['reference']['inputs'],
            newly_bound_sources=visual['delta']['missing'],original_registered_files_changed=False,source_enumeration_only=True,
            helper_sha256=sha(visual['cfg']['static_closure_module']),GPU_used=False,new_packet_decodes=0))
        save(cp,cfg);save(op,oc);inputs=merge(inputs,bind((cp,op,gate,graph)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,source_stage_scope='FULL1000_ACTUAL_RX_IMAGES_ONLY',
            allowed_stage_ids=['render'],owner_config_sha256=sha(op),phase_limits=t.PHASES,budget_registration_sha256=old['budget_registration_sha256'],
            scientific_protocol_sha256=sha(cfg['protocol']),source_bindings=sources,input_bindings=inputs,budget_before=before,
            source_count=1000,frame_count=42000,phase_frames={'whole_calibration':24000,'partial_calibration':18000},noise_seeds=[6101,6102,6103],
            GPU_jobs=1,GPU_threads=6,GPU_affinity=old['gpu_affinity'],max_archive_bytes=ARCHIVE_CAP,new_packet_decodes=0,
            policy_selection=False,development_used=False,holdout_started=False,H_full_delivery_claimed=False,C_started=False,
            scientific_protocol_modified=False,future_stage_automatic=False,created_unix=time.time(),
            shared_ledger_exclusion='No shared-ledger job until this visual owner and worker normally close.',
            receiver_rule='Actual paid header/payload only; target pixels enter PSNR/MSE scoring afterwards; unresolved canonical decode cannot be scored.')
        a.validate_config(oc,reg,sha(op));verify(sources);verify(inputs)
        require(visual['helper'].collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])==visual['actual'],
            'Tracked visual source graph changed during registration')
        require(not (h/'STOP').exists() and time.time()<DEADLINE,'STOP/deadline changed')
        require(a.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],t.PHASES,quiescent=True)==before,
            'Shared budget changed during visual registration')
        save(rp,reg)
        visual_driver=module(paths['visual_driver_module'],'full_visual_registered_entry')
        checked=visual_driver.load_registered(cp)
        require(checked['before']==before and len(checked['source_ids'])==1000 and len(checked['schedule'])==14,
            'Read-only visual driver preflight differs')
        require(a.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],t.PHASES,quiescent=True)==before,
            'Shared budget changed during read-only visual preflight')
        done=dict(status='H_FULL1000_RENDER_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(rp),owner_config_sha256=sha(op),
            outputs=bind((cp,op,rp,gate,graph)),source_count=1000,frame_count=42000,workers_started=False,GPU_used=False,new_packet_decodes=0,
            budget=before,policy_selection=False,future_stage_automatic=False,read_only_driver_preflight='PASS',
            required_launch_condition='Verify registration_completion outputs and no registration_failure; all CPU/source/selection owners closed, STOP absent and shared ledger unchanged. Admit one GPU worker; no later shared-ledger job until its normal close.')
        save(execution/'registration_completion.json',done);return done
    except BaseException as error:
        save(execution/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);a=p.parse_args()
    result=register(a.request);print(json.dumps({k:result[k] for k in ('status','registration_sha256')}))


if __name__=='__main__':main()
