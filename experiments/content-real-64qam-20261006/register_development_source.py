"""Register one source-only development100 stage, never launch it.

CPU assets require the normal finalized calibration selector. GPU source coding
also requires its normal CPU asset owner. No packet decoder/model is created by
registration; all source data remain outside policy selection.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
import development_source_common as c

QUALIFICATION = 'H_DEVELOPMENT100_SOURCE_QUALIFICATION_PASS'
REGISTERED = {'assets': 'H_DEVELOPMENT100_ASSETS_REGISTERED_NOT_LAUNCHED',
              'source': 'H_DEVELOPMENT100_SOURCE_REGISTERED_NOT_LAUNCHED'}


def pin(record):
    path = Path(record['path'])
    c.require(path.is_absolute() and c.sha(path) == record['sha256'], 'Pinned file absent/changed: '+str(path))
    return path


def expected(kind):
    if kind == 'selection':
        return dict(status='H_FULL1000_CALIBRATION_SELECTION_COMPLETE', source_count=1000,
                    measured_frames=42000, whole_count=8, selected_partial_count=2,
                    whole_policy_reselected=False, development_used=False, new_packet_decodes=0)
    return dict(status=c.ASSETS_DONE, source_count=100, new_packet_decodes=0,
                development_used=True, source_codec_complete=False, pending_source_encoding_count=100)


def close_predecessors(cfg, kind, api, wait):
    names = {'selection'} if kind == 'assets' else {'selection', 'assets'}
    c.require(set(cfg['prerequisites']) == names, 'Exact normal predecessor batches required')
    result = {}
    for name, spec in cfg['prerequisites'].items():
        c.require(set(spec) == {'config', 'registration', 'launch', 'completion'}, 'Unexpected predecessor pin fields')
        closed = c.d.closed_batch(api, wait, spec, cfg['owner_module'], expected(name), api.raw_process_state)
        c.require(closed['owner']['root'] == cfg['root'] and closed['owner']['out'] == cfg['H_out']
                  and closed['owner']['budget_path'] == cfg['ledger'], 'Predecessor workspace/ledger differs')
        result[name] = closed
    c.require(cfg['prerequisites']['selection']['completion'] == cfg['selection_completion'], 'Selection completion differs')
    return result


def original_visual(closed):
    """Read the real selector-to-render request chain, not arbitrary new paths."""
    oc, reg = closed['owner'], closed['registration']
    c.require(reg['source_stage_scope'] == 'FULL1000_CALIBRATION_SELECTION_ONLY'
              and reg['allowed_stage_ids'] == ['freeze'] and len(oc['stages']) == 1
              and len(oc['stages'][0]['jobs']) == 1, 'Wrong finalized selector scope')
    job = oc['stages'][0]['jobs'][0]; argv = job['argv']
    c.require(job['id'] == 'full1000_selection' and argv.count('--config') == 1,
              'Wrong actual selector job')
    cp = argv[argv.index('--config')+1]
    bound = c.merge(reg['source_bindings'], reg['input_bindings'])
    c.require(bound.get(cp) == c.sha(cp), 'Selector execution config not bound')
    sc = c.read(cp); request = sc['request_path']
    c.require(bound.get(request) == c.sha(request), 'Original selector request not bound')
    r = c.read(request); vp = pin(r['render_config'])
    c.require(bound.get(str(vp)) == c.sha(vp), 'Original visual config not in selector provenance')
    visual = c.read(vp)
    return visual, c.bind((cp, request, vp))


def validate_resources(cfg, kind, old, visual):
    c.require(kind in c.SCOPES and cfg['kind'] == kind
              and cfg['schema'] == 'H_DEVELOPMENT100_PREPARATION_CONFIG_V1', 'Wrong preparation kind/schema')
    c.require(cfg['root'] == old['root'] == visual['root'] and cfg['H_out'] == old['out'] == visual['H_out']
              and cfg['ledger'] == old['budget_path'] == visual['ledger']
              and cfg['budget_registration'] == old['budget_registration'] == visual['budget_registration']
              and cfg['phase_limits'] == old['phase_limits'] == c.d.PHASES
              and cfg['overall_deadline_unix'] == old['overall_deadline_unix'] == c.DEADLINE,
              'Original workspace/budget/deadline differs')
    c.require(cfg['stop_file'] == str(Path(cfg['H_out'])/'STOP') and cfg['cpu_affinity'] == old['cpu_affinities'][0]
              and len(cfg['cpu_affinity']) == 2, 'Source CPU scheduling changed')
    for key in ('protocol', 'calibration_registration', 'owner_module', 'wait_module'):
        c.require(cfg[key] == visual[key], 'Original frozen input differs: '+key)
    if kind == 'source':
        for key in ('runtime_dir', 'uep_runtime', 'native_runtime', 'var_source', 'dino_source',
                    'static_closure_module', 'numerical_reference', 'source_driver_module'):
            c.require(cfg[key] == visual[key], 'Original visual codec setting differs: '+key)
        c.require(cfg['numerical_reference_field'] == ['numerical_runtime'] and old['gpu_threads'] == 6
                  and len(old['gpu_affinity']) == 6, 'Original GPU numerical resources differ')


def make_configs(request, draft, old, kind, entry):
    e = Path(request['execution_dir']); rp=e/'execution_registration.json'; cp=e/'source_config.json'; op=e/'owner_config.json'
    cfg = copy.deepcopy(draft)
    cfg.update(kind=kind, registration=str(rp), owner_config=str(op), out=request['source_out'], max_seconds=request['max_seconds'])
    oc = copy.deepcopy(old); oc.update(owner_out=str(e), registration=str(rp))
    exp = dict(source_count=100, new_packet_decodes=0, new_image_renders=0, new_metric_calls=0,
               development_used=True, holdout_used=False)
    if kind == 'assets':
        status, resource, jobid = c.ASSETS_DONE, 'cpu', 'development100_assets'
        exp.update(GPU_used=False, source_codec_complete=False, pending_source_encoding_count=100)
    else:
        status, resource, jobid = c.SOURCE_DONE, 'gpu', 'development100_source'
        exp.update(GPU_used=True, source_codec_complete=True, pending_source_encoding_count=0,
                   independent_roundtrip=True, encoder_tokens_verified=True, new_canonical_roundtrips=400,
                   training_updates=0, policy_selection=False)
    oc['stages']=[dict(id='source', resource=resource, requires=[], max_seconds=request['max_seconds'], jobs=[dict(
        id=jobid, argv=[request['python'], '-B', str(entry), '--config', str(cp)], cwd=cfg['root'], out=cfg['out'],
        completion=str(Path(cfg['out'])/'completion.json'), accepted_statuses=[status], receipt_expect=exp)])]
    return cfg, oc


def inspect(request, kind):
    dp, qp = pin(request['draft_config']), pin(request['prepared_qualification'])
    cfg, q = c.read(dp), c.read(qp)
    c.require(cfg['kind'] == kind and cfg['source_qualification'] == str(qp), 'Qualification or requested kind differs')
    c.require(q['status'] == QUALIFICATION and q['GPU_used'] is False and q['new_packet_decodes'] == 0
              and q['results'] and all(x['exit_code'] == 0 for x in q['results']), 'Complete CPU source qualification required')
    c.verify(q['source_bindings'])
    entry_name = 'development100_assets.py' if kind == 'assets' else 'development100_source_driver.py'
    entry = Path(cfg['common_module']).parent/entry_name
    for p in (Path(__file__).absolute(), Path(c.__file__).absolute(), Path(c.d.__file__).absolute(), entry):
        c.require(q['source_bindings'].get(str(p)) == c.sha(p), 'Qualification misses actual implementation: '+str(p))
    extra = c.bind([pin(x) for x in request['additional_pins']])
    supplied = c.merge(q['source_bindings'], extra, c.bind((dp, qp)))
    for key in ('owner_module', 'wait_module', 'development_registration', 'calibration_registration'):
        c.require(supplied.get(cfg[key]) == c.sha(cfg[key]), 'Unbound admission input: '+key)
    for spec in cfg['prerequisites'].values():
        for p in spec.values():
            c.require(supplied.get(p) == c.sha(p), 'Unbound predecessor request input')
    api=c.module(cfg['owner_module'],'dev_source_registration_owner')
    wait=c.module(cfg['wait_module'],'dev_source_registration_wait')
    closures=close_predecessors(cfg,kind,api,wait)
    selection=closures['selection']; old=selection['owner']; visual, original_pins=original_visual(selection)
    validate_resources(cfg,kind,old,visual)
    pop,cal=c.read(cfg['development_registration']),c.read(cfg['calibration_registration']);ids=c.population(pop,cal)
    c.require(len(ids)==100, 'Exact development100 population required')
    sources=c.merge(q['source_bindings'],c.bind((Path(__file__).absolute(),)))
    inputs=c.merge(extra,original_pins,c.bind((dp,qp)))
    for closed in closures.values():
        sources=c.merge(sources,closed['registration']['source_bindings'])
        inputs=c.merge(inputs,closed['registration']['input_bindings'],closed['bindings'],closed['done']['outputs'])
    if kind=='assets':
        root=Path(cfg['root']); pixelroot=root/'outputs/VAR-PROGRESSIVE-CHANNEL-001'
        c.require(Path(cfg['original_pixel_root'])==pixelroot
                  and Path(cfg['token_archive'])==root/'outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/source_tokens.npz'
                  and Path(cfg['population_manifest'])==root/'outputs/COMMUNICATION-CONVERGENCE-20260915/DEVELOPMENT_001/population.json',
                  'Original development cache paths differ')
        for row in pop['data_bindings']:
            p=pixelroot/'images'/f"{row['index']:03d}"/'reconstructions.npz'
            c.require(inputs.get(str(p))==row['source_npz_sha256']==c.sha(p),'Original source archive not explicitly bound')
        actual, delta = {}, {}
    else:
        assets=closures['assets'];done=assets['done']
        c.require(cfg['asset_completion']==cfg['prerequisites']['assets']['completion']
                  and cfg['assets_registration']==cfg['prerequisites']['assets']['registration']
                  and done['outputs'].get(cfg['asset_manifest'])==c.sha(cfg['asset_manifest']), 'Assets lineage differs')
        c.validate_manifest(c.read(cfg['asset_manifest']),done,pop)
        helper=c.module(cfg['static_closure_module'],'dev_source_registration_static_closure')
        actual=helper.collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])
        oldbound=c.merge(sources,inputs);delta=helper.compare_bindings(actual,oldbound)
        c.require(not delta['changed'],'An existing source changed; separate diagnosis required')
        sources=c.merge(sources,actual)
        c.require(helper.compare_bindings(actual,c.merge(sources,inputs))['status']=='EXACT_SOURCE_CLOSURE_MATCH','Incomplete static source graph')
    bound=c.merge(sources,inputs)
    required=c.BASE_REQUIRED+(c.GPU_REQUIRED if kind=='source' else ('population_manifest','token_archive'))
    for key in required:
        if key=='owner_config':continue  # New owner config is created and bound below.
        c.require(bound.get(cfg[key])==c.sha(cfg[key]),'Missing exact preparation dependency: '+key)
    before=api.budget_snapshot(cfg['ledger'],c.sha(cfg['budget_registration']),c.d.PHASES,quiescent=True)
    c.require(before['unresolved']==before['failed']==0 and before['phase_charged']['development']==0
              and before['development_remaining']==13200,'Original development reserve is not unused')
    for closed in closures.values():
        c.require(closed['owner_done']['budget']==before,'Predecessor and current budgets differ')
    c.verify(sources);c.verify(inputs)
    return dict(cfg=cfg,api=api,wait=wait,old=old,visual=visual,entry=entry,closures=closures,
                sources=sources,inputs=inputs,before=before,static=actual,delta=delta,ids=ids)


def register(request_path,kind):
    request_path=Path(request_path).absolute();r=c.read(request_path)
    c.require(sys.platform.startswith('linux') and r['schema']=='H_DEVELOPMENT100_SOURCE_PREPARATION_REQUEST_V1'
              and r['kind']==kind and kind in c.SCOPES,'Actual Linux source-preparation request required')
    c.require(time.time()<c.DEADLINE,'Original H deadline exceeded')
    ctx=inspect(r,kind);old=ctx['old'];e,out,h=Path(r['execution_dir']),Path(r['source_out']),Path(old['out'])
    for p in (e,out):c.require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent H output required')
    c.require(e!=out and e not in out.parents and out not in e.parents,'Execution/source output paths overlap')
    c.require(Path(r['python']).is_absolute() and Path(r['python']).is_file()
              and type(r['max_seconds']) is int and 0<r['max_seconds']<=86400,'Invalid interpreter or finite source cap')
    affinity=old['cpu_affinities'][0] if kind=='assets' else old['gpu_affinity']
    c.require(set(affinity)<=os.sched_getaffinity(0),'Original source CPU affinity unavailable')
    cfg,oc=make_configs(r,ctx['cfg'],old,kind,ctx['entry'])
    inputs=c.merge(ctx['inputs'],c.bind((request_path,)));sources=ctx['sources'];e.mkdir()
    try:
        cp,op,rp=e/'source_config.json',e/'owner_config.json',e/'execution_registration.json'
        gate=e/'predecessor_closure.json';c.save(gate,dict(status='DEVELOPMENT100_SOURCE_PREDECESSORS_NORMALLY_CLOSED',
            kind=kind,closures=ctx['closures'],budget=ctx['before'],source_count=100,no_recovery_used=True))
        c.save(cp,cfg);c.save(op,oc);outputs=[gate,cp,op]
        if kind=='source':
            graph=e/'visual_source_closure.json';c.save(graph,dict(status='EXACT_SOURCE_CLOSURE_MATCH',source_bindings=ctx['static'],
                newly_bound_sources=ctx['delta']['missing'],original_registered_files_changed=False,
                source_enumeration_only=True,GPU_used=False,new_packet_decodes=0));outputs.append(graph)
        inputs=c.merge(inputs,c.bind(outputs))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=e.name,source_stage_scope=c.SCOPES[kind],
            allowed_stage_ids=['source'],owner_config_sha256=c.sha(op),phase_limits=c.d.PHASES,
            budget_registration_sha256=c.sha(cfg['budget_registration']),budget_before=ctx['before'],
            source_bindings=sources,input_bindings=inputs,source_count=100,development_used=True,holdout_started=False,
            new_packet_decodes=0,new_image_renders=0,new_metric_calls=0,training_updates=0,
            GPU_jobs=int(kind=='source'),GPU_threads=6 if kind=='source' else 0,
            source_codec_complete=False,policy_selection=False,H_full_delivery_claimed=False,C_started=False,
            scientific_protocol_modified=False,future_stage_automatic=False,created_unix=time.time())
        ctx['api'].validate_config(oc,reg,c.sha(op));c.verify(sources);c.verify(inputs)
        c.require(not Path(cfg['stop_file']).exists() and not (e/'STOP').exists() and time.time()<c.DEADLINE,'STOP/deadline changed')
        c.require(ctx['api'].budget_snapshot(cfg['ledger'],c.sha(cfg['budget_registration']),c.d.PHASES,quiescent=True)==ctx['before'],
                  'Shared H budget changed during registration')
        c.save(rp,reg)
        # Metadata/file SHA/process/ledger admission only. Do not invoke asset
        # plan_registered or construct the GPU native model during registration.
        c.load_registered(cp,kind,ctx['entry'])
        result=dict(status=REGISTERED[kind],registration_sha256=c.sha(rp),owner_config_sha256=c.sha(op),
            outputs=c.bind(outputs+[rp]),source_count=100,workers_started=False,GPU_used=False,new_packet_decodes=0,
            new_source_encodings=0,future_stage_automatic=False,
            required_launch_condition='This completion and all outputs verify, registration_failure absent, all predecessors normally exited, STOP absent and exact budget unchanged. Registration never launches.')
        c.save(e/'registration_completion.json',result);return result
    except BaseException:
        c.save(e/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('assets','source'),required=True)
    p.add_argument('--request',required=True);a=p.parse_args();print(json.dumps(register(a.request,a.stage)))


if __name__=='__main__':main()
