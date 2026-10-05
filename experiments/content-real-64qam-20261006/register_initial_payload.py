"""Seal one independent initial_true200 CPU batch after all predecessors close.

No subprocess, decoder, model or GPU is started. This is intentionally limited
to the actual coarse-prescreen FINAL path; refinement needs a separate helper
revision. Missing/live/failed prerequisites leave the new registration absent.
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


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def merge(*maps):
    result={}
    for mapping in maps:
        for path,value in mapping.items():
            require(path not in result or result[path]==value,'Conflicting immutable binding: '+path)
            result[path]=value
    return result
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def pinned(record):
    p=Path(record['path']);require(p.is_absolute() and sha(p)==record['sha256'],'Pinned evidence changed: '+str(p));return p
def command_config(argv):
    require(argv.count('--config')==1 and argv.index('--config')+1<len(argv),'Missing unique prerequisite config')
    return argv[argv.index('--config')+1]
def bind(paths):return {str(p):sha(p) for p in paths}
def save(path,value):
    path=Path(path);require(not path.exists(),'Existing evidence must not be overwritten: '+str(path))
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def verify_prescreen_closed(waiter,wapi,launch,stamp,state_reader):
    """Close every prior owner/worker using frozen receipt verifiers, read-only."""
    a,p=waiter.a,waiter.p;c=waiter.cfg;base=waiter.waiter_out;ident=launch['identity']
    expected=[launch['argv'][0],'-B',str(Path(wapi.__file__).resolve()),'--config',str(waiter.config_path)]
    require(launch['argv']==ident['argv']==expected and a.same_identity(stamp,ident)
        and launch['registration_sha256']==stamp['registration_sha256']==waiter.regsha
        and launch['config_sha256']==stamp['config_sha256']==a.sha(waiter.config_path),'Prescreen waiter launch/identity changed')
    require(p.exited(ident,state_reader),'Prescreen waiter remains live or unreaped')
    require(not (base/'failure.json').exists() and not (base/'STOP').exists()
        and not (waiter.out/'STOP').exists(),'Prescreen waiter/H failed or stopped')
    prior=wapi.verify_coarse_closed(a,p,waiter.prior,waiter.prior_launch,waiter.priorid,state_reader)
    gatepath=base/'release_gate.json';gate=a.read(gatepath)
    require(gate['registration_sha256']==waiter.regsha and all(gate.get(k)==v for k,v in prior.items()),
            'Prescreen coarse release gate differs from complete current evidence')
    lp,ep,log=base/'owner_launch.json',base/'owner_exit.json',base/'owner.log'
    started,ended=a.read(lp),a.read(ep);oid=started['identity']
    argv=[c['python'],'-B',c['stage_owner'],'--config',c['next_owner_config']]
    require(started['registration_sha256']==ended['registration_sha256']==waiter.regsha
        and started['argv']==oid['argv']==argv and ended['identity']==oid and ended['exit_code']==0
        and int(oid['uid'])==int(ident['uid']) and ended['closed_log_sha256']==a.sha(log),
        'Prescreen owner launch/exit/closed-log mismatch')
    require(p.exited(oid,state_reader),'Prescreen owner remains live or unreaped')
    closure=p.verify_batch(a,c['next_owner_config'],c['registration'],ident['uid'],state_reader)
    cp=base/'completion.json';done=a.read(cp)
    require(done['status']=='H_COARSE_WAIT_AND_PRESCREEN_COMPLETE' and done['registration_sha256']==waiter.regsha
        and done['output_bindings']==closure['bindings'] and done['prescreen_status']=='H_PRESCREEN_COMPLETE_FINAL'
        and done['ready_for_real_calibration'] is True and done['refinement_requested'] is False,
        'Prescreen did not finish with the original FINAL shortlist')
    for key in ('refinement_started','GPU_used','H_full_delivery_claimed','C_started','holdout_started'):
        require(done[key] is False,'Unexpected prerequisite scope: '+key)
    require(done['new_packet_decodes']==done['new_visual_inference']==0,'Prescreen performed undeclared inference')
    final=a.read(waiter.job['completion'])
    require(final['status']=='H_PRESCREEN_COMPLETE_FINAL' and final['ready_for_real_calibration'] is True
        and final['registration_sha256']==waiter.regsha,'Final scientific prescreen receipt differs')
    a.verify(final['outputs']);a.verify(final['source_bindings']);a.verify(final['input_bindings'])
    require(closure['bindings'].get(waiter.job['completion'])==a.sha(waiter.job['completion']),
            'Final prescreen receipt is not closed by its owner')
    # A source-stage STOP/failure must not be disguised by later completion.
    for cfg in (waiter.prior.old,waiter.prior.next,waiter.next):
        for stage in cfg['stages']:
            for job in stage['jobs']:
                for name in ('failure.json','merge_failure.json','STOP'):
                    require(not (Path(job['out'])/name).exists(),'Prerequisite stage failure/STOP: '+job['out'])
    bindings=merge(prior['bindings'],closure['bindings'],bind((gatepath,lp,ep,log,cp)))
    return dict(status='ALL_H_INITIAL_COARSE_PRESCREEN_PROCESSES_CLOSED',bindings=bindings,
        waiter_identity=ident,owner_identity=oid,
        child_identities=prior['child_identities']+closure['child_identities'],prescreen_completion=waiter.job['completion'])


def contract(core,protocol,catalogue,shortlist):
    """No execution SHA here: the frozen CPU driver injects it after seal."""
    return dict(status='H_PAYLOAD_CPU_ENGINEERING_SEALED',core_source_sha256=sha(core.__file__),
        protocol_canonical_sha256=core.digest(protocol),catalogue_canonical_sha256=core.digest(catalogue),
        shortlist_canonical_sha256=core.digest(shortlist),engineering_choices_sha256=core.digest(core.ENGINEERING_CHOICES),
        execution_registration_binding='Read actual registration SHA and inject at runtime; never stored in this pre-registration contract')


def owner_plan(request,previous,payload_config,registration):
    cfg=copy.deepcopy(previous);out=request['cpu_out'];entry=str(Path(request['runtime_dir'])/'h_payload_driver.py')
    cfg.update(owner_out=request['execution_dir'],registration=registration)
    cfg['cpu_affinities']=copy.deepcopy(request['cpu_affinities'])
    cfg['stages']=[dict(id='initial_true200',resource='cpu',requires=[],max_seconds=request['max_worker_seconds'],jobs=[
        dict(id='payload-'+str(i),argv=[request['python'],'-B',entry,'--config',payload_config,'--stage','worker','--worker-index',str(i)],
             cwd=request['root'],out=str(Path(out)/f'worker_{i}'),completion=str(Path(out)/f'worker_{i}'/'completion.json'),
             accepted_statuses=['H_INITIAL_TRUE200_CPU_WORKER_COMPLETE'],
             receipt_expect=dict(worker_index=i,workers=2,images_scored=False,GPU_used=False,development_used=False)) for i in range(2)]),
        dict(id='report',resource='cpu',requires=['initial_true200'],max_seconds=1800,jobs=[
            dict(id='payload-merge',argv=[request['python'],'-B',entry,'--config',payload_config,'--stage','merge'],
                 cwd=request['root'],out=out,completion=str(Path(out)/'completion.json'),
                 accepted_statuses=['H_INITIAL_TRUE200_CPU_RECEIVE_COMPLETE'],
                 receipt_expect=dict(source_count=200,workers=2,images_scored=False,GPU_used=False,development_used=False))])]
    return cfg


def prepare(request_path):
    request_path=Path(request_path).resolve();r=read(request_path)
    require(r['schema']=='H_INITIAL_PAYLOAD_REGISTRATION_REQUEST_V1','Wrong preparation request')
    require(sys.platform.startswith('linux'),'Actual registration gates require Linux process evidence')
    require(time.time()<DEADLINE,'Original H deadline already expired')
    runtime=Path(r['runtime_dir']);require(runtime.is_absolute(),'Absolute runtime required')
    records=r['prescreen_wait'];paths={k:pinned(records[k]) for k in ('config','launch','identity')}
    wc=read(paths['config']);wr=read(wc['registration']);prebound=merge(wr['source_bindings'],wr['input_bindings'])
    for key in ('stage_owner','wait_then_coarse'):
        require(prebound.get(wc[key])==sha(wc[key]),'Frozen predecessor verifier not bound')
    wp=str(runtime/'wait_then_prescreen.py');require(prebound.get(wp)==sha(wp),'Frozen prescreen waiter not bound')
    wapi=module(wp,'registered_initial_prescreen_verifier');w=wapi.Waiter(paths['config']);a=w.a
    require(w.cfg['root']==r['root'] and str(w.out)==r['H_out'],'Different root/H scope')
    closure=verify_prescreen_closed(w,wapi,read(paths['launch']),read(paths['identity']),a.raw_process_state)
    require(str(paths['identity'])==str(w.waiter_out/'identity.json'),'Wrong prescreen identity path')
    # Reject concurrent/new budget work; no phase is borrowed and no retry occurs.
    budgetpath=w.next['budget_path'];breg=w.next['budget_registration'];bsha=a.sha(breg)
    require(w.next['phase_limits']==PHASES,'Original H phase limits changed')
    before=a.budget_snapshot(budgetpath,bsha,PHASES,quiescent=True)
    require(before['created'] and before['phase_charged']['initial_true200']==0,'Initial payload phase was already attempted')
    require(before['phase_charged']['development']==0 and before['development_remaining']==13200,'Development reserve already used')
    # Bind actual tested prepared modules before importing their pure helpers.
    sources=dict(w.reg['source_bindings']);inputs=merge(w.reg['input_bindings'],closure['bindings'],
        bind((request_path,*paths.values(),Path(w.cfg['registration']),Path(w.cfg['next_owner_config']))))
    tested={}
    for record in r['prepared_qualifications']:
        p=pinned(record);q=read(p)
        require(q['status'] in ('PREPARED_PAYLOAD_CPU_QUALIFICATION_PASS','PREPARED_SELECTION_RENDER_CPU_QUALIFICATION_PASS')
            and q['actual_H_decoder_calls']==0 and q['GPU_used'] is False and q['execution_registered'] is False
            and all(x['exit_code']==0 for x in q['results']),'Prepared module qualification failed/differs')
        a.verify(q['source_bindings']);tested=merge(tested,q['source_bindings']);inputs[str(p)]=sha(p)
    for name in ('h_payload_driver.py','h_payload_cpu.py'):
        p=str(runtime/name);require(tested.get(p)==sha(p),'Prepared payload source lacks matching PASS evidence')
    sources=merge(sources,tested,r.get('additional_source_bindings',{}),bind((Path(__file__).resolve(),)))
    a.verify(sources)
    for path in (str(runtime),r['legacy_runtime'],str(Path(r['root'])/'src')):sys.path.insert(0,path)
    cpu=module(runtime/'h_payload_driver.py','registered_initial_driver_preflight')
    core=module(runtime/'h_payload_cpu.py','registered_initial_core_preflight')
    science=w.science;final=read(w.job['completion']);shortpath=str(Path(science['out'])/'shortlist.json')
    require(final['outputs'].get(shortpath)==sha(shortpath),'Final shortlist is unsealed')
    refinements=str(Path(science['out'])/'refinement_requests.json');rr=read(refinements)
    require(final['outputs'].get(refinements)==sha(refinements) and rr['requests']==[]
        and rr['requested_body_calls']==0,'This helper only admits coarse-direct FINAL with no refinement')
    shortlist=read(shortpath);candidates=cpu.check_shortlist(shortlist)
    require(shortlist['registration_sha256']==w.regsha and shortlist['source_ids']==read(science['source200'])['source_ids'],
            'Final shortlist/source order differs')
    maximum=2*len(candidates)*200*3;require(maximum<=19200<PHASES['initial_true200'],'Initial finite decode cap differs')
    initial=w.prior.old
    srcjob=next(s for s in initial['stages'] if s['id']=='source')['jobs'][0]
    qjob=next(s for s in initial['stages'] if s['id']=='qualification')['jobs'][0]
    srcpath=command_config(srcjob['argv']);qpath=command_config(qjob['argv']);sc=read(srcpath);qc=read(qpath)
    require(qc['ledger']==budgetpath and qc['legacy_runtime']==r['legacy_runtime'] and sc['source200']==science['source200'],
            'Frozen source/qualification input identity differs')
    execution=Path(r['execution_dir']);out=Path(r['cpu_out']);h=Path(r['H_out'])
    for p in (execution,out):require(p.is_absolute() and h in p.parents and not p.exists(),'New independent output required: '+str(p))
    require(execution not in out.parents and out not in execution.parents and execution!=out,'Evidence/science directories overlap')
    require(Path(r['python']).is_absolute() and Path(r['python']).is_file(),'CPU interpreter absent')
    require(0<r['max_worker_seconds']<=21600,'Unbounded worker runtime')
    affin=r['cpu_affinities'];require(len(affin)==2 and all(len(set(p))==len(p)==2 for p in affin)
        and len(set(sum(affin,[])))==4 and set(sum(affin,[]))<=set(os.sched_getaffinity(0)),
        'Two disjoint available two-core CPU workers required')
    regpath=execution/'execution_registration.json';pc=execution/'payload_config.json';oc=execution/'owner_config.json';ec=execution/'engineering_contract.json'
    payload=dict(root=r['root'],out=str(out),registration=str(regpath),protocol=science['protocol'],engineering_contract=str(ec),
        runtime_dir=str(runtime),legacy_runtime=r['legacy_runtime'],budget_registration=breg,budget_registration_sha256=bsha,
        ledger=budgetpath,ledger_module=qc['ledger_module'],catalogue=science['catalogue'],reference_qualification=qc['reference_qualification'],
        qualification_completion=science['qualification_completion'],shortlist=shortpath,prescreen_completion=w.job['completion'],
        source_completion=str(Path(science['source_dir'])/'completion.json'),S1_completion=str(Path(sc['S1'])/'completion.json'),
        S1_assets_completion=str(Path(sc['S1'])/'export-assets'/'completion.json'),source200=science['source200'],
        phase_limits=PHASES,workers=2,cpu_affinities=affin,max_worker_seconds=r['max_worker_seconds'],stop_file=str(h/'STOP'))
    original=read(payload['protocol']);catalogue=read(payload['catalogue']);choices=contract(core,original,catalogue,shortlist)
    # Pure schema/canonical validation only. The transient SHA below is not a
    # sealed execution identity and is never written to the contract.
    for c in candidates:core.validate_contract(original,catalogue,shortlist,dict(choices,execution_registration_sha256='0'*64),
                                              c,shortlist['source_ids'][0],0)
    for key in ('protocol','catalogue','shortlist','source200','reference_qualification','qualification_completion',
                'prescreen_completion','source_completion','S1_completion','S1_assets_completion','budget_registration','ledger_module'):
        inputs[payload[key]]=sha(payload[key])
    inputs=merge(inputs,bind((srcpath,qpath)))
    for key in ('qualification_completion','prescreen_completion','source_completion','S1_completion','S1_assets_completion'):
        cp=read(payload[key]);a.verify(cp['outputs']);inputs=merge(inputs,cp['outputs'])
        for field in ('input_bindings','source_bindings'):
            if field in cp:
                a.verify(cp[field]);inputs=merge(inputs,cp[field])
    # Full source/target chain: the final driver preflight will also check its
    # exact200 checkpoint population before this helper issues a launchable seal.
    owner=owner_plan(r,w.next,str(pc),str(regpath))
    owner['prior_qualification_gate']=dict(path=payload['qualification_completion'],sha256=sha(payload['qualification_completion']),
        accepted_statuses=['H_PHY_QUALIFICATION_PASS'],receipt_registration=w.prior.initial['registration'])
    a.verify(sources);a.verify(inputs)
    require(a.budget_snapshot(budgetpath,bsha,PHASES,quiescent=True)==before,'Budget changed during preparation')
    execution.mkdir()  # Nothing new is created before all existing gates pass.
    try:
        gatepath=execution/'predecessor_closure.json';save(gatepath,dict(closure,budget=before,new_packet_decodes=0,GPU_used=False))
        save(ec,choices);save(pc,payload);save(oc,owner)
        inputs=merge(inputs,bind((gatepath,ec,pc,oc)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,
            owner_config_sha256=sha(oc),source_bindings=sources,input_bindings=inputs,
            allowed_stage_ids=['initial_true200','report'],phase_limits=PHASES,
            scientific_protocol_sha256=sha(payload['protocol']),shortlist_sha256=sha(shortpath),
            qualification_start_unix=owner['qualification_started_unix'],qualification_deadline_unix=DEADLINE,
            maximum_sources=200,maximum_slots=16,calibration_noise_seeds=[6101,6102,6103],
            actual_candidate_count=len(candidates),maximum_actual_frames=maximum//2,maximum_actual_packet_calls=maximum,
            phase_cap=20000,unused_phase_capacity_not_reallocated=20000-maximum,
            CPU_workers=2,threads_per_worker=2,nice=15,GPU_jobs=0,
            implemented_scope='Frozen initial_true200 CPU actual reception, then receipt-only merge; no canonical source/image stage',
            engineering_contract_sha256=sha(ec),scientific_protocol_modified=False,
            budget_registration_sha256=bsha,predecessor_closure_sha256=sha(gatepath),
            GPU_automatic=False,policy_selection=False,development_used=False,holdout_started=False,
            C_started=False,H_full_delivery_claimed=False,created_unix=time.time())
        a.validate_config(owner,reg,sha(oc));a.verify(sources);a.verify(inputs);save(regpath,reg)
        # Invoke only the frozen read-only loader. It cannot construct a backend
        # or start worker processes. Failure leaves the seal non-launchable.
        ctx=cpu.load_registered(str(pc))
        require(ctx['contract']['execution_registration_sha256']==sha(regpath),'Runtime contract injection failed')
        require(a.budget_snapshot(budgetpath,bsha,PHASES,quiescent=True)==before,'Read-only preflight changed the ledger')
        done=dict(status='H_INITIAL_PAYLOAD_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(regpath),
            owner_config_sha256=sha(oc),outputs=bind((regpath,pc,oc,ec,gatepath)),new_packet_decodes=0,
            workers_started=False,GPU_used=False,read_only_driver_preflight='PASS',budget=before,
            required_launch_condition='Require this registration_completion status and exact outputs; reject registration_failure.json or a missing completion. Reverify predecessor exits and unchanged quiescent budget immediately before a separate launch')
        save(execution/'registration_completion.json',done);return done
    except BaseException as error:
        failure=execution/'registration_failure.json'
        if not failure.exists():save(failure,dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),traceback=traceback.format_exc(),
                                                workers_started=False,new_packet_decodes=0,GPU_used=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    result=prepare(args.config);print(json.dumps({k:result[k] for k in ('status','registration_sha256','owner_config_sha256')}))


if __name__=='__main__':main()
