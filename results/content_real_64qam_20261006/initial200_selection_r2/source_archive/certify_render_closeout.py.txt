"""Certify completed render output after an owner exit race; never run science.

Register and run are separate explicit operations. Original failure, registration,
outputs and owner receipts remain untouched. Only the two exact archived global
and renderer STOPs may move after a registered, successful full read-only audit.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback

DEADLINE=1791564605.9549868
PHASES={'qualification':2000,'coarse':49152,'refine':12288,'initial_true200':20000,
        'whole_calibration':48000,'partial_calibration':36000,'development':13200,'engineering_reserve':19360}
CHARGES=dict.fromkeys(PHASES,0); CHARGES.update(qualification=1608,coarse=49152,initial_true200=19200)
FINAL_STATUSES={'RAW_SOURCE_DECODED','ARITHMETIC_SOURCE_DECODED','WIRE_REJECT_GRAY','ARITHMETIC_SOURCE_INVALID_GRAY'}
REG_STATUS='H_RENDER_CLOSEOUT_READ_ONLY_REGISTERED'
CERT_STATUS='H_RENDER_OUTPUTS_AND_EXIT_CERTIFIED_AFTER_OWNER_RACE'
RELEASE_STATUS='H_RENDER_CLOSEOUT_EXACT_STOP_ARCHIVED'
DONE_STATUS='H_RENDER_CLOSEOUT_COMPLETE_STOP_RELEASED'


def require(ok,message):
    if not ok: raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def verify(values):
    require(isinstance(values,dict),'Expected path/SHA map')
    for p,s in values.items(): require(Path(p).is_absolute() and sha(p)==s,'Immutable SHA differs: '+p)

def bind(paths): return {str(p):sha(p) for p in paths}

def pin(record):
    path=Path(record['path']); require(path.is_absolute() and sha(path)==record['sha256'],'Pinned input differs: '+str(path)); return path

def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting binding: '+p); result[p]=s
    return result

def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())

def record(path): return dict(path=str(path),sha256=sha(path))

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value; spec.loader.exec_module(value); return value

def check_budget(value):
    require(value.get('created') is True and value['charged']==69960 and value['phase_charged']==CHARGES
            and value['unresolved']==0 and value.get('failed',0)==0 and value['development_remaining']==13200,
            'Original 69960-packet quiescent ledger differs')

def require_exited(identity,state_reader):
    # Even a matching zombie must be reaped before issuing a continuation gate.
    require(state_reader(identity['pid']) is None,'Owned PID still exists or was reused; no exit certificate')

def gpu_empty():
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    require(len(devices)==1 and not processes,'Single GPU must have no compute processes')
    return dict(device_uuids=devices,compute_processes=[],verified_unix=time.time())

def audit_archive(archive,stop,required):
    require(archive['status']=='SECOND_FAILURE_PRESERVED_STOP_NOT_RELEASED' and len(archive['files'])==22
            and archive['original_files_unchanged'] is True and archive['STOP_released'] is False,'Wrong second-failure archive')
    require(set(map(str,required))<=set(archive['files']),'Archive omits required failure/exit/STOP evidence')
    bindings={}
    for original,row in archive['files'].items():
        require(sha(original)==row['sha256']==sha(row['copy']),'Archived evidence changed: '+original)
        bindings[row['copy']]=row['sha256']
        if original!=str(stop):bindings[original]=row['sha256']
    # Moving STOP after registration must not invalidate other immutable inputs.
    require(str(stop) not in bindings,'Global STOP must use its explicit move contract')
    return bindings

def audit_grid(render_cfg,done,shortlist):
    ids=shortlist['source_ids'];candidates=shortlist['whole_candidates']; out=Path(render_cfg['out'])
    require(len(ids)==len(set(ids))==200 and len(candidates)==16,'Expected 200 sources and 16 frozen whole candidates')
    byslot={c['slot']:c for c in candidates};require(set(byslot)==set(range(16)),'Frozen candidate slots differ')
    outputs=done['outputs']; metrics=out/'frame_metrics.json'
    require(outputs.get(str(metrics))==sha(metrics),'Combined metrics unsealed'); rows=read(metrics)
    require(len(rows)==9600,'Expected 9600 completed receive frames')
    seen=set();grouped={i:[] for i in range(200)};events=set()
    for row in rows:
        i=row['source_index'];slot=row['slot'];seed=row['noise_seed']
        require(type(i) is int and i in grouped and row['source_id']==ids[i]
                and type(slot) is int and slot in byslot and seed in (6101,6102,6103),'Unexpected source/slot/noise frame')
        key=(i,slot,seed); require(key not in seen,'Duplicate frame');seen.add(key)
        candidate=byslot[slot]
        for k in ('candidate_id','arm','target_m','q','nominal_rate','snr_db'):
            require(row[k]==candidate[k],'Frozen candidate field differs: '+k)
        require(row.get('K',0)==candidate.get('K',0)==0,'Initial phase is whole-prefix only')
        require(row['event_id'] not in events,'Duplicate receive event');events.add(row['event_id'])
        rx=row['rx_summary'];status=row['source_status']
        require(status in FINAL_STATUSES and rx['source_status']==status and rx['source_decode_complete'] is True
                and type(row['gray']) is bool and row['gray']==rx['gray']==status.endswith('_GRAY')
                and rx['status']=='H_ACTUAL_RX_RECONSTRUCTION_COMPLETE' and rx['new_packet_decodes']==0,
                'Incomplete canonical receive status')
        require(all(rx[k] is False for k in ('target_image_used_for_reconstruction','truth_correction','cached_clean_image_used')),
                'Receiver used truth or a clean-image substitute')
        for k in ('image_sha256','receiver_view_sha256'):
            s=row[k];require(isinstance(s,str) and len(s)==64 and all(x in '0123456789abcdef' for x in s)
                            and rx[k]==s,'Image/receive evidence differs')
        require(0<row['mse']<=1 and math.isfinite(row['mse']) and math.isfinite(row['psnr_db'])
                and abs(row['psnr_db']+10*math.log10(row['mse']))<=1e-9,'Invalid or inconsistent image score')
        require(isinstance(row['image_key'],str) and row['image_key'],'Missing archive image key')
        grouped[i].append(row)
    require(len(seen)==200*16*3,'Incomplete Cartesian frame grid')
    require({p.name for p in (out/'source_checkpoints').glob('*.json')}=={f'{i:04d}.json' for i in range(200)},'Checkpoint count differs')
    inputs={}
    for i,sid in enumerate(ids):
        cp=out/'source_checkpoints'/f'{i:04d}.json';sp=out/'sources'/f'{i:04d}.json';image=out/'images'/f'{i:04d}.npz'
        require(all(outputs.get(str(p))==sha(p) for p in (cp,sp,image)),'Unsealed per-source outputs')
        c=read(cp)
        require(c['status']=='H_INITIAL_RX_SOURCE_COMPLETE' and c['source_index']==i and c['source_id']==sid
                and c['frame_count']==len(grouped[i])==48 and c['source_decode_complete'] is True
                and c['images_scored'] is True and c['new_packet_decodes']==0
                and c['registration_sha256']==done['registration_sha256'] and c['config_sha256']==done['config_sha256'],
                'Source completion differs')
        require(c['outputs']==bind((image,sp)) and read(sp)==grouped[i],'Source rows differ from combined table')
        require(all(r['image_archive']==str(image) for r in grouped[i]),'Image archive crosses source')
        verify(c['input_bindings']);inputs=merge(inputs,c['input_bindings'])
    return dict(status='ALL_200_SOURCES_9600_FRAMES_AND_ARCHIVE_SHAS_VERIFIED',source_count=200,frame_count=9600,
                source_input_bindings=inputs,RGB_arrays_redecoded=False,archive_bytes_SHA_revalidated=True)

def audit(request,*,state_reader=None,gpu_checker=gpu_empty):
    paths={k:pin(request[k]) for k in ('original_owner_config','original_registration','original_launch',
        'render_config','diagnosis','failure_archive','renderer_stop_evidence','prepared_qualification')}
    rcfg=read(paths['render_config']);cfg=read(paths['original_owner_config']);reg=read(paths['original_registration'])
    rsha=sha(paths['original_registration']);bound=merge(reg['source_bindings'],reg['input_bindings']);verify(bound)
    api_path=Path(rcfg['owner_module']);require(bound.get(str(api_path))==sha(api_path),'Frozen owner verifier unbound')
    api=load(api_path,'closeout_original_owner');state_reader=state_reader or api.raw_process_state
    api.validate_config(cfg,reg,sha(paths['original_owner_config']))
    require(request['root']==rcfg['root']==cfg['root'],'Request root differs from registered original')
    require(cfg['registration']==str(paths['original_registration']) and rcfg['registration']==str(paths['original_registration'])
        and rcfg['render_owner_config']==str(paths['original_owner_config']) and reg['allowed_stage_ids']==['render']
        and len(cfg['stages'])==1 and cfg['phase_limits']==PHASES,'Wrong original render scope')
    stage=cfg['stages'][0];require(stage['id']=='render' and stage['resource']=='gpu' and len(stage['jobs'])==1,'Wrong visual job')
    job=stage['jobs'][0];ownerout=Path(cfg['owner_out']);out=Path(rcfg['out']);h=Path(cfg['out']);stop=h/'STOP'
    require(paths['original_launch']==ownerout/'launch.json' and Path(job['out'])==out
            and Path(job['completion'])==out/'completion.json' and rcfg['stop_file']==str(stop),'Original paths differ')
    require(job['argv'].count('--config')==1 and job['argv'][job['argv'].index('--config')+1]==str(paths['render_config']),
            'Original worker config differs')
    lp=ownerout/'launch.json';ip=ownerout/'owner_identity.json';fp=ownerout/'failure.json';dp=ownerout/'drain_receipt.json'
    wp=ownerout/'stages/render/workers'/job['id'];wl=wp/'launch.json';we=wp/'exit_receipt.json';log=wp/'worker.log'
    launch=read(lp);ident=launch['identity'];stored=read(ip);exit_value=read(we);worker=read(wl)
    expected=[launch['argv'][0],'-B',str(api_path),'--config',str(paths['original_owner_config'])]
    require(ident['argv']==launch['argv']==expected and api.same_identity(ident,stored)
        and launch['registration_sha256']==stored['registration_sha256']==rsha
        and launch['owner_config_sha256']==stored['config_sha256']==sha(paths['original_owner_config']), 'Owner identity changed')
    child=exit_value['identity']
    require(exit_value['exit_code']==0 and exit_value['job_id']==job['id'] and child['argv']==job['argv']
        and api.same_identity(child,worker['identity']) and worker['argv']==job['argv']
        and worker['registration_sha256']==exit_value['registration_sha256']==rsha
        and int(ident['uid'])==int(child['uid'])==os.getuid() and sha(log)==exit_value['closed_log_sha256'],
        'Worker exit/identity/log differs')
    require_exited(ident,state_reader);require_exited(child,state_reader)
    failure=read(fp);drain=read(dp)
    require(failure['registration_sha256']==rsha and failure['current_stage']=='render'
        and 'check_live' in failure['traceback'] and 'RuntimeError: Process is no longer live' in failure['traceback']
        and len(failure['active'])==1 and failure['active'][0]['job']==job['id']
        and api.same_identity(failure['active'][0]['identity'],child),'Failure is not the diagnosed owner exit race')
    require(drain['status']=='ALL_OWNED_CHILDREN_EXITED' and drain['registration_sha256']==rsha
        and drain['force_kill'] is False and not (ownerout/'completion.json').exists()
        and not (out/'failure.json').exists() and 'failure' not in exit_value,'Original scope has another failure or manufactured success')
    archive=read(paths['failure_archive']); archive_bindings=audit_archive(archive,stop,(fp,dp,lp,ip,wl,we,log,stop,out/'completion.json'))
    require(not (ownerout/'STOP').exists(),'Unexpected owner-local STOP')
    local_stop=out/'STOP';supplement=read(paths['renderer_stop_evidence'])
    require(supplement['status']=='RENDER_LOCAL_STOP_EVIDENCE_PRESERVED'
        and supplement['original_path']==str(local_stop) and supplement['owner_failure_sha256']==sha(fp)
        and supplement['original_failure_archive_receipt_sha256']==sha(paths['failure_archive'])
        and supplement['diagnosis_sha256']==sha(paths['diagnosis'])
        and supplement['original_files_unchanged'] is True and supplement['STOP_released'] is False
        and sha(local_stop)==sha(supplement['preserved_path'])==supplement['sha256'],
        'Renderer-local STOP lacks matching independent supplemental evidence')
    require(supplement['preserved_path']!=str(local_stop),'Supplement must preserve a distinct STOP copy')
    archive_bindings=merge(archive_bindings,{supplement['preserved_path']:supplement['sha256']})
    diagnosis=read(paths['diagnosis'])
    require(diagnosis['status']=='OWNER_EXIT_FAILURE_DIAGNOSED_OUTPUTS_VERIFIED_AWAIT_INDEPENDENT_CLOSEOUT'
        and diagnosis['failure_archive_receipt_sha256']==sha(paths['failure_archive'])
        and diagnosis['owner_source_sha256']==sha(api_path) and diagnosis['original_owner_success'] is False
        and diagnosis['original_registration_sha256']==rsha and diagnosis['original_owner_failure_sha256']==sha(fp)
        and diagnosis['worker_exit_receipt_sha256']==sha(we),'Diagnosis not bound to this failure')
    cp=out/'completion.json'
    require(exit_value['completion']==str(cp) and exit_value['completion_sha256']==sha(cp)==diagnosis['render_completion_sha256'],
            'Worker completion evidence differs')
    # This is the exact validation skipped by original drain()->finish(False).
    done=api.receipt(cp,job['accepted_statuses'],rsha,job.get('receipt_expect'),job['out'])
    require(done['status']=='H_INITIAL_TRUE200_RX_COMPLETE' and done['source_count']==200 and done['frame_count']==9600
        and done['config_sha256']==sha(paths['render_config']) and done['images_scored'] is True
        and done['source_decode_complete'] is True and done['new_packet_decodes']==0 and done['budget_unchanged'] is True
        and done['policy_selection'] is False and done['development_used'] is False and done['holdout_used'] is False,
        'Scientific receipt is incomplete or out of scope')
    for key in ('outputs','input_bindings','source_bindings','predecessor_closure_bindings','visual_launch_bindings'):verify(done[key])
    require(done['cpu_completion_sha256']==sha(rcfg['cpu_completion']) and done['shortlist_sha256']==sha(rcfg['shortlist']),
            'CPU predecessor or shortlist changed')
    grid=audit_grid(rcfg,done,read(rcfg['shortlist']))
    budget=api.budget_snapshot(cfg['budget_path'],cfg['budget_registration_sha256'],PHASES,quiescent=True);check_budget(budget)
    require(budget==diagnosis['budget'],'Budget changed since diagnosis')
    gpu=gpu_checker();require_exited(ident,state_reader);require_exited(child,state_reader)
    qualification=read(paths['prepared_qualification'])
    require(qualification['status']=='H_RENDER_CLOSEOUT_CPU_QUALIFICATION_PASS' and qualification['results']
        and all(r['exit_code']==0 for r in qualification['results']) and qualification['GPU_used'] is False
        and qualification['new_packet_decodes']==0,'Closeout helper qualification not PASS')
    verify(qualification['source_bindings'])
    require(qualification['source_bindings'].get(str(Path(__file__).resolve()))==sha(__file__),'Helper source not qualified')
    bindings=merge(bound,archive_bindings,bind(paths.values()),bind((fp,dp,lp,ip,wl,we,log,cp)),done['outputs'],
        done['predecessor_closure_bindings'],done['visual_launch_bindings'],grid['source_input_bindings'])
    require(str(stop) not in bindings and str(local_stop) not in bindings,'Movable STOP unexpectedly immutable-bound')
    return dict(api=api,paths=paths,original_registration_sha256=rsha,owner_identity=ident,worker_identity=child,
        render_completion=record(cp),original_owner_config=record(paths['original_owner_config']),original_launch=record(lp),
        worker_exit_receipt=record(we),original_failure=record(fp),bindings=bindings,outputs=done['outputs'],
        source_bindings=merge(reg['source_bindings'],qualification['source_bindings']),budget=budget,
        grid_audit=grid,gpu_admission=gpu,expected_stop=record(stop),expected_stops=[record(stop),record(local_stop)],
        h=h,owner_config=cfg,qualification=qualification)

def execution_path(request):
    out=Path(request['execution_dir']);h=Path(read(pin(request['original_owner_config']))['out'])
    require(out.is_absolute() and h in out.parents,'Closeout must use a new H evidence directory')
    for key in ('original_owner_config','render_config'):
        old=pin(request[key]).parent
        require(out!=old and old not in out.parents and out not in old.parents,'Closeout overlaps original evidence')
    return out

def environment(request):
    require(sys.platform.startswith('linux'),'Actual closeout requires Linux process and GPU inventory evidence')
    require(request['schema']=='H_RENDER_CLOSEOUT_REQUEST_V1' and time.time()<DEADLINE,'Wrong request or expired H deadline')
    require(len(request['cpu_affinity'])==len(set(request['cpu_affinity']))==2,'Exactly two CPU cores required')
    os.sched_setaffinity(0,set(request['cpu_affinity']))
    if os.getpriority(os.PRIO_PROCESS,0)<15:os.setpriority(os.PRIO_PROCESS,0,15)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')

def register(request_path):
    rpath=Path(request_path).resolve();r=read(rpath);environment(r);out=execution_path(r)
    require(not out.exists(),'Closeout attempt exists; preserve and diagnose, no retry');out.mkdir()
    try:
        ctx=audit(r)
        reg=dict(status=REG_STATUS,root=r['root'],out=str(out),request=record(rpath),
            original_registration_sha256=ctx['original_registration_sha256'],original_owner_success=False,
            input_bindings=merge(ctx['bindings'],bind((rpath,))),source_bindings=ctx['source_bindings'],
            expected_stop=ctx['expected_stop'],expected_stops=ctx['expected_stops'],budget_before=ctx['budget'],owner_identity=ctx['owner_identity'],
            worker_identity=ctx['worker_identity'],source_count=200,frame_count=9600,new_packet_decodes=0,
            new_visual_inference=0,GPU_used=False,scientific_protocol_modified=False,deadline_unix=DEADLINE,
            only_action='Read-only certification, then both exact registered global/renderer STOP archival; no original owner completion',
            created_unix=time.time())
        rp=out/'execution_registration.json';save(rp,reg)
        save(out/'registration_completion.json',dict(status='H_RENDER_CLOSEOUT_REGISTERED_NOT_RUN',
            registration_sha256=sha(rp),outputs=bind((rp,)),new_packet_decodes=0,new_visual_inference=0,
            original_owner_success=False,required_run_condition='No failure.json; same request/source/input/STOP SHA; rerun complete read-only audit'))
        return reg
    except BaseException:
        save(out/'failure.json',dict(stage='register',status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),
            new_packet_decodes=0,new_visual_inference=0));raise

def release_stop(ctx,out,certificate,regsha):
    require(len(ctx['expected_stops'])==2 and ctx['expected_stops'][0]==ctx['expected_stop'],'Exact two-STOP contract required')
    targets=(out/'released_STOP.original',out/'released_RENDER_STOP.original')
    for item,target in zip(ctx['expected_stops'],targets):
        require(sha(item['path'])==item['sha256'] and not target.exists(),'Registered STOP changed')
    require_exited(ctx['owner_identity'],ctx['api'].raw_process_state)
    require_exited(ctx['worker_identity'],ctx['api'].raw_process_state)
    markers=[]
    for item,target in zip(ctx['expected_stops'],targets):
        stop=Path(item['path']);require(sha(stop)==item['sha256'],'Registered STOP changed immediately before move')
        os.rename(stop,target)
        require(not stop.exists() and sha(target)==item['sha256'],'STOP move did not preserve bytes')
        markers.append(dict(original_path=str(stop),preserved_path=str(target),sha256=sha(target)))
        save(out/f'stop_release_{len(markers)-1}.json',dict(closeout_registration_sha256=regsha,**markers[-1]))
    value=dict(status=RELEASE_STATUS,certificate_sha256=sha(certificate),closeout_registration_sha256=regsha,
        **markers[0],released=True,released_markers=markers,
        original_owner_success=False,only_registered_two_STOPs_released=True,old_owner_and_worker_exited=True,
        budget_unchanged=True,new_packet_decodes=0,new_visual_inference=0)
    save(out/'release_receipt.json',value);return value

def run(request_path):
    rpath=Path(request_path).resolve();r=read(rpath);environment(r);out=execution_path(r)
    require(out.is_dir(),'Register this independent closeout before run')
    import fcntl
    with (out/'execution.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not any((out/n).exists() for n in ('failure.json','attempt.json','completion.json','closeout_completion.json')),
                'Closeout attempt exists; no automatic retry')
        try:
            rp=out/'execution_registration.json';reg=read(rp);regsha=sha(rp)
            require(reg['status']==REG_STATUS and reg['request']==record(rpath),'Closeout registration differs')
            seal=read(out/'registration_completion.json')
            require(seal['status']=='H_RENDER_CLOSEOUT_REGISTERED_NOT_RUN' and seal['registration_sha256']==regsha,'Missing registration gate')
            verify(seal['outputs']);verify(reg['source_bindings']);verify(reg['input_bindings'])
            save(out/'attempt.json',dict(status='READ_ONLY_CLOSEOUT_STARTED',registration_sha256=regsha,time=time.time()))
            ctx=audit(r)
            require(ctx['expected_stop']==reg['expected_stop'] and ctx['expected_stops']==reg['expected_stops'] and ctx['budget']==reg['budget_before']
                    and ctx['bindings'].items()<=reg['input_bindings'].items(),'Original chain changed after registration')
            require(time.time()<DEADLINE,'Original H deadline expired')
            fields=('original_registration_sha256','owner_identity','worker_identity','render_completion','original_owner_config',
                'original_launch','worker_exit_receipt','original_failure','bindings','outputs','source_bindings','budget','grid_audit','gpu_admission','expected_stops')
            certificate=dict(status=CERT_STATUS,closeout_registration_sha256=regsha,**{k:ctx[k] for k in fields},
                original_owner_success=False,scientific_outputs_complete=True,source_count=200,frame_count=9600,
                new_packet_decodes=0,new_visual_inference=0,GPU_used=False,scientific_protocol_modified=False,
                policy_selection=False,development_used=False,holdout_used=False,H_full_delivery_claimed=False,
                original_owner_completion_written=False,STOP_release_pending=True,
                downstream_gate='Require closeout_completion.json plus release_receipt.json; this certificate alone never releases continuation')
            cp=out/'completion.json';save(cp,certificate)
            # Recheck the same budget and both exited processes immediately before releasing STOP.
            cfg=ctx['owner_config'];budget=ctx['api'].budget_snapshot(cfg['budget_path'],cfg['budget_registration_sha256'],PHASES,quiescent=True)
            require(budget==ctx['budget'],'Budget changed before STOP release');gpu_empty()
            release_stop(ctx,out,cp,regsha)
            final=dict(status=DONE_STATUS,closeout_registration_sha256=regsha,original_owner_success=False,
                scientific_outputs_complete=True,source_count=200,frame_count=9600,budget=ctx['budget'],
                outputs=bind((cp,out/'release_receipt.json',out/'released_STOP.original',out/'released_RENDER_STOP.original',
                              out/'stop_release_0.json',out/'stop_release_1.json')),original_registration_sha256=ctx['original_registration_sha256'],
                new_packet_decodes=0,new_visual_inference=0,GPU_used=False,original_owner_completion_written=False,
                selection_started=False,H_full_delivery_claimed=False)
            save(out/'closeout_completion.json',final);return final
        except BaseException:
            if not (out/'failure.json').exists():save(out/'failure.json',dict(stage='run',status='FAILED_PRESERVE_NO_RETRY',
                traceback=traceback.format_exc(),new_packet_decodes=0,new_visual_inference=0))
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('register','run'),required=True);args=parser.parse_args()
    result=(register if args.stage=='register' else run)(args.config);print(json.dumps({'status':result['status']}))

if __name__=='__main__':main()
