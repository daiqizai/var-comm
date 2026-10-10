#!/usr/bin/env python3
"""Metadata registration and exclusive fixed16 T4 timing; never starts via import."""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import t1_phy as phy
from t1_codec_runtime import SourceCodec
from t4_timing_endpoint import EntropyEndpoint,TimingReservations,fresh_case,FIXED16,SNRS,WARMUPS,REPEATS
from t4_raw_endpoint import RawEndpoint

SCHEMA='WCL_T4_FIXED16_ONLINE_TIMING_V1'
METHODS=('RAW64_WHOLE','RAW64_PARTIAL')+phy.FAMILIES
HELPER_SHA='9ee58a09810fb92b04c14db77abcad03c81e6dd2e22c8672bcb99ebc045bddee'
FRAMES_PER_METHOD=16*3*(WARMUPS+REPEATS)
PACKET_CAP=len(METHODS)*FRAMES_PER_METHOD*2
WAITER_SHA='dfeba4b30aa7cc3febe67253d584c18c6e1ede11f41dbbe84f6a7fdfdd829d89'
STOP=False
def stop(*_):
    global STOP;STOP=True
def pin(path):return dict(path=str(Path(path).resolve()),sha256=phy.sha(path))
def checked(d):
    phy.require(set(d)=={'path','sha256'} and phy.sha(d['path'])==d['sha256'],'Changed descriptor: '+str(d))
    return phy.read(d['path'])
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(phy.canonical(value)+'\n');f.flush();os.fsync(f.fileno())
def status(path,value):
    path=Path(path);temp=path.with_suffix('.pending');temp.write_text(phy.canonical(value)+'\n');os.replace(temp,path)
def csvout(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
@contextlib.contextmanager
def exclusive(path):
    import fcntl
    with Path(path).open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def check_sources(records):
    phy.require([x['source_index'] for x in records]==list(FIXED16),'Historical fixed16 ordering required')
    result=[]
    for rec in records:
        p=Path(rec['archive']);phy.require('/development100_assets/' in str(p),'Only original development pixels')
        phy.require(phy.sha(p)==rec['archive_sha256'],'Original source archive changed')
        with np.load(p,allow_pickle=False) as z:
            phy.require(set(z.files)=={'pixels','tokens'},'Exact original source archive')
            pixels=z['pixels'].copy();tokens=z['tokens'].copy()
        phy.require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256),'Original image tensor shape/type')
        phy.require(hashlib.sha256(pixels.tobytes()).hexdigest()==rec['preprocessing_id'],'Original source pixels changed')
        phy.require(tokens.dtype==np.int64 and tokens.shape==(680,) and token_sha(tokens)==rec['tokens_sha256'],'Original token identity changed')
        result.append(dict(record=rec,pixels=pixels))
    return result
def token_sha(tokens):return hashlib.sha256(b'int64:680\0'+np.asarray(tokens,dtype='<i8').tobytes()).hexdigest()

def register_native_additions(root,baseline,bindings):
    """The original loader inventories all tracked sources, including later studies."""
    old=baseline['P_native_source_bindings']
    for path,expected in old.items():
        phy.require(bindings.get(path)==expected==phy.sha(path),'Original Native source changed: '+path)
    tracked=subprocess.check_output(['git','ls-files','*.py','*.cpp','configs/*.yaml','configs/*.json'],cwd=root,text=True).splitlines()
    return {str(Path(root)/p):phy.sha(Path(root)/p) for p in sorted(tracked) if str(Path(root)/p) not in old}

def validate_native(native,baseline,bindings,additions):
    old=baseline['P_native_source_bindings']
    phy.require(native.loaded['identity']==baseline['P_native_runtime']['frozen_visual_identity']
        and native.flags==baseline['P_native_runtime']['numerical_runtime'],'Original Native weights/numerical runtime changed')
    phy.require(not set(old).intersection(additions),'Registered additions cannot replace original sources')
    phy.require(native.driver_bindings==dict(old,**additions),'Native source inventory differs from registered old-plus-new inventory')
    for path,expected in old.items():
        phy.require(bindings.get(path)==expected==phy.sha(path),'Original Native source not pinned: '+path)
    phy.verify_bindings(additions)
def frozen_entropy(path):
    f=phy.read(path);done=checked(f['calibration_completion'])
    phy.require(done['status']=='T1_FULL_CALIBRATION_COMPLETE' and done['source_count']==1000 and done['noise_count']==3
        and set(done['families'])==set(phy.FAMILIES) and done['holdout_used'] is False,'Both full calibration families required')
    rankings=Path(done['rankings_path']);phy.require(phy.sha(rankings)==done['outputs'][str(rankings)],'Frozen ranking CSV changed')
    with rankings.open(newline='') as stream:rows=list(csv.DictReader(stream))
    wanted={(family,snr) for family in phy.FAMILIES for snr in SNRS}
    phy.require(len(f['rows'])==6 and {(p['family'],p['snr_db']) for p in f['rows']}==wanted,'Exactly six frozen entropy points')
    points={family:{} for family in phy.FAMILIES}
    for p in f['rows']:
        matches=[x for x in rows if x['family']==p['family'] and int(x['snr_db'])==p['snr_db'] and int(x['rank'])==1]
        phy.require(len(matches)==1,'One original-calibration winner required');winner=matches[0]
        for key in ('candidate_id','nominal_rate'):phy.require(p[key]==winner[key],'Policy differs from frozen full-calibration winner')
        for key in ('target_m','q'):phy.require(p[key]==int(winner[key]),'Winner resource field changed')
        points[p['family']][str(p['snr_db'])]={k:p[k] for k in ('family','snr_db','candidate_id','target_m','q','nominal_rate')}
    return points,done
def prepare(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve();old=phy.read(args.a3_request);cost=phy.read(args.raw_cost_request)
    phy.require(root==Path(old['root']) and out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'Original repository and new independent output required')
    phy.require(not out.exists() and 0<args.max_seconds<=14400,'Fresh bounded output required')
    phy.require(old['schema']=='A3_FIXED16_UNIFIED_TIMING_V1' and old['source_indices']==list(FIXED16),'Authenticated original timing population required')
    phy.require(cost['plan']['source_indices']==list(FIXED16) and cost['plan']['source_ids']==old['source_ids'],'Original fixed16 registration differs')
    check_sources(old['records']);points,done=frozen_entropy(args.frozen_policy)
    qual=phy.read(args.phy_qualification)
    phy.require(qual['status']=='PASS' and qual['packet_decode_count']==241 and qual['ledger']['unresolved']==0,'Completed real T1 PHY qualification first')
    phy.verify_bindings(qual['source_bindings']);phy.verify_bindings(qual['input_bindings']);phy.verify_bindings(qual['outputs'])
    static=phy.read(args.static_completion)
    phy.require(static['status']=='T1_STATIC_TRAIN20K_CDF_COMPLETE' and static['source_count']==20000,'Frozen static model required')
    phy.verify_bindings(static['outputs'])
    raw_policy=phy.read(cost['selected'])
    phy.require(raw_policy['status']=='POLICIES_FROZEN_ON_CALIBRATION1000' and not raw_policy['holdout_used'],'Original raw calibration policy required')
    raw_points={method:{} for method in METHODS[:2]}
    for method in raw_points:
        family=method.removeprefix('RAW64_')
        for snr in SNRS:
            matches=[p for p in cost['plan']['points'] if p['timing_family']==family and p['snr_db']==snr]
            phy.require(len(matches)==1,'One original frozen raw point');p=matches[0]
            winner=next(w for w in raw_policy['winners'] if w['family']==family and w['snr_db']==snr)
            phy.require(winner['candidate_id']==p['candidate_id'],'Raw anchor is not the original frozen winner')
            raw_points[method][str(snr)]=p
    phy.require(raw_points['RAW64_PARTIAL']['19']['K']==142,'Preserve original K142')
    keep=('python','pythonpath','threads','interop_threads','affinity','nice','visual_lock','baseline_request','raw_config','bindings',
        'executable_bindings','private_sionna_manifest','private_sionna_parent','sionna_source','helper_module',
        'source_driver','pair_module','timing_module','development_plan_module')
    legacy={k:old[k] for k in keep}
    native_additions=register_native_additions(root,checked(legacy['baseline_request']),legacy['bindings'])
    phy.require(phy.sha(legacy['helper_module'])==HELPER_SHA,'Original unified native helper changed')
    own=('t4_timing_owner.py','t4_timing_endpoint.py','t4_raw_endpoint.py','t1_codec_runtime.py','t1_entropy_core.py','t1_phy.py','run_recorded_child.py')
    phy.require(phy.sha(Path(__file__).with_name('run_recorded_child.py'))==WAITER_SHA,'Exact wait-only observer required')
    source_bindings={str(Path(__file__).with_name(n).resolve()):phy.sha(Path(__file__).with_name(n)) for n in own}
    inputs={str(Path(p).resolve()):phy.sha(p) for p in (args.a3_request,args.raw_cost_request,args.frozen_policy,args.phy_qualification,args.static_completion,cost['selected'])}
    for name in ('baseline_request','raw_config'):checked(legacy[name]);inputs[legacy[name]['path']]=legacy[name]['sha256']
    inputs[legacy['private_sionna_manifest']]=phy.sha(legacy['private_sionna_manifest'])
    r=dict(schema=SCHEMA,root=str(root),out=str(out),methods=list(METHODS),snrs=list(SNRS),source_indices=list(FIXED16),source_ids=old['source_ids'],
        records=old['records'],source_bindings=source_bindings,input_bindings=inputs,legacy=legacy,raw_points=raw_points,entropy_points=points,
        registered_native_source_additions=native_additions,
        frozen_policy=pin(args.frozen_policy),phy_qualification=pin(args.phy_qualification),static_completion=pin(args.static_completion),
        calibration_completion=phy.read(args.frozen_policy)['calibration_completion'],warmup_per_source_snr=WARMUPS,measured_per_source_snr=REPEATS,
        frames_per_method=FRAMES_PER_METHOD,total_frames=4*FRAMES_PER_METHOD,measured_frames=576,warmup_frames=576,packet_cap=PACKET_CAP,
        batch_size=1,precision='FP32',deadline_unix=args.deadline_unix,max_seconds=args.max_seconds,
        stop_files=[str(root/'STOP'),str(out/'STOP')],training_updates=0,policy_selection=False,old_ledgers_opened=False,
        noise_seed=6201,raw_noise='unchanged original development namespace/source/SNR/6201',
        entropy_noise='t1_phy full-frame source/6201 variates; no q/m/family in seed',
        method_order=list(METHODS),auto_resume=False,holdout_used=False,
        wait_observer=dict(script=pin(Path(__file__).with_name('run_recorded_child.py')),record_dir=str(out/'parent_observer')),
        audit_file_hash_memoization='Only original raw planner file hash, verified before/after; no numerical computation cache')
    out.mkdir(parents=True);save(out/'request.json',r)
    return dict(status='REGISTERED_NOT_RUN',request=pin(out/'request.json'),frames=r['total_frames'],measured=576,packet_cap=PACKET_CAP)

def verify_request(path):
    r=phy.read(path)
    phy.require(r['schema']==SCHEMA and r['methods']==list(METHODS) and r['snrs']==list(SNRS) and r['source_indices']==list(FIXED16),'Frozen timing grid required')
    phy.require(r['warmup_per_source_snr']==3 and r['measured_per_source_snr']==3 and r['total_frames']==1152 and r['packet_cap']==2304,'Strict fixed timing budget')
    phy.verify_bindings(r['source_bindings']);phy.verify_bindings(r['input_bindings'])
    phy.verify_bindings(r['registered_native_source_additions'])
    points,_=frozen_entropy(r['frozen_policy']['path']);phy.require(points==r['entropy_points'],'Frozen entropy points changed')
    return r
def proc_identity(pid):
    path=Path('/proc')/str(pid);stat=(path/'stat').read_text();tail=stat[stat.rindex(')')+2:].split()
    uid=next(line.split()[1:] for line in (path/'status').read_text().splitlines() if line.startswith('Uid:'))
    return dict(pid=pid,parent_pid=int(tail[1]),start_ticks=int(tail[19]),uids=[int(x) for x in uid],
        argv=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x],executable=str((path/'exe').resolve()))
def admitted_wait_observer(r):
    """Allow only our immediate, pinned wait-only parent; never all ancestors."""
    bound=r['wait_observer'];script=bound['script'];parent=os.getppid();identity=proc_identity(parent)
    if script['path'] not in identity['argv']:return None
    phy.require(script['sha256']==WAITER_SHA and phy.sha(script['path'])==WAITER_SHA,'Wait observer code changed')
    phy.require(identity['uids']==[os.getuid()]*4,'Wait observer UID mismatch')
    child=proc_identity(os.getpid());prefix=[identity['argv'][0]]
    if identity['argv'][1:2]==['-B']:prefix.append('-B')
    expected=prefix+[script['path'],'--record-dir',bound['record_dir'],'--cwd',r['root'],'--']+child['argv']
    phy.require(identity['argv']==expected and identity['start_ticks']<=child['start_ticks'],'Exact wait observer argv/start identity required')
    launch=Path(bound['record_dir'])/'launch.json'
    for _ in range(20):
        if launch.exists() and (Path('/proc')/str(parent)/'wchan').read_text().strip()=='do_wait':break
        time.sleep(.05)
    record=phy.read(launch)
    phy.require(record['owner_pid']==parent and record['child_pid']==os.getpid() and record['argv']==child['argv']
        and record['cwd']==r['root'] and record['owner_script_sha256']==WAITER_SHA,'Observer launch record differs')
    boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    child_started=boot+child['start_ticks']/os.sysconf('SC_CLK_TCK')
    phy.require(abs(child_started-record['started_unix'])<2 and proc_identity(parent)==identity,'Wait observer changed/reused PID')
    phy.require((Path('/proc')/str(parent)/'wchan').read_text().strip()=='do_wait','Observer must currently wait for its child')
    phy.require(not (Path(bound['record_dir'])/'exit.json').exists(),'Observer already has an exit record')
    return dict(**identity,launch_sha256=phy.sha(launch),wchan='do_wait')
def admission(r):
    phy.require(sys.platform.startswith('linux') and str(Path(sys.executable).absolute())==r['legacy']['python'],'Original unified environment required')
    phy.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Explicit single-GPU deterministic environment')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    gpu_pids=[int(p.strip()) for p in gpu.splitlines() if p.strip().isdigit()]
    phy.require(not set(gpu_pids)-{os.getpid()},'Timing requires idle GPU; no process is stopped automatically')
    observer=admitted_wait_observer(r);allowed={os.getpid()}
    if observer is not None:allowed.add(observer['pid'])
    processes=subprocess.check_output(['ps','-eo','pid=,comm=,args='],text=True);conflicts=[]
    for line in processes.splitlines():
        bits=line.strip().split(None,2)
        if len(bits)==3 and int(bits[0]) not in allowed and bits[1].startswith('python') and r['root'] in bits[2]:conflicts.append(int(bits[0]))
    phy.require(not conflicts,'Other repository Python jobs must quiesce before exclusive timing: '+str(conflicts))
    return dict(gpu_pids=gpu_pids,conflicting_repository_python_pids=conflicts,wait_only_observer=observer,checked_unix=time.time())
def private_sionna(legacy):
    manifest=phy.read(legacy['private_sionna_manifest'])
    phy.require(manifest['status']=='ACTUAL_ORIGINAL_SIONNA_PURE_PACKAGE_PRIVATE_COPY_VERIFIED_V1' and
        manifest['private_parent']==legacy['private_sionna_parent'] and manifest['source']==legacy['sionna_source'],'Existing private Sionna copy required')
    for relative,row in manifest['entries'].items():
        phy.require(row['source']==str(Path(legacy['sionna_source'])/relative) and row['copy']==str(Path(legacy['private_sionna_parent'])/'sionna'/relative),'Private package path differs')
        phy.require(phy.sha(row['source'])==row['sha256']==phy.sha(row['copy']),'Original/private Sionna differs')
    phy.require('sionna' not in sys.modules,'One admitted Sionna package per timing process')
    sys.path.insert(0,legacy['private_sionna_parent'])
def summarize(rows):
    result=[]
    for method in METHODS:
        for snr in SNRS:
            selected=[r for r in rows if r['method']==method and r['snr_db']==snr and r['phase']=='measured']
            phy.require(len(selected)==48 and len({r['source_id'] for r in selected})==16,'Exact16 x3 measured sample coverage')
            for condition in ('all','non_gray_output','gray_fallback'):
                group=selected if condition=='all' else [r for r in selected if r['gray']==(condition=='gray_fallback')]
                for metric in ('TX_seconds','RX_seconds','software_e2e_excluding_channel_seconds','actual_frame_energy','rho'):
                    values=np.asarray([r[metric] for r in group],dtype=np.float64)
                    result.append(dict(method=method,snr_db=snr,condition=condition,metric=metric,sample_count=len(group),source_count=len({r['source_id'] for r in group}),
                        mean=None if not len(values) else float(values.mean()),median=None if not len(values) else float(np.median(values)),
                        p95=None if not len(values) else float(np.percentile(values,95)),minimum=None if not len(values) else float(values.min()),maximum=None if not len(values) else float(values.max()),
                        header_rejected_samples=sum(not r['header_accepted'] for r in group),body_crc_rejected_samples=sum(r['body_crc_accepted'] is False for r in group)))
    return result
def run(path):
    r=verify_request(path);rh=phy.sha(path);out=Path(r['out']);legacy=r['legacy'];began=time.monotonic();rows=[];budget=None
    if (out/'completion.json').exists():
        done=phy.read(out/'completion.json');phy.require(done['request_sha256']==rh,'Completed request differs');phy.verify_bindings(done['outputs'])
        return dict(status='REUSE_COMPLETE_NO_NEW_TIMING',completion=str(out/'completion.json'))
    phy.require(not (out/'attempt.json').exists() and not (out/'failure.json').exists(),'Existing attempt preserved; automatic retry disabled')
    def guard():
        phy.require(not STOP and time.time()<r['deadline_unix'] and time.monotonic()-began<r['max_seconds'],'Timing STOP or deadline reached')
        phy.require(not any(Path(p).exists() for p in r['stop_files']),'Explicit STOP file')
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    with exclusive(out/'owner.lock'),exclusive(legacy['visual_lock']):
        guard();admitted=admission(r)
        os.sched_setaffinity(0,legacy['affinity']);os.setpriority(os.PRIO_PROCESS,0,legacy['nice'])
        save(out/'attempt.json',dict(status='STARTED_NOT_COMPLETE',request_sha256=rh,admission=admitted,pid=os.getpid(),started_unix=time.time()))
        try:
            phy.verify_bindings(legacy['executable_bindings'])
            items=check_sources(r['records']);private_sionna(legacy);sys.path[:0]=legacy['pythonpath']
            # Restore the verified private package to first position.
            sys.path.insert(0,legacy['private_sionna_parent'])
            phy.require(phy.sha(legacy['helper_module'])==HELPER_SHA,'Bound helper changed')
            helper=phy.load_exact(legacy['helper_module'],'_wcl_t4_legacy_helper')
            baseline=checked(legacy['baseline_request'])
            quality=helper.load(baseline['quality_driver'],'_wcl_t4_original_quality',legacy['bindings'])
            native=quality.build_native(Path(r['root']),baseline['native_runtime'],helper.stop)
            validate_native(native,baseline,legacy['bindings'],r['registered_native_source_additions']);t=native.torch
            runtime=helper.runtime_flags(t)
            phy.require(runtime['threads']==6 and runtime['interop_threads']==2 and str(t.__version__).startswith('2.11.0') and not t.is_autocast_enabled(),'Unified FP32/B1 six-thread runtime required')
            for key in ('lpips','dino'):native.loaded[key].cpu()
            t.cuda.synchronize();t.cuda.empty_cache()
            # Keep model-loader 6/2 threads; qualification binds actual algorithm,
            # source and backend identity. New raw anchors share these settings.
            physical=phy.create_runtime(r['root'],r['phy_qualification']['path'],configure_threads=False)
            codec=SourceCodec(r['root'],r['static_completion']['path'],native)
            source=helper.load(legacy['source_driver'],'_wcl_t4_entropy_render',legacy['bindings'])
            budget=TimingReservations(out/'packet_reservations.jsonl',rh,PACKET_CAP)
            save(out/'runtime.json',dict(runtime=runtime,affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),
                source_population='original development fixed16',prior_cpu_PHY_qualification_threads=2,timing_threads=6,
                source_models=native.loaded['identity'],qualification=physical.qualification,
                registered_native_source_additions=r['registered_native_source_additions']))
            for method in METHODS:
                guard()
                endpoint=(RawEndpoint(r,native,helper,baseline,method,budget) if method.startswith('RAW64_') else
                    EntropyEndpoint(physical,codec,native,source.render_received,r['entropy_points'][method],method,budget))
                for snr in SNRS:
                    for item in items:
                        reference=None;rec=item['record'];counter=endpoint.counter_for(snr,rec['source_index'])
                        for repeat in range(WARMUPS+REPEATS):
                            guard();phase='warmup' if repeat<WARMUPS else 'measured'
                            event=f'{SCHEMA}/{method}/snr{snr}/source{rec["source_index"]}/{phase}{repeat}'
                            row=fresh_case(endpoint,item['pixels'],snr,rec['source_id'],counter,event,budget)
                            phy.require(token_sha(endpoint.last_tokens)==rec['tokens_sha256'],'Fresh original Encoder/VQ token identity changed')
                            fingerprint={k:row[k] for k in ('waveform_sha256','observation_sha256','output_sha256','receiver_status')}
                            if reference is None:reference=fingerprint
                            phy.require(reference==fingerprint,'Identical uncached repeated case changed output; diagnose without retry')
                            row.update(method=method,source_index=rec['source_index'],source_id=rec['source_id'],snr_db=snr,noise_seed=6201,
                                phase=phase,repetition=repeat if phase=='warmup' else repeat-WARMUPS,batch_size=1,
                                request_sha256=rh,source_role='development_timing',counter=counter)
                            save(out/'cases'/f'{len(rows):04d}.json',row);rows.append(row)
                        status(out/'status.json',dict(status='TIMING_RUNNING',method=method,snr_db=snr,complete_frames=len(rows),budget=budget.snapshot()))
                        print(phy.canonical(dict(method=method,snr_db=snr,source=rec['source_index'],frames=len(rows))),flush=True)
                endpoint.finish();del endpoint
            native.frozen();guard();phy.verify_bindings(r['source_bindings']);phy.verify_bindings(r['input_bindings']);phy.verify_bindings(legacy['executable_bindings'])
            validate_native(native,baseline,legacy['bindings'],r['registered_native_source_additions'])
            phy.require(len(rows)==1152 and sum(x['phase']=='measured' for x in rows)==576,'Incomplete fixed timing scope')
            snap=budget.snapshot();phy.require(snap['complete_frames']==1152 and snap['reserved_packet_slots']==2304 and not snap['unresolved_frames'],'Timing packet reservations incomplete')
            budget.close();budget=None
            flat=[{k:(phy.canonical(v) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in rows]
            csvout(out/'timing_per_call.csv',flat);csvout(out/'timing_summary.csv',summarize(rows))
            # The external observer writes its final stdout/exit after this child
            # returns. Seal immutable science records here; its actual wait
            # receipt is verified separately after termination.
            outputs={str(p):phy.sha(p) for p in sorted(out.rglob('*'))
                if p.is_file() and p.suffix in ('.json','.jsonl','.csv')
                and p.name not in ('status.json','request.json')
                and not p.is_relative_to(Path(r['wait_observer']['record_dir']))}
            save(out/'completion.json',dict(status='T4_FIXED16_ONLINE_TIMING_COMPLETE',schema=SCHEMA,request_sha256=rh,request_path=str(Path(path).resolve()),
                methods=list(METHODS),snrs=list(SNRS),source_count=16,measured_frames=576,warmup_frames=576,total_frames=1152,packet_budget=snap,
                source_bindings=r['source_bindings'],input_bindings=r['input_bindings'],outputs=outputs,runtime=runtime,
                no_online_tokens_or_output_cache=True,new_metric_calls=0,new_bootstrap_calls=0,training_updates=0,policy_selection=False,
                original_ledgers_opened=False,elapsed_seconds=time.monotonic()-began))
            status(out/'status.json',dict(status='COMPLETE',budget=snap))
            return dict(status='COMPLETE',completion=str(out/'completion.json'),packet_budget=snap)
        except BaseException as error:
            snap=None if budget is None else budget.snapshot()
            if budget is not None:budget.close();budget=None
            save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_AUTOMATIC_RETRY',error=repr(error),complete_frames=len(rows),packet_budget=snap,request_sha256=rh))
            raise
def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for n in ('root','out','a3-request','raw-cost-request','frozen-policy','phy-qualification','static-completion'):p.add_argument('--'+n,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--max-seconds',type=int,default=14400)
    p=sub.add_parser('run');p.add_argument('--request',required=True)
    p=sub.add_parser('check-admission');p.add_argument('--request',required=True)
    a=ap.parse_args();result=prepare(a) if a.command=='prepare' else run(a.request) if a.command=='run' else admission(verify_request(a.request))
    print(phy.canonical(result),flush=True)
if __name__=='__main__':main()
