"""Deferred actual Swin/ADJSCC parity and uniform CPU endpoint timing.

Uses the pinned author interpreter, not the VAR runtime. All GPU work waits for
registered C, archived metric scoring and N4084 replay to complete and exit.
"""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from tools.author_reference_execution import METHODS, SNRS, array_sha, execute, frame_counter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/external-baseline-positioning-20260916'
BASE = ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
OLD = ROOT/'outputs/EXTERNAL-BASELINE-POSITIONING-20260916'
OUT = BASE/'author_native_timing_v1'
RESULTS = ROOT/'results/token_channel_efficiency_20260923'
AUDIT = RESULTS/'author_phy_audit_v1/audit.json'
PYTHON = EXP/'.venv/bin/python'

def read(path):
    return json.loads(Path(path).read_text())

def code_files():
    files = [Path(__file__), Path(__file__).with_name('author_reference_execution.py'),
             Path(__file__).with_name('run_author_reference_timing.sh'), AUDIT,
             ROOT/'src/var_comm/study.py', EXP/'configs/protocol.json',
             BASE/'author_phy_audit_v1/full_bindings.json']
    for relative in ('src/external_positioning','vendor/SwinJSCC','vendor/diffcom_code'):
        files.extend((EXP/relative).rglob('*.py'))
    files.extend((ROOT/'src/var_comm').rglob('*.py'))
    old=ROOT/'experiments/var-latent-enhancement-20260917'
    for folder in [ROOT/'experiments/var-short-prefix-hybrid-20260923/src', old/'src', old/'phase_b/src']:
        files.extend(folder.rglob('*.py'))
    files.extend(EXP/'vendor/diffcom_code'/p for p in ('utils/util.py','guided_diffusion/measurement.py','configs/diffcom.yaml'))
    for module in ('token_efficiency.evaluation_io','token_efficiency.common','token_efficiency.thermal_guard','token_efficiency.coordinator','latent_enhancement.runtime'):
        import importlib
        files.append(Path(importlib.import_module(module).__file__))
    return sorted(set(files))

def live(status):
    from token_efficiency.coordinator import proc_identity
    if not status.get('pid') or not status.get('start_ticks'):
        return False
    current = proc_identity(status['pid'])
    return bool(current and current['start_ticks']==status['start_ticks'] and current['state']!='Z')

def release_ready(states, completions, is_live=live):
    expected = ('REGISTERED_GRIDS_EXECUTED_REFERENCE_AUDIT_PUBLICATION_AND_REMOTE_ACCEPTANCE_PENDING',
                'LOW_BUDGET_MILESTONE_COMPLETE_LATER_STAGES_PENDING',
                'REAL_REFERENCE_METRICS_COMPLETE_PENDING_REVIEW_AND_PUBLICATION',
                'REAL_N4084_REPLAY_COMPLETE_PENDING_REVIEW_AND_PUBLICATION')
    return len(states)==4 and all(completions) and all(s.get('status')==e and not is_live(s) for s,e in zip(states,expected))

def upstream():
    files = ['delivery_chain_v1/status.json','thermal_guard_v2/status.json',
             'reference_common_metrics_v1/worker_status.json','n4084_common_replay_v1/worker_status.json']
    return [read(BASE/p) for p in files]

def runtime():
    import torch
    if torch.__version__!='1.12.1+cu116' or Path(sys.executable).absolute()!=PYTHON.absolute():
        raise RuntimeError('use the pinned author interpreter and torch1.12.1+cu116')
    return {'python':sys.version,'interpreter':str(PYTHON),'torch':torch.__version__,
            'cuda_build':torch.version.cuda,'threads':6,'interop_threads':2,
            'precision':'FP32 TF32 disabled','warmups_per_source_method':3,'repeats':2,
            'runtime_caveat':'author torch1.12.1/cu116 differs from current VAR torch; report separately, no runtime-controlled speedup claim'}

def pixels(item):
    from latent_enhancement.runtime import digest
    path = Path(item['path'])
    if digest(path)!=item['file_sha256']:
        raise RuntimeError('source file identity')
    value = np.load(path, allow_pickle=False)
    if array_sha(value)!=item['source_pixels_sha256']:
        raise RuntimeError('source pixel identity')
    return value

def frozen_sha(model):
    h = hashlib.sha256()
    net = model.model if hasattr(model,'model') else model.operator.model
    if net.training or any(p.requires_grad for p in net.parameters()):
        raise RuntimeError('author parameters must remain frozen in eval mode')
    for key,value in sorted(net.state_dict().items()):
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run():
    import torch
    from external_positioning.author_models import SwinAuthor, DiffComAuthor
    from latent_enhancement.runtime import digest, write_json, verify_snapshot, require_available, configure
    from token_efficiency.common import register
    from token_efficiency.evaluation_io import SafeEvaluation, load_cell, save_cell
    identity_runtime = runtime();configure();require_available();safe = SafeEvaluation()
    OUT.mkdir(parents=True,exist_ok=True)
    lock = (OUT/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    worker = read(OUT/'worker_registration.json');verify_snapshot(worker['bindings'])
    audit = read(AUDIT);manifest = BASE/'author_phy_audit_v1/full_bindings.json'
    if digest(manifest)!=audit['full_bindings_sha256'] or audit['synthetic'] is not False or audit['rows']!=9000:
        raise RuntimeError('full real PHY audit identity')
    original = {str(ROOT/p) if not Path(p).is_absolute() else p:h for p,h in read(manifest).items()}
    verify_snapshot(original)
    calibration_path = OLD/'inputs_001/calibration_inputs.json'
    development_path = OLD/'inputs_001/development_inputs.json'
    cal = read(calibration_path);dev = read(development_path)
    if len(cal)!=8 or len(dev)!=100 or [x['image_index'] for x in dev]!=list(range(100)):
        raise RuntimeError('registered original populations only')
    bound = dict(worker['bindings']);bound[str(calibration_path)] = digest(calibration_path)
    bound[str(development_path)] = digest(development_path)
    protocol = read(EXP/'configs/protocol.json')
    for name,expected in protocol['checkpoints'].items():
        if name.startswith(('Swin','ADJSCC')):
            path = EXP/'checkpoints'/name
            if digest(path)!=expected:raise RuntimeError('author checkpoint identity')
            bound[str(path)] = expected
    registration = {'bindings':bound,'runtime':identity_runtime,'methods':METHODS,
                    'sources':{x['image_id']:x['source_pixels_sha256'] for x in dev},
                    'calibration_indices':[cal[0]['image_index'],cal[-1]['image_index']],
                    'native_parity_SNRs':[13.,19.], 'native_parity_noise':4101,
                    'timing_indices':list(range(0,100,11)),'SNRs':SNRS,'noise':2001,
                    'new_holdout':False,'retraining':False,'synthetic':False,
                    'protocol':'unchanged common_paid_information; CRC-rejected legal metadata retained',
                    'qualification_tolerance':2e-5,'archived_pixels_tolerance':2e-5}
    # JSON tuples become arrays; use canonical representation for resume equality.
    registration = json.loads(json.dumps(registration))
    register(OUT/'registration.json',registration);regsha=digest(OUT/'registration.json')
    all_timing=[];all_warm=[];all_qual=[];seals={};model_states={}
    for family,rate in METHODS:
        safe.check();method=f'swin_ra{rate}' if family=='swin' else f'adjscc_c{rate}'
        cp=EXP/'checkpoints'/('SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model' if family=='swin' else f'ADJSCC_C={rate}.pth.tar')
        model=SwinAuthor(cp) if family=='swin' else DiffComAuthor(cp,rate)
        state=frozen_sha(model);model_states[method]=state
        path=OUT/'qualification'/f'{method}.json'
        qual=load_cell(path,regsha,'original_calibration:0,999',method)
        if qual is None:
            checks=[]
            for item in (cal[0],cal[-1]):
                source=pixels(item)
                for snr in (13.,19.):
                    safe.check();counter=frame_counter(item['image_index'],snr,4101,True)
                    image,info,noise=execute(model,family,rate,source,item['image_id'],snr,4101,counter)
                    # Native reference is a separate offline diagnostic. Formal RX above
                    # used only decoded paid metadata, never the transmitter context.
                    if family=='swin' and not info['metadata_exact']:
                        write_json(OUT/f'qualification_failure_{time.time_ns()}.json',{'method':method,'info':info,'reason':'native comparison requires actually decoded exact metadata','synthetic':False})
                        raise RuntimeError('native parity metadata not exact; inspect evidence')
                    native=model.native_reference(source,snr,rate if family=='swin' else counter,noise)
                    error=float(np.max(np.abs(image-native)))
                    check={'method':method,'source_id':item['image_id'],'source_index':item['image_index'],'snr_db':snr,'noise_seed':4101,'native_max_abs_error':error,**info}
                    checks.append(check)
                    if error>2e-5 or not np.isfinite(error):
                        write_json(OUT/f'qualification_failure_{time.time_ns()}.json',{'checks':checks,'synthetic':False})
                        raise RuntimeError('real author native parity failed')
            if frozen_sha(model)!=state:raise RuntimeError('frozen author state changed in qualification')
            qual={'registration_sha256':regsha,'source_id':'original_calibration:0,999','method':method,'checks':checks,'status':'REAL_NATIVE_PARITY_PASS','synthetic':False}
            save_cell(path,qual)
        all_qual.extend(qual['checks'])
        for index in range(0,100,11):
            safe.check();item=dev[index];path=OUT/'cells'/f'{method}_{index:03d}.json'
            cell=load_cell(path,regsha,item['image_id'],method)
            if cell is None:
                source=pixels(item);timing=[];warm=[]
                for repeat in range(3):
                    safe.check();_,info,_=execute(model,family,rate,source,item['image_id'],13.,2001,frame_counter(index,13.,2001))
                    warm.append({'method':method,'source_index':index,'repeat':repeat,**info})
                for snr in SNRS:
                    directory=OLD/f'development_{family}_001/frames'/f'rate{rate}_seed2001_snr{snr:g}_source{index:04d}'
                    archive=directory/'reconstructions.npz';frame=read(directory/'frame.json')
                    previous=next(r for r in frame['rows'] if r['protocol']=='common_paid_information')
                    if digest(archive)!=frame['archive_sha256']:raise RuntimeError('archived receive file changed')
                    with np.load(archive,allow_pickle=False) as z:expected=z['images'][int(previous['image_slot'])]
                    if array_sha(expected)!=previous['image_sha256']:raise RuntimeError('archived receive pixels changed')
                    for repeat in range(2):
                        safe.check();image,info,_=execute(model,family,rate,source,item['image_id'],snr,2001,frame_counter(index,snr,2001))
                        error=float(np.max(np.abs(image-expected)))
                        hashes={k:info[k]==previous[k] for k in ('data_transmitted_sha256','data_observed_sha256','standard_data_noise_sha256')}
                        row={'method':method,'source_id':item['image_id'],'source_index':index,'preprocessing_id':item['source_pixels_sha256'],'population':'original_development','snr_db':snr,'noise_seed':2001,'repeat':repeat,'context_sha256':regsha,'run_id':'author_native_timing_v1','archived_RGB_max_abs_error':error,'exact_waveform_noise_identity':all(hashes.values()),**info}
                        if error>2e-5 or not np.isfinite(error) or not all(hashes.values()) or info['metadata_usable']!=previous['metadata_usable'] or info['metadata_crc_accepted']!=previous['metadata_crc_accepted'] or info['metadata_exact']!=previous['offline_metadata_exact'] or info['fallback']!=previous['fallback']:
                            write_json(OUT/f'compatibility_failure_{time.time_ns()}.json',{'row':row,'hash_checks':hashes,'synthetic':False})
                            raise RuntimeError('real archived waveform/receiver parity failed; no bulk acceptance')
                        timing.append(row)
                cell={'registration_sha256':regsha,'source_id':item['image_id'],'method':method,'timing':timing,'warmup':warm}
                save_cell(path,cell)
            all_timing.extend(cell['timing']);all_warm.extend(cell['warmup']);seals[path.name]=digest(path)
            write_json(OUT/'status.json',{'status':'REAL_AUTHOR_PARITY_AND_TIMING','completed_cells':len(seals),'expected_cells':60,'hardware':safe.hardware})
        if frozen_sha(model)!=state:raise RuntimeError('frozen author parameters/buffers changed')
        del model;gc.collect();torch.cuda.empty_cache()
    if len(all_qual)!=24 or len(all_timing)!=600 or len(all_warm)!=180:
        raise RuntimeError('complete registered author coverage required')
    verify_snapshot(bound)
    for name,data in [('timing.csv',all_timing),('warmup.csv',all_warm),('native_parity.csv',all_qual)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write_json(OUT/'completion.json',{'status':'REAL_AUTHOR_NATIVE_PARITY_AND_UNIFORM_TIMING_COMPLETE','synthetic':False,'new_holdout':False,'native_checks':24,'timed_calls':600,'warmups':180,'registration_sha256':regsha,'runtime':identity_runtime,'model_state_sha256':model_states,'cell_sha256':seals,'files':{p.name:digest(p) for p in OUT.glob('*.csv')},'pending':'review, natural-resource labels, paired all-method analysis, publication and remote checkout; not entire study completion'})

def wait():
    from latent_enhancement.runtime import digest, write_json, verify_snapshot, foreign_gpu_processes
    from token_efficiency.common import register
    from token_efficiency.coordinator import proc_identity
    from token_efficiency.thermal_guard import sample, Window
    runtime();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=[False]
    def halt(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def status(state,**extra):
        write_json(OUT/'worker_status.json',{'status':state,'pid':os.getpid(),'start_ticks':proc_identity(os.getpid())['start_ticks'],'time':time.time(),**extra})
    def check_stop():
        if stopping[0]:status('STOPPED_BY_REQUEST');raise SystemExit(75)
    registration={'bindings':{str(p.resolve()):digest(p) for p in code_files()},'runtime':runtime(),'trigger':'after C, main chain, reference metrics and N4084 replay complete AND exit; GPU0 free and cool','GPU':0,'new_holdout':False}
    register(OUT/'worker_registration.json',registration)
    try:
        while True:
            check_stop();verify_snapshot(registration['bindings']);states=upstream()
            completions=[(BASE/p).is_file() for p in ('C_followups/completion.json','reference_common_metrics_v1/completion.json','n4084_common_replay_v1/completion.json')]
            if release_ready(states,completions):break
            if any('FAIL' in s['status'] or s['status']=='STOPPED_BY_REQUEST' for s in states):
                raise RuntimeError('upstream stopped/failed; preserve state for repair')
            status('WAITING_FOR_N4084_AND_ALL_PREDECESSORS',upstream=[s['status'] for s in states]);time.sleep(30)
        if (OUT/'accepted_stage.json').exists():
            accepted=read(OUT/'accepted_stage.json')
            if digest(OUT/'completion.json')!=accepted['completion_sha256']:raise RuntimeError('accepted completion changed')
            status('REAL_AUTHOR_TIMING_COMPLETE_PENDING_REVIEW_AND_PUBLICATION');return
        while True:
            check_stop();verify_snapshot(registration['bindings']);window=Window()
            while True:
                check_stop();hardware=sample();_,ready=window.add(hardware);busy=foreign_gpu_processes()
                status('WAITING_FOR_COOL_FREE_GPU0',hardware=hardware,foreign_gpu_work=busy,cool_samples=window.cool_count)
                if ready and not busy:break
                time.sleep(10)
            attempt=OUT/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
            command=[str(PYTHON),'-u','-m','tools.replay_author_reference_timing','--run']
            with (attempt/'console.log').open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            expected=proc_identity(child.pid)
            if not expected:raise RuntimeError('author execution exited before identity capture')
            write_json(attempt/'launch.json',{'command':command,'process':expected,'session_id':os.getsid(child.pid)})
            window=Window();reason=None
            while child.poll() is None:
                try:hardware=sample();hot,_=window.add(hardware)
                except Exception as exc:hardware={'sensor_error':repr(exc)};hot=True;reason='sensor_failure'
                status('RUNNING_AUTHOR_STAGE',process=expected,hardware=hardware,attempt=str(attempt))
                if stopping[0] or hot:
                    reason=reason or ('requested_stop' if stopping[0] else 'thermal')
                    actual=proc_identity(child.pid)
                    if actual and actual['state']!='Z':
                        if actual['start_ticks']!=expected['start_ticks'] or actual['cmdline']!=expected['cmdline']:
                            raise RuntimeError('author child PID changed; signal refused')
                        os.kill(child.pid,signal.SIGTERM)
                    write_json(attempt/'stop_request.json',{'reason':reason,'process':expected,'hardware':hardware});break
                time.sleep(10)
            while child.poll() is None:
                status('WAITING_FOR_SAFE_AUTHOR_PAUSE',process=expected,reason=reason);time.sleep(2)
            write_json(attempt/'exit.json',{'returncode':child.returncode,'reason':reason})
            check_stop()
            if reason=='sensor_failure':raise RuntimeError('sensor failure; no automatic retry')
            if child.returncode==75:
                status('SAFE_AUTHOR_PAUSE_COOLDOWN');continue
            if child.returncode!=0:raise RuntimeError(f'author stage failed rc={child.returncode}: {attempt}')
            done=read(OUT/'completion.json')
            if done['status']!='REAL_AUTHOR_NATIVE_PARITY_AND_UNIFORM_TIMING_COMPLETE' or done['synthetic'] is not False or done['timed_calls']!=600 or done['native_checks']!=24:
                raise RuntimeError('actual author completion gate')
            verify_snapshot(registration['bindings'])
            write_json(OUT/'accepted_stage.json',{'completion_sha256':digest(OUT/'completion.json'),'attempt':str(attempt),'time':time.time()})
            status('REAL_AUTHOR_TIMING_COMPLETE_PENDING_REVIEW_AND_PUBLICATION');return
    except Exception as exc:
        status('FAILED_AUTHOR_PARITY_TIMING_WORKER',error=repr(exc));raise

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',action='store_true');group.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    from latent_enhancement.runtime import ResourceBusy
    try:run() if args.run else wait()
    except ResourceBusy as exc:print('SAFE_AUTHOR_PAUSE',str(exc),flush=True);raise SystemExit(75)

if __name__=='__main__':main()
