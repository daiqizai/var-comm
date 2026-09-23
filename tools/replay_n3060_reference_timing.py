"""Serial real N3060 archived parity and uniform timing after all registered work."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from tools.audit_n3060_reference_phy import ROOT,BASE,HISTORY,DEEP,R3,SNRS,METHODS,read,array_sha

OUT=BASE/'n3060_native_timing_v1'
AUDIT=ROOT/'results/token_channel_efficiency_20260923/n3060_phy_audit_v1/audit.json'


def runtime_identity():
    import torch,einops
    expected=ROOT/'experiments/backbone-eval-20260912/.venv/bin/python'
    if Path(sys.executable).absolute()!=expected.absolute() or torch.__version__!='2.11.0+cu128' or einops.__version__!='0.8.1':
        raise RuntimeError('registered backbone runtime required; do not mutate active environments')
    return {'python':sys.executable,'torch':torch.__version__,'einops':einops.__version__}


def dependency_files():
    # Include self-written runtime and local upstream implementations, not data/cache trees.
    paths=subprocess.check_output(['git','ls-files','src','experiments'],cwd=ROOT,text=True).splitlines()
    result=[ROOT/p for p in paths if p.endswith(('.py','.yaml','.json')) and not p.startswith('experiments/external-baseline-positioning-20260916/vendor/')]
    result += [Path(__file__),ROOT/'tools/n3060_reference_execution.py',ROOT/'tools/audit_n3060_reference_phy.py',ROOT/'tools/run_n3060_reference_timing.sh',ROOT/'tools/rescore_token_efficiency_references.py',AUDIT,BASE/'n3060_phy_audit_v1/full_bindings.json']
    return sorted(set(result))


def release_ready(states,completions,live):
    expected=['REGISTERED_GRIDS_EXECUTED_REFERENCE_AUDIT_PUBLICATION_AND_REMOTE_ACCEPTANCE_PENDING',
              'LOW_BUDGET_MILESTONE_COMPLETE_LATER_STAGES_PENDING','REAL_REFERENCE_METRICS_COMPLETE_PENDING_REVIEW_AND_PUBLICATION',
              'REAL_N4084_REPLAY_COMPLETE_PENDING_REVIEW_AND_PUBLICATION','REAL_AUTHOR_TIMING_COMPLETE_PENDING_REVIEW_AND_PUBLICATION']
    return len(states)==5 and all(completions) and all(s.get('status')==e and not live(s) for s,e in zip(states,expected))


def archived(index,snr,seed,method,refs,deep_snrs):
    ref=refs[method,index,snr,seed]
    path=Path(ref['image_archive'])
    if str(path).startswith('/workspace/projects/VAR_COMM/'):
        path=ROOT/path.relative_to('/workspace/projects/VAR_COMM')
    with np.load(path,allow_pickle=False) as z:image=z['images'][int(ref['image_slot'])]
    if array_sha(image)!=ref['image_sha256']:raise RuntimeError('archived received pixels changed')
    if method.endswith('_adaptive'):
        family=method.split('_')[0];mode=read(HISTORY/'POLICIES_001/policies.json')['actions'][family]['quality'][str(float(snr))]
        with np.load(HISTORY/f'DEVELOPMENT_001/images/{index:04d}/transmissions.npz',allow_pickle=False) as z:tx=z[f'signal_{family}_m{mode}'].astype(np.float64)
        from var_comm.study import seeded_noise
        rx=tx+seeded_noise(ref['image_id'],seed,(3060,2))/np.sqrt(10**(snr/10))
    elif method=='perceptual_deepjscc':
        name=f'frame_{deep_snrs.index(snr)}_{seed-2001}_deep'
        with np.load(DEEP/f'images/{index:03d}/waveforms.npz',allow_pickle=False) as z:tx,rx=z[name+'_tx'],z[name+'_rx']
    else:
        name=f'r3__full_grid_prediction_features__snr{float(snr)}'
        with np.load(R3/f'images/{index:03d}/waveforms.npz',allow_pickle=False) as z:tx,rx=z[name][0],z[name+f'__seed{seed}'][0]
    return image,tx,rx


def parity(image,tx,rx,expected,method):
    errors={k:float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) for k,a,b in zip(('RGB','TX','RX'),(image,tx,rx),expected)}
    limit=0 if method.endswith('_adaptive') else 2e-5
    if not all(np.isfinite(v) for v in errors.values()) or errors['RGB']>2e-5 or errors['TX']>limit or errors['RX']>limit:
        raise ValueError('actual archived parity failed: '+str(errors))
    return errors


def run():
    import torch,yaml
    from tools.n3060_reference_execution import Systems,execute
    from latent_enhancement.runtime import configure,require_available,digest,write_json,verify_snapshot
    from token_efficiency.common import register
    from token_efficiency.evaluation_io import SafeEvaluation,load_cell,save_cell
    runtime_identity();configure();require_available();safe=SafeEvaluation();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    worker=read(OUT/'worker_registration.json');verify_snapshot(worker['bindings'])
    audit=read(AUDIT);manifest=BASE/'n3060_phy_audit_v1/full_bindings.json'
    if audit['rows']!=6000 or audit['synthetic'] is not False or digest(manifest)!=audit['full_bindings_sha256']:raise RuntimeError('N3060 actual PHY audit identity')
    bindings={str(ROOT/p) if not Path(p).is_absolute() else p:h for p,h in read(manifest).items()};verify_snapshot(bindings)
    source_root=ROOT/'outputs/EXTERNAL-BASELINE-POSITIONING-20260916/inputs_001'
    inputs=read(source_root/'development_inputs.json')
    with (source_root/'reference_per_frame.csv').open() as f:refs={(r['method'],int(r['image_index']),float(r['snr_db']),int(r['seed'])):r for r in csv.DictReader(f)}
    deep_snrs=yaml.safe_load((ROOT/'configs/learned_prefix_jscc.yaml').read_text())['evaluation']['snrs_db']
    models=Systems('cuda:0')
    # Model constructors may configure threads; restore registered timing configuration.
    torch.set_num_threads(6)
    bindings.update(worker['bindings'])
    identity={'bindings':bindings,'model_state_sha256':models.before,'selected':models.selected,
              'source_role':'original development; compatibility only, no selection','methods':list(METHODS),
              'source_indices':list(range(0,100,11)),'SNRs':list(SNRS),'noise_seed':2001,'warmups':3,'repeats':2,
              'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'TF32':False},
              'N':3060,'E':6120,'synthetic':False,'new_holdout':False,'new_training':False,'common_metrics':'existing reference worker; not duplicated'}
    register(OUT/'registration.json',identity);regsha=digest(OUT/'registration.json')
    def source(index):
        item=inputs[index]
        if digest(item['path'])!=item['file_sha256']:raise RuntimeError('source file changed')
        px=np.load(item['path'],allow_pickle=False)
        if array_sha(px)!=item['source_pixels_sha256']:raise RuntimeError('source preprocessing changed')
        return item,px
    def checked(index,snr,method):
        item,px=source(index);image,tx,rx,info=execute(models,px,int(item['class_index']),item['image_id'],snr,2001,method)
        expected=archived(index,snr,2001,method,refs,deep_snrs)
        try:errors=parity(image,tx,rx,expected,method)
        except ValueError as exc:
            write_json(OUT/f'compatibility_failure_{time.time_ns()}.json',{'method':method,'source_index':index,'snr_db':snr,'info':info,'error':str(exc),'synthetic':False});raise
        return {'method':method,'source_id':item['image_id'],'source_index':index,'preprocessing_id':item['source_pixels_sha256'],'population':'original_development','snr_db':snr,'noise_seed':2001,'context_sha256':regsha,'run_id':'n3060_native_timing_v1',**{k+'_max_abs_error':v for k,v in errors.items()},**info}
    qpath=OUT/'qualification.json';q=load_cell(qpath,regsha,'original_development:0,11','all_four_N3060')
    if q is None:
        checks=[]
        for index in (0,11):
            for method in METHODS:
                for snr in (1.,13.):safe.check();checks.append(checked(index,snr,method))
        q={'registration_sha256':regsha,'source_id':'original_development:0,11','method':'all_four_N3060','checks':checks,'synthetic':False,'status':'REAL_N3060_PARITY_PASS'};save_cell(qpath,q)
    timing=[];warmup=[];seals={}
    for index in range(0,100,11):
        item,px=source(index)
        for method in METHODS:
            safe.check();path=OUT/'cells'/f'{index:03d}_{method}.json';cell=load_cell(path,regsha,item['image_id'],method)
            if cell is None:
                warm=[];rr=[]
                for repeat in range(3):safe.check();warm.append({'repeat':repeat,**checked(index,13.,method)})
                for snr in SNRS:
                    for repeat in range(2):safe.check();rr.append({'repeat':repeat,**checked(index,snr,method)})
                cell={'registration_sha256':regsha,'source_id':item['image_id'],'method':method,'timing':rr,'warmup':warm};save_cell(path,cell)
            timing.extend(cell['timing']);warmup.extend(cell['warmup']);seals[path.name]=digest(path)
            write_json(OUT/'status.json',{'status':'REAL_N3060_UNIFORM_TIMING','cells':len(seals),'expected_cells':40,'hardware':safe.hardware})
    if len(q['checks'])!=16 or len(timing)!=400 or len(warmup)!=120 or models.state_hashes()!=models.before:raise RuntimeError('complete coverage or frozen model identity')
    verify_snapshot(bindings)
    for name,rr in [('timing.csv',timing),('warmup.csv',warmup),('qualification.csv',q['checks'])]:
        with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    write_json(OUT/'completion.json',{'status':'REAL_N3060_PARITY_AND_SHARED_ONLINE_TIMING_COMPLETE','synthetic':False,'new_holdout':False,'timed_calls':400,'warmups':120,'qualification_checks':16,'registration_sha256':regsha,'cell_sha256':seals,'files':{p.name:digest(p) for p in OUT.glob('*.csv')},'pending':'review deltas, pair all methods, publish report/figures and verify remote; no whole-study completion claim'})


def wait():
    from latent_enhancement.runtime import digest,write_json,verify_snapshot
    from token_efficiency.common import register
    from token_efficiency.delivery_chain import Runner
    from token_efficiency.coordinator import proc_identity
    from tools.rescore_token_efficiency_references import guard_is_live
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    class ReferenceRunner(Runner):
        def status(self,state,**extra):write_json(OUT/'worker_status.json',{'status':state,'pid':os.getpid(),'start_ticks':proc_identity(os.getpid())['start_ticks'],'time':time.time(),**extra})
    runner=ReferenceRunner();reg={'runtime':runtime_identity(),'bindings':{str(p.resolve()):digest(p) for p in dependency_files()},'GPU':0,'new_holdout':False,'trigger':'after all previous registered workers including author timing complete and exit'}
    register(OUT/'worker_registration.json',reg)
    try:
        while True:
            runner.check_stop();verify_snapshot(reg['bindings'])
            states=[read(BASE/p) for p in ['delivery_chain_v1/status.json','thermal_guard_v2/status.json','reference_common_metrics_v1/worker_status.json','n4084_common_replay_v1/worker_status.json','author_native_timing_v1/worker_status.json']]
            complete=[(BASE/p).is_file() for p in ['C_followups/completion.json','reference_common_metrics_v1/completion.json','n4084_common_replay_v1/completion.json','author_native_timing_v1/completion.json']]
            if release_ready(states,complete,guard_is_live):break
            if any('FAIL' in s['status'] or s['status']=='STOPPED_BY_REQUEST' for s in states):raise RuntimeError('upstream failure or requested stop requires review')
            runner.status('WAITING_FOR_AUTHOR_AND_ALL_PREDECESSORS',upstream=[s['status'] for s in states]);time.sleep(30)
        runner.run('n3060_native_timing_v1',['tools.replay_n3060_reference_timing','--run'],OUT/'completion.json')
        runner.status('REAL_N3060_TIMING_COMPLETE_PENDING_REVIEW_AND_PUBLICATION')
    except Exception as exc:runner.status('FAILED_N3060_TIMING_WORKER',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run',action='store_true');g.add_argument('--wait',action='store_true');a=p.parse_args()
    from latent_enhancement.runtime import ResourceBusy
    try:run() if a.run else wait()
    except ResourceBusy as exc:print('SAFE_N3060_PAUSE',str(exc),flush=True);raise SystemExit(75)
if __name__=='__main__':main()
