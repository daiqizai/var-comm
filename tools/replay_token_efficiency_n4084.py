"""Deferred full real N4084 replay, preserving old policies, weights and evidence."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from tools.legacy_reference_execution import METHODS,execute
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
OUT=BASE/'n4084_common_replay_v1'
AUDIT=ROOT/'results/token_channel_efficiency_20260923/n4084_reference_audit_v1/audit.json'
LEGACY=ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/followup'

def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def dependency_files():
    experiment=ROOT/'experiments/var-latent-enhancement-20260917'
    directories=[ROOT/'src/var_comm',ROOT/'experiments/token_channel_efficiency_20260923/src',ROOT/'experiments/var-short-prefix-hybrid-20260923/src',experiment/'src',*[experiment/part/'src' for part in ('phase_b','evaluation','followup','mechanisms','research')]]
    return sorted({p.resolve() for directory in directories for p in directory.rglob('*.py')} | {ROOT/'tools/run_legacy_n4084_replay.sh'})

def release_ready(chain,guard,reference,reference_complete,c_complete,live):
    return bool(c_complete and reference_complete and chain.get('status')=='REGISTERED_GRIDS_EXECUTED_REFERENCE_AUDIT_PUBLICATION_AND_REMOTE_ACCEPTANCE_PENDING' and reference.get('status')=='REAL_REFERENCE_METRICS_COMPLETE_PENDING_REVIEW_AND_PUBLICATION' and not any(live(s) for s in (chain,guard,reference)))

def run():
    import torch,yaml
    from latent_enhancement.runtime import require_available,digest,write_json,verify_snapshot,model_paths,settings
    from latent_enhancement_b.common import load_decoder,scale_statistics
    from latent_followup.timing_clean import load_arm
    from latent_followup.run_identity import checked_checkpoint
    from latent_research.train import make_models
    from latent_enhancement_eval.runner import load_targets,write_rows
    from var_comm.next_scale_prior import load_models,state_sha256
    from var_comm.quality import load_quality_models,quality_metrics
    from token_efficiency.common import configure_runtime,register
    from token_efficiency.digital_grid import population
    from token_efficiency.evaluation_io import SafeEvaluation,load_cell,save_cell
    configure_runtime();require_available();safe=SafeEvaluation();OUT.mkdir(parents=True,exist_ok=True)
    audit=read(AUDIT)
    if audit['status']!='REAL_N4084_PROVENANCE_AND_FROZEN_POLICY_AUDIT_PASS_COMPATIBILITY_PENDING':raise RuntimeError('real provenance audit gate')
    audit_bindings={str(ROOT/p) if not Path(p).is_absolute() else p:h for p,h in audit['bindings'].items()};verify_snapshot(audit_bindings)
    device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device)
    parent_rec=read(ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v1/training/selected_enhancement1024.json')
    parent=torch.load(checked_checkpoint(parent_rec,ROOT),map_location='cpu',weights_only=True)
    models=make_models(scale,settings()['stage_B'],parent,device);del parent
    training=LEGACY/'research_20260923_training_v3';completed=read(training/'completion.json')
    selected={}
    for method in ('full_tx_control','light_tx'):
        rec=read(training/f'selected_{method}.json');cp=checked_checkpoint(rec,ROOT);payload=torch.load(cp,map_location='cpu',weights_only=True)
        if rec!=completed['selected'][method] or payload['registration_sha256']!=completed['registration_sha256'] or rec['arm_key']!=method:raise RuntimeError('legacy selected identity')
        prefix=method+'.';models[method].load_state_dict({k[len(prefix):]:v for k,v in payload['models'].items() if k.startswith(prefix)},strict=True);models[method].eval().requires_grad_(False);selected[method]=rec
    del models['pure_continuous']
    arms={name:load_arm(name,(32,16,16),scale,device) for name in ('enhancement512','enhancement1024')}
    policy_dir=LEGACY/'research_20260923_system_policy_v2';policy=read(policy_dir/'frozen_policy.json');digital=read(policy_dir/'digital_adaptive_policies.json')
    if state_sha256(decoder)!=policy['decoder_state_sha256']:raise RuntimeError('Dc identity')
    targets=load_targets();current,_=population('development')
    if len(targets)!=100 or len(current)!=100:raise RuntimeError('original development size')
    for t,c in zip(targets,current):
        if t['target']['image_id']!=c['image_id'] or not np.array_equal(t['pixels'],c['pixels']):raise RuntimeError('source preprocessing')
    cfg=ROOT/'configs/progressive_channel.yaml';lp,dino,linear=load_quality_models(yaml.safe_load(cfg.read_text())['quality'],device)
    states={'decoder':state_sha256(decoder),'LPIPS':state_sha256(lp),'DINO':state_sha256(dino)}
    code_files=[Path(__file__),Path(__file__).with_name('legacy_reference_execution.py'),cfg,Path(linear),AUDIT,*dependency_files()]
    for module in ['latent_research.evaluate','latent_followup.timing_clean','latent_enhancement_eval.deployment','latent_enhancement_eval.runner','token_efficiency.evaluation_io','var_comm.quality']:
        import importlib
        code_files.append(Path(importlib.import_module(module).__file__))
    bound={str(p.resolve()):digest(p) for p in code_files};bound.update(audit_bindings)
    identity={'bindings':bound,'selected':selected,'state_sha256':states,'sources':{r['image_id']:r['preprocessing_id'] for r in current},'methods':list(METHODS),'precision':'FP32 deterministic TF32 off','N':4084,'E':8168,'selection':'unchanged historical calibration-frozen policies and selected checkpoints','source_role':'original100 development; no new holdout','retraining':False,'full_online_TX':True,'full_replay_reason':'complete current metric and online timing identity; no historic RX pixel archives for this scope','timing':{'source_indices':list(range(0,100,11)),'warmups':3,'repeats':2,'noise_seed':2001},'synthetic':False}
    register(OUT/'registration.json',identity);regsha=digest(OUT/'registration.json')
    old={(r['method'],r['image_id'],float(r['snr_db']),int(r['seed'])):r for r in rows(ROOT/'results/review_20260923_phase2/comparison/per_frame.csv')}
    # Real-weight bounded acceptance precedes the full replay; never infer PASS from CPU tests.
    qualification_path=OUT/'qualification.json'
    qualification=load_cell(qualification_path,regsha,'development:0,11','legacy_acceptance')
    if qualification is None:
        checks=[]
        for index in (0,11):
            record=targets[index]
            for method in METHODS:
                for snr in (1.,13.):
                    safe.check();image,info=execute(record,method,snr,2001,vae,var,decoder,models,arms,device,policy,digital)
                    metrics,*_=quality_metrics(record['pixels'].astype(np.float32)/255,[image],lp,dino,device)
                    previous=old[(method,record['target']['image_id'],snr,2001)]
                    errors={m:abs(metrics[0][m]-float(previous[m])) for m in ('psnr_db','lpips_alex','dino_cosine')}
                    entry={'source_index':index,'method':method,'snr_db':snr,**errors}
                    if info['endpoint']=='m8_plus_latent_512_fold_N4084':
                        from latent_followup.timing_clean import source_prepare
                        from latent_enhancement.latent import complete_latent
                        from latent_enhancement_eval.runner import raw_transmit_budget
                        from latent_research.system_policy import folded_receive
                        from latent_enhancement_b.model import render_received
                        with torch.no_grad():
                            tensor=torch.from_numpy(record['pixels'][None].astype(np.float32)/127.5-1).to(device)
                            f,tokens=source_prepare(vae,tensor);label=int(record['target']['class_index'])
                            tx=complete_latent(vae,var,tokens[:8],label,device);base,_=raw_transmit_budget(tokens,label,8,3572)
                            wave=arms['enhancement512'].encoder(f-tx,tx)
                            z,status=folded_receive(base,wave,tx,record['target']['image_id'],snr,2001,vae,var,arms['enhancement512'],device,{})
                            expected=render_received(decoder,z,status)[0].cpu().numpy()
                        entry['folded_original_engine_max_abs_error']=float(np.max(np.abs(image-expected)))
                    checks.append(entry)
                    if errors['psnr_db']>1e-4 or max(errors['lpips_alex'],errors['dino_cosine'])>1e-5 or entry.get('folded_original_engine_max_abs_error',0)>2e-5:
                        write_json(OUT/f'qualification_failure_{time.time_ns()}.json',{'status':'REAL_LEGACY_COMPATIBILITY_REQUIRES_REVIEW','checks':checks,'synthetic':False})
                        raise RuntimeError('real legacy compatibility mismatch; evidence retained, no bulk replay')
        qualification={'registration_sha256':regsha,'source_id':'development:0,11','method':'legacy_acceptance','status':'REAL_LEGACY_METRIC_AND_FOLDED_ENGINE_ACCEPTANCE_PASS','checks':checks,'synthetic':False}
        save_cell(qualification_path,qualification)
    all_rows=[];all_timing=[];all_warm=[];seals={}
    for index,(record,source) in enumerate(zip(targets,current)):
        sid=source['image_id']
        for method in METHODS:
            safe.check();path=OUT/'cells'/f'{index:03d}_{method}.json';saved=load_cell(path,regsha,sid,method)
            if saved is None:
                images=[];frame=[]
                for snr in (1.,4.,7.,13.,19.):
                    for seed in (2001,2002,2003):
                        safe.check();image,info=execute(record,method,snr,seed,vae,var,decoder,models,arms,device,policy,digital)
                        images.append(image);frame.append({'method':method,'source_id':sid,'source_index':index,'population':'development','preprocessing_id':source['preprocessing_id'],'snr_db':snr,'noise_seed':seed,'N':4084,'E':info['E'],'endpoint':info['endpoint'],'context_sha256':regsha,'run_id':'n4084_common_replay_v1','noise_scope':'historical exact method namespace/segment lengths; equal seed is not identical cross-method observation'})
                safe.check();metrics,*_=quality_metrics(record['pixels'].astype(np.float32)/255,images,lp,dino,device)
                for row,image,metric in zip(frame,images,metrics):
                    previous=old[(method,sid,row['snr_db'],row['noise_seed'])]
                    row.update(metric);row['mse']=float(np.square(image-record['pixels'].astype(np.float32)/255,dtype=np.float64).mean())
                    for m in ('psnr_db','lpips_alex','dino_cosine'):row['old_'+m]=float(previous[m]);row['delta_'+m]=row[m]-float(previous[m])
                    if not all(np.isfinite(row[m]) for m in ('mse','psnr_db','lpips_alex','dino_cosine')):raise RuntimeError('nonfinite real quality')
                timing=[];warm=[]
                if index in range(0,100,11):
                    for repeat in range(3):
                        safe.check();_,info=execute(record,method,13.,2001,vae,var,decoder,models,arms,device,policy,digital)
                        warm.append({'method':method,'source_index':index,'repeat':repeat,**{k:info[k] for k in ('tx_ms','rx_ms','total_ms')}})
                    for si,snr in enumerate((1.,4.,7.,13.,19.)):
                        for repeat in range(2):
                            safe.check();image,info=execute(record,method,snr,2001,vae,var,decoder,models,arms,device,policy,digital)
                            error=float(np.max(np.abs(image-images[si*3])))
                            if error>2e-5:raise RuntimeError('real shared quality/timing output mismatch')
                            timing.append({'method':method,'source_id':sid,'source_index':index,'snr_db':snr,'noise_seed':2001,'repeat':repeat,'quality_max_abs_error':error,'N':4084,'E':info['E'],'context_sha256':regsha,**{k:info[k] for k in ('tx_ms','rx_ms','total_ms','timing_endpoints','endpoint')}})
                saved={'registration_sha256':regsha,'source_id':sid,'method':method,'rows':frame,'timing':timing,'warmup':warm};save_cell(path,saved)
            all_rows.extend(saved['rows']);all_timing.extend(saved['timing']);all_warm.extend(saved['warmup']);seals[path.name]=digest(path)
            write_json(OUT/'status.json',{'status':'REAL_LEGACY_N4084_REPLAY','completed_cells':len(seals),'expected_cells':600,'hardware':safe.hardware})
    expected={(m,r['image_id'],s,n) for m in METHODS for r in current for s in (1.,4.,7.,13.,19.) for n in (2001,2002,2003)}
    if len(all_rows)!=9000 or {(r['method'],r['source_id'],r['snr_db'],r['noise_seed']) for r in all_rows}!=expected or len(all_timing)!=600:raise RuntimeError('complete real replay coverage')
    verify_snapshot(bound)
    if states!={'decoder':state_sha256(decoder),'LPIPS':state_sha256(lp),'DINO':state_sha256(dino)}:raise RuntimeError('frozen quality states changed')
    for name,data in [('per_frame.csv',all_rows),('timing.csv',all_timing),('warmup.csv',all_warm)]:write_rows(OUT/name,data)
    impact={m:{k:max(abs(r['delta_'+k]) for r in all_rows if r['method']==m) for k in ('psnr_db','lpips_alex','dino_cosine')} for m in METHODS}
    write_json(OUT/'historical_metric_impact.json',impact)
    write_json(OUT/'completion.json',{'status':'REAL_N4084_CURRENT_METRICS_AND_SHARED_TIMING_COMPLETE_REVIEW_PENDING','rows':len(all_rows),'timed_calls':len(all_timing),'registration_sha256':regsha,'files':{p.name:digest(p) for p in OUT.glob('*.csv')},'cell_sha256':seals,'synthetic':False,'new_holdout':False,'pending':'review deltas, pair with new B/C results, external PHY/timing audit, publication and remote acceptance'})

def wait():
    from latent_enhancement.runtime import digest,write_json,verify_snapshot
    from token_efficiency.common import register
    from token_efficiency.delivery_chain import Runner
    from token_efficiency.coordinator import proc_identity
    from tools.rescore_token_efficiency_references import guard_is_live
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    class LegacyRunner(Runner):
        def status(self,state,**extra):write_json(OUT/'worker_status.json',{'status':state,'pid':os.getpid(),'start_ticks':proc_identity(os.getpid())['start_ticks'],'time':time.time(),**extra})
    runner=LegacyRunner();files=[Path(__file__),Path(__file__).with_name('legacy_reference_execution.py'),AUDIT,*dependency_files()]
    reg={'bindings':{str(p.resolve()):digest(p) for p in files},'trigger':'after actual C, main-chain exit AND archived-reference metric worker exit; GPU0 cool/free','GPU':0,'new_holdout':False}
    register(OUT/'worker_registration.json',reg)
    try:
        while True:
            runner.check_stop();verify_snapshot(reg['bindings']);chain=read(BASE/'delivery_chain_v1/status.json');guard=read(BASE/'thermal_guard_v2/status.json');reference=read(BASE/'reference_common_metrics_v1/worker_status.json')
            if release_ready(chain,guard,reference,(BASE/'reference_common_metrics_v1/completion.json').exists(),(BASE/'C_followups/completion.json').exists(),guard_is_live):break
            if 'FAIL' in chain['status'] or 'FAIL' in reference['status']:raise RuntimeError('upstream failure requires repair')
            runner.status('WAITING_FOR_MAIN_AND_REFERENCE_METRIC_RELEASE',delivery_state=chain['status'],reference_state=reference['status']);time.sleep(30)
        runner.run('n4084_common_replay_v1',['tools.replay_token_efficiency_n4084','--run'],OUT/'completion.json')
        runner.status('REAL_N4084_REPLAY_COMPLETE_PENDING_REVIEW_AND_PUBLICATION')
    except Exception as exc:
        runner.status('FAILED_N4084_REPLAY_WORKER',error=repr(exc));raise

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--run',action='store_true');group.add_argument('--wait',action='store_true');args=parser.parse_args()
    from latent_enhancement.runtime import ResourceBusy
    try:run() if args.run else wait()
    except ResourceBusy as exc:print('SAFE_LEGACY_REPLAY_PAUSE',str(exc),flush=True);raise SystemExit(75)
if __name__=='__main__':main()
