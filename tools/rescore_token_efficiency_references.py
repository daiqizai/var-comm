#!/usr/bin/env python3
"""Common real-GPU metric scoring of audited historical RX outputs, after C frees GPU0."""
import argparse,fcntl,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
AUDIT=BASE/'reference_audit_v1'
OUT=BASE/'reference_common_metrics_v1'

def read(p):return json.loads(Path(p).read_text())

def guard_is_live(status):
    from token_efficiency.coordinator import proc_identity
    if not status.get('pid') or not status.get('start_ticks'):return False
    p=proc_identity(status['pid'])
    return bool(p and p['start_ticks']==status['start_ticks'] and p['state']!='Z')

def run_metrics():
    import torch,yaml
    from latent_enhancement.runtime import require_available,digest,write_json,verify_snapshot,ResourceBusy
    from latent_enhancement_eval.runner import write_rows
    from token_efficiency.common import configure_runtime,register
    from token_efficiency.digital_grid import population
    from token_efficiency.evaluation_io import SafeEvaluation,load_cell,save_cell
    from var_comm.quality import load_quality_models,quality_metrics
    from var_comm.next_scale_prior import state_sha256
    from tools.audit_token_efficiency_references import resolve_archive,array_sha,validate_rows
    configure_runtime();require_available();safe=SafeEvaluation();done=read(AUDIT/'completion.json')
    if done['status']!='REAL_HISTORICAL_SOURCE_RESOURCE_PIXEL_AUDIT_PASS_COMMON_METRICS_PENDING' or done['rescore_inputs_sha256']!=digest(AUDIT/'rescore_inputs.json'):raise RuntimeError('full real reference audit gate')
    inputs=read(AUDIT/'rescore_inputs.json');verify_snapshot(inputs['bindings']);records,_=population('development');sources={r['image_id']:r['preprocessing_id'] for r in records}
    if sources!=inputs['sources']:raise RuntimeError('current source preprocessing changed')
    validate_rows(inputs['rows'],sources)
    cfgpath=ROOT/'configs/progressive_channel.yaml';cfg=yaml.safe_load(cfgpath.read_text())['quality'];device=torch.device('cuda:0')
    lp,dino,linear=load_quality_models(cfg,device);frozen={'LPIPS':state_sha256(lp),'DINO':state_sha256(dino)}
    code={str(Path(__file__).resolve()):digest(__file__),str(cfgpath):digest(cfgpath),str(Path(linear).resolve()):digest(linear),str(ROOT/'src/var_comm/quality.py'):digest(ROOT/'src/var_comm/quality.py'),str(AUDIT/'rescore_inputs.json'):digest(AUDIT/'rescore_inputs.json')}
    for k in ('alexnet_checkpoint','dino_checkpoint'):code[str(Path(cfg[k]).resolve())]=digest(cfg[k])
    identity={'status':'REGISTERED_REAL_ARCHIVED_RX_COMMON_METRICS','bindings':code,'frozen_metric_states':frozen,'sources':sources,'audit_completion_sha256':digest(AUDIT/'completion.json'),'new_model_forward':False,'new_channel_noise':False,'new_holdout':False,'CPU_source_RGB_float32_0_1':True,'precision':'FP32 deterministic TF32 off','method_decoder_scope':'historical own Decoder; never relabel as common Dc causal comparison'}
    register(OUT/'registration.json',identity);regsha=digest(OUT/'registration.json');all_rows=[];seals={};archive_checked=set()
    for index,record in enumerate(records):
        safe.check();path=OUT/'cells'/f'{index:04d}.json';saved=load_cell(path,regsha,record['image_id'],'all_historical_references')
        if saved is None:
            selected=[r for r in inputs['rows'] if r['image_id']==record['image_id']];images=[];loaded={}
            for row in selected:
                archive=resolve_archive(row['image_archive']);key=str(archive)
                if key not in archive_checked:
                    if digest(archive)!=inputs['archive_sha256'][key]:raise RuntimeError('audited received-image archive changed')
                    archive_checked.add(key)
                if key not in loaded:
                    with np.load(archive,allow_pickle=False) as z:loaded[key]=z['images']
                image=loaded[key][int(row['image_slot'])]
                if array_sha(image)!=row['image_sha256']:raise RuntimeError('audited actual received pixels changed')
                images.append(image)
            reference=record['pixels'].astype(np.float32)/255;scores,*_=quality_metrics(reference,images,lp,dino,device);new=[]
            for row,image,metric in zip(selected,images,scores):
                if abs(metric['psnr_db']-float(row['psnr_db']))>1e-4:raise RuntimeError('same actual pixels no longer reproduce original PSNR')
                new.append({'method':row['method'],'protocol':row.get('protocol') or 'legacy_paid_information','population':'development','source_id':row['image_id'],'source_index':index,'preprocessing_id':record['preprocessing_id'],'snr_db':float(row['snr_db']),'noise_seed':int(row['seed']),'N':int(row['complex_uses']),'E':float(row.get('total_energy') or row['actual_total_energy']),'image_sha256':row['image_sha256'],'original_row_csv_sha256':row['origin_csv_sha256'],'context_sha256':regsha,'mse':float(np.square(image-reference,dtype=np.float64).mean()),**metric,'old_psnr_db':float(row['psnr_db']),'old_lpips_alex':float(row['lpips']),'old_dino_cosine':float(row['dino']),'timing_scope':'historical only; no new TX/RX timing in metric rerun','all_original_attempts_retained':True})
            saved={'registration_sha256':regsha,'source_id':record['image_id'],'method':'all_historical_references','rows':new};save_cell(path,saved)
        all_rows.extend(saved['rows']);seals[path.name]=digest(path)
        write_json(OUT/'metric_status.json',{'status':'SCORING_ACTUAL_HISTORICAL_RX','sources_complete':index+1,'total_sources':100,'rows':len(all_rows),'hardware':safe.hardware})
    verify_snapshot(code)
    if frozen!={'LPIPS':state_sha256(lp),'DINO':state_sha256(dino)}:raise RuntimeError('frozen quality models changed')
    expected={(r['method'],r.get('protocol') or 'legacy_paid_information',r['image_id'],float(r['snr_db']),int(r['seed'])) for r in inputs['rows']}
    actual={(r['method'],r['protocol'],r['source_id'],r['snr_db'],r['noise_seed']) for r in all_rows}
    if actual!=expected or len(all_rows)!=len(expected):raise RuntimeError('full historical attempt coverage')
    write_rows(OUT/'per_frame.csv',all_rows)
    impact={method:{metric:max(abs(r[metric]-r['old_'+metric]) for r in all_rows if r['method']==method) for metric in ('psnr_db','lpips_alex','dino_cosine')} for method in sorted({r['method'] for r in all_rows})}
    write_json(OUT/'historical_metric_impact.json',impact)
    write_json(OUT/'completion.json',{'status':'REAL_HISTORICAL_RX_COMMON_GPU_METRICS_COMPLETE','synthetic':False,'new_holdout':False,'rows':len(all_rows),'sources':100,'registration_sha256':regsha,'per_frame_sha256':digest(OUT/'per_frame.csv'),'cell_sha256':seals,'new_training':False,'new_method_forward':False,'pending':'review and combine source-paired contextual comparisons; audit old N4084 lineage/timing; publish verified artifacts'})

def wait_and_run():
    from latent_enhancement.runtime import digest,write_json,verify_snapshot
    from token_efficiency.common import register
    from token_efficiency.delivery_chain import Runner
    from token_efficiency.coordinator import proc_identity
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    class ReferenceRunner(Runner):
        def status(self,state,**extra):write_json(OUT/'worker_status.json',{'status':state,'pid':os.getpid(),'start_ticks':proc_identity(os.getpid())['start_ticks'],'time':time.time(),**extra})
    runner=ReferenceRunner();registration={'bindings':{str(Path(__file__).resolve()):digest(__file__),str(AUDIT/'completion.json'):digest(AUDIT/'completion.json'),str(AUDIT/'rescore_inputs.json'):digest(AUDIT/'rescore_inputs.json')},'trigger':'C completion and main delivery chain exited; no foreign GPU work','GPU':0,'new_holdout':False}
    register(OUT/'worker_registration.json',registration)
    while True:
        runner.check_stop();verify_snapshot(registration['bindings']);chain=read(BASE/'delivery_chain_v1/status.json');guard=read(BASE/'thermal_guard_v2/status.json')
        if (BASE/'C_followups/completion.json').exists() and chain['status']=='REGISTERED_GRIDS_EXECUTED_REFERENCE_AUDIT_PUBLICATION_AND_REMOTE_ACCEPTANCE_PENDING' and not guard_is_live(chain) and not guard_is_live(guard):break
        if chain['status']=='FAILED_DELIVERY_CHAIN':runner.status('BLOCKED_UPSTREAM_FAILURE_REQUIRES_AGENT_REPAIR');raise SystemExit(75)
        runner.status('WAITING_FOR_REGISTERED_B1_B2_C_RELEASE',delivery_state=chain['status']);time.sleep(30)
    runner.run('reference_common_metrics_v1',['tools.rescore_token_efficiency_references','--run'],OUT/'completion.json')
    runner.status('REAL_REFERENCE_METRICS_COMPLETE_PENDING_REVIEW_AND_PUBLICATION')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--wait',action='store_true');a=p.parse_args()
    if a.run==a.wait:raise ValueError('choose exactly one execution mode')
    from latent_enhancement.runtime import ResourceBusy
    try:run_metrics() if a.run else wait_and_run()
    except ResourceBusy as exc:print('SAFE_REFERENCE_METRIC_PAUSE',str(exc),flush=True);raise SystemExit(75)
if __name__=='__main__':main()
