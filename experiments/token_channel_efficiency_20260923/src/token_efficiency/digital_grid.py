"""Actual digital calibration/development grids using one shared TX/channel/RX.

Prepared TX source computation is reused for offline quality only. All reported
online timings start from CPU pixels and repeat the complete TX computation.
"""
import argparse,hashlib,time
from pathlib import Path
import numpy as np,torch,yaml
from latent_enhancement.runtime import model_paths,write_json,digest,verify_snapshot,require_available,ResourceBusy
from latent_enhancement_b.common import load_decoder
from latent_enhancement_eval.runner import load_targets,write_rows
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.prefix_training_data import read_image_population
from var_comm.quality import load_quality_models,quality_metrics
from short_prefix.common import identity_files
from .common import ROOT,EXP,OUT,CONFIG,read,register,config,configure_runtime
from .execution import Cell,prepare_digital,transmit_prepared,apply_channel,receive,execute,waveform_sha
from .phy import dimensions,PROTOCOL
from .codec import SIZES
from .statistics import FrameTable,canonical_sha,freeze_digital_policy,validate_policy_on_development
from .evaluation_io import SafeEvaluation,load_cell,save_cell,save_preview

VERSION='digital_grid_v1'

def candidate_cells(mcs):
    cells=[];ledger=[]
    for family in ('raw','arithmetic'):
        for N in (2048,3060,4084):
            for m in range(6,11):
                cell=Cell(family,N,m,mcs);nh,nd,slots=dimensions(N,family,mcs);raw_bits=12*sum(s*s for s in SIZES[:m])
                eligible=family=='arithmetic' or raw_bits+22<=slots
                ledger.append({'method':cell.name,'family':family,'N':N,'m_requested':m,'mcs':mcs,'N_header':nh,'N_data':nd,'raw_source_bits':raw_bits,'coded_slots':slots,'status':'ELIGIBLE' if eligible else 'UNENCODABLE_INFORMATION_LENGTH','per_source_arithmetic_fallback_charged':family=='arithmetic'})
                if eligible:cells.append(cell)
    return cells,ledger

def population(role):
    if role=='calibration':
        images,labels,ids,bindings=read_image_population(role)
        records=[{'pixels':images[i].numpy(),'class_index':int(labels[i]),'image_id':ids[i]} for i in range(len(ids))]
    elif role=='development':
        targets=load_targets();records=[{'pixels':r['pixels'],'class_index':int(r['target']['class_index']),'image_id':r['target']['image_id']} for r in targets];bindings=[{k:r[k] for k in ('index','rgb_sha256','source_npz_sha256')} for r in targets]
    else:raise ValueError('registered calibration/development only; no holdout')
    if len(records)!=config()['data'][role] or len({r['image_id'] for r in records})!=len(records):raise RuntimeError('population scope')
    for r in records:
        if r['pixels'].dtype!=np.uint8 or r['pixels'].shape!=(3,256,256):raise RuntimeError('actual CPU pixels')
        r['preprocessing_id']=hashlib.sha256(r['pixels'].tobytes()).hexdigest()
    return records,bindings

def require_real_execution_acceptance():
    folder=OUT/'execution_qualification_v1';done=read(folder/'acceptance.json');reg=read(folder/'registration.json')
    if done['status']!='REAL_WEIGHT_SHARED_EXECUTION_PASS' or done['synthetic'] is not False or done['registration_sha256']!=digest(folder/'registration.json'):raise RuntimeError('real shared execution qualification required')
    verify_snapshot(reg['bindings'])
    for name in ('execution.py','phy.py','codec.py','models.py','common.py'):
        path=Path(__file__).with_name(name)
        if reg['bindings'].get(str(path.resolve()))!=digest(path):raise RuntimeError('shared execution qualification source mismatch')
    return done

def validate_cell_rows(rows,role,record,cell):
    return FrameTable(rows,role,{record['image_id']:record['preprocessing_id']},config()['snrs_db'],config()[role+'_seeds'],[cell.name])

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['calibration','development'],required=True);parser.add_argument('--mcs',choices=['QPSK','16QAM'],required=True);a=parser.parse_args()
    configure_runtime();require_available();accept=require_real_execution_acceptance();cfg=config();folder=OUT/VERSION/a.mcs/a.role;folder.mkdir(parents=True,exist_ok=True)
    calfolder=folder.parent/'calibration';policy=None
    targets=read(EXP/'quality_targets.json')
    if targets['status']!='FROZEN_FROM_CALIBRATION_BEFORE_NEW_DEVELOPMENT' or targets['source_csv_sha256']!=digest(OUT/'source/calibration/source_codec_per_image.csv'):raise RuntimeError('frozen quality target identity')
    if a.role=='development':
        done=read(calfolder/'completion.json');policy=read(calfolder/'policy.json')
        if done['status']!='REAL_DIGITAL_GRID_COMPLETE' or done['policy_sha256']!=digest(calfolder/'policy.json') or done['synthetic'] is not False:raise RuntimeError('full calibration policy must precede development access')
        verify_snapshot({str(calfolder/n):sha for n,sha in done['files'].items()})
    records,image_bindings=population(a.role);cells,candidate_ledger=candidate_cells(a.mcs)
    device=torch.device('cuda:0');paths=model_paths();vae,var=load_models(paths,device);decoder=load_decoder(vae,device)
    qcfg=yaml.safe_load((ROOT/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,linear=load_quality_models(qcfg,device)
    local=Path(__file__).parent;files=[CONFIG,EXP/'quality_targets.json',*[local/n for n in ('digital_grid.py','evaluation_io.py','execution.py','phy.py','codec.py','statistics.py','common.py','thermal_guard.py')]]
    code=identity_files(files)
    for name in ('vae_checkpoint','var_checkpoint'):code[str(Path(paths[name]).resolve())]=digest(paths[name])
    for name in ('alexnet_checkpoint','dino_checkpoint'):code[str(Path(qcfg[name]).resolve())]=digest(qcfg[name])
    code[str(Path(linear).resolve())]=digest(linear)
    context={'bindings':code,'decoder_sha256':state_sha256(decoder),'precision':cfg['precision'],'qualification_sha256':digest(OUT/'execution_qualification_v1/acceptance.json'),'family_version':PROTOCOL,'candidate_ledger':candidate_ledger}
    context_sha=canonical_sha(context)
    if context['decoder_sha256']!=accept['frozen_model_states']['Dc']:raise RuntimeError('qualified decoder changed')
    identity={'version':VERSION,'role':a.role,'mcs':a.mcs,'context':context,'context_sha256':context_sha,'sources':{r['image_id']:r['preprocessing_id'] for r in records},'image_bindings':image_bindings,'snrs':cfg['snrs_db'],'seeds':cfg[a.role+'_seeds'],'policy_sha256':digest(calfolder/'policy.json') if policy else None,'new_holdout':False,'synthetic':False}
    register(folder/'registration.json',identity);regsha=digest(folder/'registration.json');write_json(folder/'candidates.json',candidate_ledger)
    if policy:
        if policy['status']!='FROZEN_CALIBRATION_POLICY' or any(c['context_sha256']!=context_sha for c in policy['choices']):raise RuntimeError('policy execution context changed before development')
    safe=SafeEvaluation();all_rows=[];all_timing=[];all_warmup=[];seals={};started=time.time()
    for i,record in enumerate(records):
        prepared={}
        for ci,cell in enumerate(cells):
            safe.check();path=folder/'cells'/f'{i:04d}_{ci:02d}.json';saved=load_cell(path,regsha,record['image_id'],cell.name)
            if saved is None:
                if cell.family not in prepared:prepared[cell.family]=prepare_digital(record['pixels'],record['class_index'],cell.family,vae,var,device)
                wave,ledger=transmit_prepared(prepared[cell.family],record['class_index'],cell);images=[];rows=[]
                for snr in cfg['snrs_db']:
                    for seed in cfg[a.role+'_seeds']:
                        safe.check();observed=apply_channel(wave,snr,record['image_id'],seed,cell);img,event=receive(observed,snr,cell,vae,var,decoder,device)
                        images.append(img)
                        rows.append({'method':cell.name,'run_id':cfg['study']+'/'+VERSION+'/'+a.mcs+'/'+a.role,'context_sha256':context_sha,'family':cell.family,'population':a.role,'source_id':record['image_id'],'source_index':i,'preprocessing_id':record['preprocessing_id'],'snr_db':snr,'noise_seed':seed,'noise_id':f'{PROTOCOL}/{cell.family}/{a.mcs}/N{cell.N}|{record["image_id"]}|{seed}','noise_namespace':f'{PROTOCOL}/{cell.family}/{a.mcs}/N{cell.N}','header_ok':bool(event['header_ok']),'header_crc_ok':bool(event['header_crc_ok']),'body_crc_ok':bool(event['body_crc_ok']),'source_complete':bool(event['source_complete']),'source_error':event['source_error'],'decoded_label':event['decoded_label'],'decoded_mode':event['decoded_mode'],'rx_source_overflow_erasure':bool(event['source_overflow_erasure']),'waveform_sha256':waveform_sha(wave),'observation_sha256':waveform_sha(observed),**ledger})
                reference=record['pixels'].astype(np.float32)/255
                metrics,*_=quality_metrics(reference,images,lp,dino,device)
                for row,img,metric in zip(rows,images,metrics):row.update(metric,mse=float(np.mean(np.square(img-reference),dtype=np.float64)))
                validate_cell_rows(rows,a.role,record,cell)
                timing=[];warmup=[]
                if a.role=='development' and i in cfg['fixed_examples']['development_indices']:
                    # Full online TX, including all source computation, every time.
                    for repeat in range(3):
                        safe.check();_,info=execute(record,cell,13,2001,vae,var,decoder,device)
                        warmup.append({'method':cell.name,'source_id':record['image_id'],'repeat':repeat,**{k:info[k] for k in ('tx_ms','rx_ms','total_ms')}})
                    for si,snr in enumerate(cfg['snrs_db']):
                        for repeat in range(2):
                            safe.check();img,info=execute(record,cell,snr,2001,vae,var,decoder,device);error=float(np.max(np.abs(img-images[si*3])))
                            if error>2e-5 or info['waveform_sha256']!=rows[si*3]['waveform_sha256'] or info['observation_sha256']!=rows[si*3]['observation_sha256']:raise RuntimeError('offline quality/full online timing mismatch')
                            timing.append({'method':cell.name,'source_id':record['image_id'],'preprocessing_id':record['preprocessing_id'],'snr_db':snr,'noise_seed':2001,'repeat':repeat,'N':cell.N,'E':info['ledger']['E'],'context_sha256':context_sha,'run_id':rows[0]['run_id'],'quality_max_abs_error':error,'timing_endpoints':info['timing_endpoints'],**{k:info[k] for k in ('tx_ms','rx_ms','total_ms')}})
                previews={}
                if a.role=='development' and i in cfg['fixed_examples']['development_indices']:
                    for si,snr in enumerate(cfg['snrs_db']):previews.update(save_preview(folder/'previews'/f'{i:04d}_{ci:02d}_{snr}.png',images[si*3]))
                saved={'previews':previews,'registration_sha256':regsha,'source_id':record['image_id'],'method':cell.name,'rows':rows,'timing':timing,'warmup':warmup};save_cell(path,saved)
            verify_snapshot(saved['previews']);validate_cell_rows(saved['rows'],a.role,record,cell);all_rows.extend(saved['rows']);all_timing.extend(saved['timing']);all_warmup.extend(saved['warmup']);seals[path.name]=digest(path)
            write_json(folder/'status.json',{'status':'REAL_DIGITAL_GRID_RUNNING','source_index':i,'cell_index':ci,'completed_cells':len(seals),'expected_cells':len(records)*len(cells),'elapsed_seconds_this_process':time.time()-started,'hardware':safe.hardware,'synthetic':False})
        print('digital',a.mcs,a.role,i+1,len(records),flush=True)
    verify_snapshot(code);assert state_sha256(decoder)==context['decoder_sha256']
    table=FrameTable(all_rows,a.role,identity['sources'],identity['snrs'],identity['seeds'],[c.name for c in cells])
    if a.role=='calibration':register(folder/'policy.json',freeze_digital_policy(table))
    else:validate_policy_on_development(table,policy)
    write_rows(folder/'per_frame.csv',all_rows);write_rows(folder/'summary.csv',table.summary())
    if all_timing:write_rows(folder/'timing.csv',all_timing);write_rows(folder/'warmup.csv',all_warmup)
    write_json(folder/'completion.json',{'status':'REAL_DIGITAL_GRID_COMPLETE','registration_sha256':regsha,'role':a.role,'mcs':a.mcs,'sources':len(records),'eligible_methods':len(cells),'frame_rows':len(all_rows),'table_sha256':table.sha256,'policy_sha256':digest(calfolder/'policy.json'),'files':{p.name:digest(p) for p in folder.glob('*.csv')},'cell_sha256':seals,'timed_calls':len(all_timing),'synthetic':False,'new_holdout':False})
    print('DIGITAL_GRID_COMPLETE',a.mcs,a.role,len(all_rows),flush=True)

if __name__=='__main__':
    try:main()
    except ResourceBusy as exc:print('SAFE_EVALUATION_PAUSE',str(exc),flush=True);raise SystemExit(75)
