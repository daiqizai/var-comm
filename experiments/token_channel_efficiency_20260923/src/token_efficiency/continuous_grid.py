"""Selected low-budget models plus verified historical P4084, shared online quality/timing."""
from pathlib import Path
import time,numpy as np,torch,yaml
from latent_enhancement.runtime import model_paths,write_json,digest,verify_snapshot,require_available,ResourceBusy
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_eval.runner import write_rows
from latent_followup.run_identity import checked_checkpoint
from latent_research.models import PureContinuous
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.quality import load_quality_models,quality_metrics
from var_comm.study import seeded_noise
from short_prefix.common import identity_files
from .common import ROOT,EXP,OUT,CONFIG,read,register,config,configure_runtime
from .execution import Cell,execute,receive,load_selected_budget
from .digital_grid import population,require_real_execution_acceptance
from .evaluation_io import SafeEvaluation,load_cell,save_cell,save_preview
from .statistics import FrameTable,canonical_sha

LEGACY=ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/followup/research_20260923_training_v3'

def load_legacy(scale,device):
    selected=read(LEGACY/'selected_pure_continuous.json');done=read(LEGACY/'completion.json');reg=read(LEGACY/'registration.json');verify_snapshot(reg['bindings'])
    if selected!=done['selected']['pure_continuous'] or selected['registration_sha256']!=digest(LEGACY/'registration.json') or selected['arm_key']!='pure_continuous':raise RuntimeError('legacy selected provenance')
    cp=checked_checkpoint(selected,ROOT);payload=torch.load(cp,map_location='cpu',weights_only=True)
    if payload['registration_sha256']!=selected['registration_sha256'] or payload['state']['step']!=selected['step']:raise RuntimeError('legacy checkpoint identity')
    model=PureContinuous(scale.cpu());prefix='pure_continuous.';model.load_state_dict({k[len(prefix):]:v for k,v in payload['models'].items() if k.startswith(prefix)},strict=True)
    return model.to(device).eval().requires_grad_(False),{'selected':str(LEGACY/'selected_pure_continuous.json'),'selected_sha256':digest(LEGACY/'selected_pure_continuous.json'),'checkpoint':str(cp),'checkpoint_sha256':digest(cp),'N':4084,'step':selected['step'],'decoder_sha256':reg['decoder_state_sha256'],'historical_training_retained':True,'recomputed_shared_online_evaluation':True}

@torch.no_grad()
def main():
    configure_runtime();require_available();require_real_execution_acceptance();cfg=config();out=OUT/'continuous_grid_v1';out.mkdir(exist_ok=True);safe=SafeEvaluation()
    finals={}
    for N in (2048,3060):
        p=OUT/f'training/P{N}_seed2026092304';f=read(p/'finalization.json')
        if f['status']!='CALIBRATION_EXTENSION_RULE_STOP_SUPPORTED' or f['selected']!=read(p/f'selected_P{N}.json') or digest(f['decision'])!=f['decision_sha256']:raise RuntimeError('calibration-only finalization required before development')
        finals[str(N)]={'path':str(p/'finalization.json'),'sha256':digest(p/'finalization.json')}
    device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device);models={};selected={}
    for N in (2048,3060):models[N],selected[str(N)]=load_selected_budget(OUT/f'training/P{N}_seed2026092304/selected_P{N}.json',scale,device)
    models[4084],selected['4084']=load_legacy(scale,device)
    decoder_sha=state_sha256(decoder)
    if any(r['decoder_sha256']!=decoder_sha for r in selected.values()):raise RuntimeError('same frozen Decoder required')
    # Actual selected weights are replay-qualified on calibration only first.
    cal,_=population('calibration');qualification=[]
    for N,model in models.items():
        for record in cal[:2]:
            for snr in (1,13):
                safe.check();image,info=execute(record,Cell('continuous',N),snr,4101,vae,var,decoder,device,model)
                replay,_=receive(info['observation'],snr,Cell('continuous',N),vae,var,decoder,device,model);np.testing.assert_array_equal(image,replay)
                F=vae.quant_conv(vae.encoder(torch.as_tensor(record['pixels'][None],device=device,dtype=torch.float32)/127.5-1))
                noise=torch.as_tensor(seeded_noise(f'VAR-CONTINUOUS-{N}|'+record['image_id'],4101,(N,2)),device=device,dtype=torch.float32)[None]
                latent,wave=model(F,torch.tensor([snr],device=device),noise)
                np.testing.assert_array_equal(wave[0].cpu().numpy(),info['waveform']);np.testing.assert_allclose(decoder(latent)[0].cpu().numpy(),image,atol=2e-5,rtol=1e-5)
                qualification.append({'N':N,'source_id':record['image_id'],'snr_db':snr,'actual_selected_replay_pass':True,'checkpoint_sha256':selected[str(N)]['checkpoint_sha256']})
    records,image_bindings=population('development');qcfg=yaml.safe_load((ROOT/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,linear=load_quality_models(qcfg,device)
    local=Path(__file__).parent;code=identity_files([CONFIG,EXP/'quality_targets.json',*[local/n for n in ('continuous_grid.py','digital_grid.py','execution.py','evaluation_io.py','microbatch_runtime.py','statistics.py','thermal_guard.py')]])
    for r in selected.values():code[r['checkpoint']]=r['checkpoint_sha256'];code[r['selected']]=r['selected_sha256']
    for k in ('alexnet_checkpoint','dino_checkpoint'):code[str(Path(qcfg[k]).resolve())]=digest(qcfg[k])
    code[str(Path(linear).resolve())]=digest(linear)
    context={'selected':selected,'decoder_sha256':decoder_sha,'bindings':code,'precision':cfg['precision'],'finalizations':finals};context_sha=canonical_sha(context)
    sources={r['image_id']:r['preprocessing_id'] for r in records};identity={'context':context,'context_sha256':context_sha,'sources':sources,'image_bindings':image_bindings,'snrs':cfg['snrs_db'],'seeds':cfg['development_seeds'],'population':'development','synthetic':False}
    register(out/'registration.json',identity);regsha=digest(out/'registration.json');register(out/'qualification.json',{'status':'REAL_SELECTED_CONTINUOUS_REPLAY_PASS','registration_sha256':regsha,'checks':qualification,'synthetic':False})
    all_rows=[];all_timing=[];all_warm=[];seals={}
    for i,record in enumerate(records):
        for N,model in models.items():
            safe.check();cell=Cell('continuous',N);path=out/'cells'/f'{i:04d}_{N}.json';saved=load_cell(path,regsha,record['image_id'],cell.name)
            if saved is None:
                images=[];rows=[]
                for snr in cfg['snrs_db']:
                    for seed in cfg['development_seeds']:
                        safe.check();img,info=execute(record,cell,snr,seed,vae,var,decoder,device,model);images.append(img)
                        rows.append({'method':cell.name,'run_id':cfg['study']+'/continuous_grid_v1','context_sha256':context_sha,'family':'continuous','population':'development','source_id':record['image_id'],'source_index':i,'preprocessing_id':record['preprocessing_id'],'snr_db':snr,'noise_seed':seed,'noise_namespace':f'VAR-CONTINUOUS-{N}','noise_id':f'VAR-CONTINUOUS-{N}|{record["image_id"]}|{seed}','header_ok':'not_applicable','body_crc_ok':'not_applicable','checkpoint_sha256':selected[str(N)]['checkpoint_sha256'],'selected_step':selected[str(N)]['step'],'waveform_sha256':info['waveform_sha256'],'observation_sha256':info['observation_sha256'],**info['ledger']})
                reference=record['pixels'].astype(np.float32)/255;metrics,*_=quality_metrics(reference,images,lp,dino,device)
                for r,img,metric in zip(rows,images,metrics):r.update(metric,mse=float(np.mean(np.square(img-reference),dtype=np.float64)))
                timing=[];warm=[]
                if i in cfg['fixed_examples']['development_indices']:
                    for repeat in range(3):
                        safe.check();_,info=execute(record,cell,13,2001,vae,var,decoder,device,model);warm.append({'method':cell.name,'source_id':record['image_id'],'repeat':repeat,**{k:info[k] for k in ('tx_ms','rx_ms','total_ms')}})
                    for si,snr in enumerate(cfg['snrs_db']):
                        for repeat in range(2):
                            safe.check();img,info=execute(record,cell,snr,2001,vae,var,decoder,device,model);error=float(np.max(np.abs(img-images[si*3])))
                            if error>2e-5 or info['observation_sha256']!=rows[si*3]['observation_sha256']:raise RuntimeError('continuous quality/timing mismatch')
                            timing.append({'method':cell.name,'source_id':record['image_id'],'preprocessing_id':record['preprocessing_id'],'snr_db':snr,'noise_seed':2001,'repeat':repeat,'N':N,'E':info['ledger']['E'],'context_sha256':context_sha,'run_id':rows[0]['run_id'],'quality_max_abs_error':error,'timing_endpoints':info['timing_endpoints'],**{k:info[k] for k in ('tx_ms','rx_ms','total_ms')}})
                previews={}
                if i in cfg['fixed_examples']['development_indices']:
                    for si,snr in enumerate(cfg['snrs_db']):previews.update(save_preview(out/'previews'/f'{i:04d}_{N}_{snr}.png',images[si*3]))
                saved={'previews':previews,'registration_sha256':regsha,'source_id':record['image_id'],'method':cell.name,'rows':rows,'timing':timing,'warmup':warm};save_cell(path,saved)
            verify_snapshot(saved['previews'])
            FrameTable(saved['rows'],'development',{record['image_id']:record['preprocessing_id']},cfg['snrs_db'],cfg['development_seeds'],[cell.name]);all_rows.extend(saved['rows']);all_timing.extend(saved['timing']);all_warm.extend(saved['warmup']);seals[path.name]=digest(path)
            write_json(out/'status.json',{'status':'REAL_CONTINUOUS_EVALUATION','completed_cells':len(seals),'expected_cells':len(records)*3,'hardware':safe.hardware})
        print('continuous development',i+1,flush=True)
    verify_snapshot(code);assert state_sha256(decoder)==decoder_sha
    table=FrameTable(all_rows,'development',sources,cfg['snrs_db'],cfg['development_seeds'],[f'P{N}' for N in models]);write_rows(out/'per_frame.csv',all_rows);write_rows(out/'summary.csv',table.summary());write_rows(out/'timing.csv',all_timing);write_rows(out/'warmup.csv',all_warm)
    write_json(out/'completion.json',{'status':'REAL_CONTINUOUS_GRID_COMPLETE','registration_sha256':regsha,'frame_rows':len(all_rows),'table_sha256':table.sha256,'timed_calls':len(all_timing),'files':{p.name:digest(p) for p in out.glob('*.csv')},'cell_sha256':seals,'synthetic':False,'new_holdout':False})
if __name__=='__main__':
    try:main()
    except ResourceBusy as exc:print('SAFE_EVALUATION_PAUSE',str(exc),flush=True);raise SystemExit(75)
