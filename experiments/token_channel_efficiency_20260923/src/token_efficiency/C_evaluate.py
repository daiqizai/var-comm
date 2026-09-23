"""Real selected C models, including N3060 and independent training-seed repeats."""
import argparse,time
from pathlib import Path
import numpy as np,torch,yaml
from latent_enhancement.runtime import require_available,model_paths,write_json,digest,verify_snapshot,ResourceBusy
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.model import render_received
from latent_enhancement_eval.runner import write_rows
from latent_followup.run_identity import checked_checkpoint
from latent_research.models import PureContinuous
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.quality import load_quality_models,quality_metrics
from short_prefix.models import Hybrid
from short_prefix.data import Population
from short_prefix.train import forward
from short_prefix.execution import execute as hybrid_execute,receive as hybrid_receive
from short_prefix.common import identity_files
from .common import ROOT,EXP,OUT,CONFIG,read,register,config,configure_runtime
from .execution import Cell,execute as continuous_execute,waveform_sha
from .digital_grid import population
from .evaluation_io import SafeEvaluation,load_cell,save_cell,save_preview
from .statistics import FrameTable,canonical_sha


def load_selected(desc,scale,device):
    folder=Path(desc['training']);arm=desc['arm'];rec=read(folder/f'selected_{arm}.json');reg=read(folder/'registration.json');done=read(folder/'completion.json');verify_snapshot(reg['bindings'])
    if rec!=done['selected'][arm] or rec['arm_key']!=arm or rec['registration_sha256']!=digest(folder/'registration.json') or reg['seed']!=desc['training_seed'] or reg.get('N',4084)!=desc['N']:raise RuntimeError('C selected scope/seed/budget identity')
    cp=checked_checkpoint(rec,ROOT);payload=torch.load(cp,map_location='cpu',weights_only=True)
    if payload['registration_sha256']!=rec['registration_sha256'] or payload['state']['step']!=rec['step']:raise RuntimeError('C selected checkpoint identity')
    if arm=='P4084':
        if desc['N']!=4084 or reg['group']!='pure':raise RuntimeError('pure C scope')
        model=PureContinuous(scale.cpu());kind='continuous'
    else:
        m=int(arm[1]);family='VAR' if arm.endswith('-V') else 'PREFIX'
        if reg['group']!=f'm{m}':raise RuntimeError('C arm group')
        model=Hybrid(m,family,scale.cpu(),desc['N']);kind='hybrid'
    prefix=arm+'.';model.load_state_dict({k[len(prefix):]:v for k,v in payload['models'].items() if k.startswith(prefix)},strict=True)
    return model.to(device).eval().requires_grad_(False),{**desc,'kind':kind,'selected':rec,'selected_file':str(folder/f'selected_{arm}.json'),'selected_sha256':digest(folder/f'selected_{arm}.json'),'checkpoint':str(cp),'decoder_sha256':reg['decoder_state_sha256'],'training_completed_step':done['state']['step']}


def execute(record,model,meta,snr,seed,vae,var,decoder,device):
    if meta['kind']=='continuous':
        image,info=continuous_execute(record,Cell('continuous',meta['N']),snr,seed,vae,var,decoder,device,model)
        return image,info
    image,info=hybrid_execute(record,model,snr,seed,vae,var,decoder,device);a=model.ledger;event=info['rx']['phy'];info['received_conditions']=info['rx'];wave=info['waveform']
    if wave.shape!=(a['N'],2) or not np.isfinite(wave).all() or not np.isclose(np.square(wave,dtype=np.float64).sum(),2*a['N'],atol=.02,rtol=1e-5):raise RuntimeError('actual C N/E')
    info.update(ledger={'N':a['N'],'N_header':a['NH'],'N_data':a['ND'],'N_continuous':a['NA'],'E':float(np.square(wave,dtype=np.float64).sum()),'mcs':'QPSK','energy_constraint':'per_frame_2N','source_overflow_erasure':False,'m_actual':a['m']},rx={'header_ok':bool(event['header_ok']),'body_crc_ok':bool(event['body_crc_ok']),'decoded_label':event['decoded_label'],'decoded_mode':event['decoded_mode']},waveform_sha256=waveform_sha(wave),observation_sha256=waveform_sha(info['observation']),timing_endpoints='CPU uint8 RGB -> CPU waveform; CPU observation -> CPU RGB; noise outside RX')
    return image,info

@torch.no_grad()
def qualify_models(models,metadata,vae,var,decoder,scale,device,safe):
    records,_=population('calibration');receipts=[]
    for name,model in models.items():
        meta=metadata[name];m=None if meta['kind']=='continuous' else model.ledger['m'];pop=Population('calibration',m,N=meta['N'])
        for i in range(2):
            record=records[i]
            if pop.ids[i]!=record['image_id']:raise RuntimeError('C cache/population identity')
            for snr in (1,13):
                safe.check();img,info=execute(record,model,meta,snr,4101,vae,var,decoder,device)
                ids=torch.tensor([i]);si=torch.tensor([pop.snrs.index(snr)]);b=pop.batch(ids,si,torch.zeros_like(ids),[4101],device);z,wave=forward(model,b)
                expected=decoder(z) if m is None else render_received(decoder,z,b['status'])
                np.testing.assert_allclose(expected[0].cpu().numpy(),img,atol=2e-5,rtol=1e-5)
                actual_wave=info['waveform'] if m is None else info['waveform'][-model.ledger['NA']:]
                np.testing.assert_allclose(wave[0].cpu().numpy(),actual_wave,atol=2e-5,rtol=1e-5)
                receipts.append({'method':name,'N':meta['N'],'source_id':record['image_id'],'snr_db':snr,'cache_training_online_match':True})
            if m is not None:
                _,clean=hybrid_receive(info['waveform'],19,{'fixed':model},model.family,vae,var,decoder,device,meta['N']);event=clean['phy']
                assert event['header_ok'] and event['body_crc_ok'] and event['decoded_label']==record['class_index'] and event['decoded_mode']==m
        del pop
    return receipts

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);a=parser.parse_args();spec=read(a.spec)
    if spec['new_holdout'] is not False or spec['population']!='development':raise RuntimeError('registered development only')
    configure_runtime();require_available();safe=SafeEvaluation();cfg=config();out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True);device=torch.device('cuda:0')
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device);models={};metadata={}
    for desc in spec['models']:models[desc['method']],metadata[desc['method']]=load_selected(desc,scale,device)
    if len(models)!=len(spec['models']):raise RuntimeError('unique C method identity')
    decoder_sha=state_sha256(decoder)
    if any(r['decoder_sha256']!=decoder_sha for r in metadata.values()):raise RuntimeError('C frozen decoder mismatch')
    qualification=qualify_models(models,metadata,vae,var,decoder,scale,device,safe)
    records,image_bindings=population('development');qcfg=yaml.safe_load((ROOT/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,linear=load_quality_models(qcfg,device)
    code=identity_files([__file__,a.spec,CONFIG,Path(__file__).with_name('digital_grid.py'),Path(__file__).with_name('execution.py'),Path(__file__).with_name('evaluation_io.py'),ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/execution.py',ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/train.py'])
    for r in metadata.values():code[r['checkpoint']]=r['selected']['checkpoint_sha256'];code[r['selected_file']]=r['selected_sha256']
    for k in ('alexnet_checkpoint','dino_checkpoint'):code[str(Path(qcfg[k]).resolve())]=digest(qcfg[k])
    code[str(Path(linear).resolve())]=digest(linear)
    context={'models':metadata,'decoder_sha256':decoder_sha,'bindings':code,'precision':cfg['precision']};context_sha=canonical_sha(context);sources={r['image_id']:r['preprocessing_id'] for r in records}
    identity={'context':context,'context_sha256':context_sha,'sources':sources,'image_bindings':image_bindings,'snrs':cfg['snrs_db'],'seeds':cfg['development_seeds'],'synthetic':False,'new_holdout':False};register(out/'registration.json',identity);regsha=digest(out/'registration.json');register(out/'qualification.json',{'status':'REAL_C_SELECTED_CACHE_ONLINE_REPLAY_PASS','checks':qualification,'registration_sha256':regsha,'synthetic':False})
    all_rows=[];all_timing=[];seals={}
    for i,record in enumerate(records):
        for ci,(name,model) in enumerate(models.items()):
            meta=metadata[name];safe.check();path=out/'cells'/f'{i:04d}_{ci:02d}.json';saved=load_cell(path,regsha,record['image_id'],name)
            if saved is None:
                images=[];rows=[];base_images=[];base_rows=[]
                for snr in cfg['snrs_db']:
                    for seed in cfg['development_seeds']:
                        safe.check();img,info=execute(record,model,meta,snr,seed,vae,var,decoder,device);images.append(img)
                        if meta['kind']=='hybrid':
                            received=info['received_conditions'];event=received['phy']
                            for condition in ('B_RX','C_RX'):
                                if event['header_ok']:
                                    status=torch.tensor([[1.,float(event['body_crc_ok']),model.ledger['m']]],device=device)
                                    base=render_received(decoder,received[condition],status)[0].cpu().numpy()
                                else:base=np.full_like(img,.5)
                                base_images.append(base);base_rows.append({'method':name,'source_id':record['image_id'],'snr_db':snr,'noise_seed':seed,'condition':condition,'N_paid':meta['N'],'enhancement_removed':True,'header_ok':event['header_ok'],'body_crc_ok':event['body_crc_ok']})
                        namespace=f'VAR-CONTINUOUS-{meta["N"]}' if meta['kind']=='continuous' else f'SHORT-PREFIX-v1/N{meta["N"]}/m{model.ledger["m"]}/ND{model.ledger["ND"]}/NA{model.ledger["NA"]}'
                        rows.append({'method':name,'run_id':spec['run_id'],'context_sha256':context_sha,'family':meta['kind'],'population':'development','source_id':record['image_id'],'source_index':i,'preprocessing_id':record['preprocessing_id'],'snr_db':snr,'noise_seed':seed,'noise_namespace':namespace,'training_seed':meta['training_seed'],'arm':meta['arm'],'selected_step':meta['selected']['step'],'training_completed_step':meta['training_completed_step'],'header_ok':info['rx']['header_ok'],'body_crc_ok':info['rx']['body_crc_ok'],'waveform_sha256':info['waveform_sha256'],'observation_sha256':info['observation_sha256'],**info['ledger']})
                reference=record['pixels'].astype(np.float32)/255;metrics,*_=quality_metrics(reference,images,lp,dino,device)
                for row,img,metric in zip(rows,images,metrics):row.update(metric,mse=float(np.square(img-reference,dtype=np.float64).mean()))
                if base_images:
                    base_metrics,*_=quality_metrics(reference,base_images,lp,dino,device)
                    for row,img,metric in zip(base_rows,base_images,base_metrics):row.update(metric,mse=float(np.square(img-reference,dtype=np.float64).mean()))
                timings=[];warmup=[];previews={}
                if i in cfg['fixed_examples']['development_indices']:
                    for si,snr in enumerate(cfg['snrs_db']):previews.update(save_preview(out/'previews'/f'{i:04d}_{ci:02d}_{snr}.png',images[si*3]))
                    for repeat in range(3):
                        safe.check();_,info=execute(record,model,meta,13,2001,vae,var,decoder,device);warmup.append({'method':name,'source_id':record['image_id'],'repeat':repeat,**{k:info[k] for k in ('tx_ms','rx_ms','total_ms')}})
                    for si,snr in enumerate(cfg['snrs_db']):
                        for repeat in range(2):
                            safe.check();img,info=execute(record,model,meta,snr,2001,vae,var,decoder,device);error=float(np.max(np.abs(img-images[si*3])))
                            if error>2e-5 or info['observation_sha256']!=rows[si*3]['observation_sha256']:raise RuntimeError('C quality/full online timing mismatch')
                            timings.append({'method':name,'source_id':record['image_id'],'snr_db':snr,'noise_seed':2001,'repeat':repeat,'N':meta['N'],'E':info['ledger']['E'],'context_sha256':context_sha,'quality_max_abs_error':error,**{k:info[k] for k in ('tx_ms','rx_ms','total_ms')}})
                saved={'registration_sha256':regsha,'source_id':record['image_id'],'method':name,'rows':rows,'base_rows':base_rows,'timing':timings,'warmup':warmup,'previews':previews};save_cell(path,saved)
            verify_snapshot(saved['previews']);FrameTable(saved['rows'],'development',{record['image_id']:record['preprocessing_id']},cfg['snrs_db'],cfg['development_seeds'],[name]);all_rows.extend(saved['rows']);all_timing.extend(saved['timing']);seals[path.name]=digest(path)
            write_json(out/'status.json',{'status':'REAL_C_SELECTED_EVALUATION','completed_cells':len(seals),'expected_cells':len(records)*len(models),'hardware':safe.hardware})
        print('C development',i+1,flush=True)
    verify_snapshot(code);assert state_sha256(decoder)==decoder_sha
    table=FrameTable(all_rows,'development',sources,cfg['snrs_db'],cfg['development_seeds'],list(models));write_rows(out/'per_frame.csv',all_rows);write_rows(out/'summary.csv',table.summary());write_rows(out/'timing.csv',all_timing)
    write_json(out/'completion.json',{'status':'REAL_C_SELECTED_GRID_COMPLETE','registration_sha256':regsha,'table_sha256':table.sha256,'frame_rows':len(all_rows),'timed_calls':len(all_timing),'files':{p.name:digest(p) for p in out.glob('*.csv')},'cell_sha256':seals,'synthetic':False,'new_holdout':False})
if __name__=='__main__':
    try:main()
    except ResourceBusy as exc:print('SAFE_C_EVALUATION_PAUSE',str(exc),flush=True);raise SystemExit(75)
