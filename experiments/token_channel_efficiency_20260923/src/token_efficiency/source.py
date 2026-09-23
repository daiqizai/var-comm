"""Actual source-only codec and representation measurements, calibration first."""
import argparse,csv,hashlib,signal,time
from pathlib import Path
import numpy as np
import torch,yaml
from latent_enhancement.runtime import model_paths,write_json,digest,verify_snapshot,require_available,ResourceBusy
from latent_enhancement.latent import complete_latent,original_rgb
from latent_enhancement_b.common import CACHE,load_decoder
from latent_enhancement_eval.runner import load_targets,write_rows
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.prefix_training_data import read_image_population,IMAGE_CACHE
from var_comm.progressive import split_prefix
from var_comm.quality import load_quality_models,quality_metrics
from var_comm.scale_channel import bits_to_indices
from short_prefix.common import identity_files
from .common import ROOT,EXP,OUT,CONFIG,read,register,config,configure_runtime,Safety,bindings
from .codec import encode_all,decode_arithmetic,cumulative,write_stream

METRICS=('psnr_db','lpips_alex','dino_cosine')

def freeze_targets(rows,source_sha):
    """Only complete calibration source rows may determine target thresholds."""
    if not rows or {r['population'] for r in rows}!={'calibration'}:raise ValueError('calibration only')
    refs={'high':'Dc_Fq','balanced':'Dc_VAR_m8','coarse':'Dc_prefix_m6'}
    targets={}
    for level,method in refs.items():
        chosen=[r for r in rows if r['method']==method]
        if len(chosen)!=1000 or len({r['source_id'] for r in chosen})!=1000:raise ValueError('complete unique calibration references required')
        values=np.asarray([[float(r[k]) for k in METRICS] for r in chosen])
        if not np.isfinite(values).all():raise ValueError('nonfinite reference metrics')
        targets[level]={'reference':method,'psnr_min':float(np.median(values[:,0])),'lpips_max':float(np.median(values[:,1])),'calibration_sources':len(chosen)}
    return {'status':'FROZEN_FROM_CALIBRATION_BEFORE_NEW_DEVELOPMENT','rule':config()['quality_targets_rule'],'source_csv_sha256':source_sha,'targets':targets,'keep_unreachable_targets':True,'finite_channel_quality_claim':False}

def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['population'],r['method']),[]).append(r)
    result=[]
    for (population,method),group in sorted(groups.items()):
        record={'population':population,'method':method,'sources':len(group),'N':'not_applicable_source_reference'}
        for k in ('raw_bits','arithmetic_bits','selected_payload_bits',*METRICS):
            values=[float(r[k]) for r in group if r.get(k) not in ('',None)]
            if values:
                x=np.asarray(values)
                if not np.isfinite(x).all():raise ValueError('nonfinite summary')
                for key,value in zip(('mean','median','p90','p95','min','max'),(x.mean(),np.median(x),np.quantile(x,.9),np.quantile(x,.95),x.min(),x.max())):record[f'{k}_{key}']=float(value)
        fallback=[int(r['raw_fallback']) for r in group if r.get('raw_fallback') not in ('',None)]
        if fallback:record['raw_fallback_fraction']=float(np.mean(fallback))
        result.append(record)
    return result

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['calibration','development'],required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    if a.preflight and a.role!='calibration':raise ValueError('preflight calibration only')
    configure_runtime();require_available();device=torch.device('cuda:0')
    folder=OUT/('source_preflight' if a.preflight else 'source')/a.role;folder.mkdir(parents=True,exist_ok=True)
    # Access to new development metrics is gated before loading its population.
    targets=None
    if a.role=='development':
        targets=read(EXP/'quality_targets.json')
        cal=OUT/'source/calibration';receipt=read(cal/'completion.json')
        if receipt['status']!='REAL_SOURCE_CODEC_COMPLETE' or digest(cal/'source_codec_per_image.csv')!=targets['source_csv_sha256']:raise RuntimeError('quality target freeze mismatch')
    qcfg=yaml.safe_load((ROOT/'configs/progressive_channel.yaml').read_text())['quality']
    if a.role=='calibration':
        images,labels,ids,image_bindings=read_image_population('calibration');records=None
        data_bindings={str(IMAGE_CACHE/'manifest.json'):digest(IMAGE_CACHE/'manifest.json')}
        for p in sorted((CACHE/'calibration').glob('shard_*.pt')):
            if digest(p)!=read(p.with_suffix('.json'))['sha256']:raise RuntimeError('source shard identity')
            data_bindings[str(p)]=digest(p)
    else:
        records=load_targets();ids=[r['target']['image_id'] for r in records];images=torch.as_tensor(np.stack([r['pixels'] for r in records]));labels=torch.tensor([r['target']['class_index'] for r in records]);image_bindings=[{k:r[k] for k in ('index','rgb_sha256','source_npz_sha256')} for r in records];data_bindings={}
    if len(ids)!=config()['data'][a.role] or len(set(ids))!=len(ids):raise RuntimeError('source population')
    files=[CONFIG,Path(__file__),Path(__file__).with_name('codec.py'),Path(__file__).with_name('common.py'),ROOT/'experiments/var-latent-enhancement-20260917/evaluation/src/latent_enhancement_eval/runner.py']
    code_bindings=identity_files(files)
    for key in ('alexnet_checkpoint','dino_checkpoint'):code_bindings[str(Path(qcfg[key]).resolve())]=digest(qcfg[key])
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);lp,dino,linear=load_quality_models(qcfg,device);code_bindings[str(linear.resolve())]=digest(linear)
    identity={'study':config()['study'],'protocol_sha256':digest(CONFIG),'role':a.role,'preflight':a.preflight,'ids':ids[:2] if a.preflight else ids,'image_bindings':image_bindings,'data_bindings':data_bindings,'bindings':code_bindings,'decoder_sha256':state_sha256(decoder),'precision':config()['precision'],'targets':targets}
    register(folder/'registration.json',identity);regsha=digest(folder/'registration.json');limit=2 if a.preflight else len(ids)
    safety=Safety();stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    all_rows=[];ledger=[];receipts={};started=time.time();shard_index=-1;shard=None
    for i in range(limit):
        if stop[0] or safety.check():raise ResourceBusy('safe source boundary pause')
        path=folder/'images'/f'{i:04d}.json'
        if path.exists():
            rec=read(path)
            if rec['registration_sha256']!=regsha or rec['source_id']!=ids[i]:raise RuntimeError('source resume scope mismatch')
            verify_snapshot(rec['stream_files']);all_rows.extend(rec['rows']);ledger.extend(rec['ledger']);receipts[path.name]=digest(path);continue
        if a.role=='calibration':
            if i//100!=shard_index:
                shard_index=i//100;shard=torch.load(CACHE/'calibration'/f'shard_{shard_index:04d}.pt',map_location='cpu',weights_only=True)
            j=i%100
            if shard['image_ids'][j]!=ids[i] or int(shard['labels'][j])!=int(labels[i]):raise RuntimeError('cache/source ordering')
            F=shard['F'][j:j+1].to(device);scales=split_prefix(shard['full_tokens'][j].numpy(),10)
        else:
            F=vae.quant_conv(vae.encoder(images[i:i+1].to(device).float()/127.5-1));scales=split_prefix(records[i]['tokens'],10)
        # Verify actual frozen encoder and official quantizer at every shard boundary.
        if i%100==0:
            fresh_F=vae.quant_conv(vae.encoder(images[i:i+1].to(device).float()/127.5-1))
            torch.testing.assert_close(F,fresh_F,atol=2e-5,rtol=1e-5)
            fresh_tokens=vae.quantize.f_to_idxBl_or_fhat(F,to_fhat=False)
            for x,y in zip(fresh_tokens,scales):np.testing.assert_array_equal(x[0].cpu().numpy(),y)
        base={'run_id':config()['study']+('/preflight' if a.preflight else '/source'),'population':a.role,'source_id':ids[i],'source_index':i,'preprocessing_id':hashlib.sha256(images[i].numpy().tobytes()).hexdigest(),'N':'not_applicable_source_reference','E':'not_applicable_source_reference','synthetic':False}
        encoded=encode_all(vae,var,scales,int(labels[i]),device);streams={};source_rows=[];costs=[]
        streamdir=folder/'streams'/f'{i:04d}';streamdir.mkdir(parents=True,exist_ok=True)
        for m,data in encoded.items():
            np.testing.assert_array_equal(bits_to_indices(data['raw']),np.concatenate(scales[:m]))
            restored,padding=decode_arithmetic(data['arithmetic'],m,int(labels[i]),vae,var,device)
            for x,y in zip(restored,scales[:m]):np.testing.assert_array_equal(x,y)
            for family,key in (('raw','raw'),('arithmetic','arithmetic')):
                output=streamdir/f'm{m}_{family}.bin';meta=write_stream(output,data[key]);streams[str(output)]=digest(output)
                costs.append({**base,'m':m,'family':family,'source_bits':len(data[key]),'selected_source_bits':len(data['raw' if family=='raw' else 'payload']),'attempted_arithmetic_bits':len(data['arithmetic']) if family=='arithmetic' else '', 'selected_stream_family':'raw' if family=='raw' or data['raw_fallback'] else 'arithmetic','finish_flush_bits':data['finish_flush_bits'] if family=='arithmetic' else 0,'raw_fallback':int(data['raw_fallback']) if family=='arithmetic' else 0,'class_control_bits':10,'mode_control_bits':3,'length_control_bits':13 if family=='arithmetic' else 0,'new_protocol_header_uses':96 if family=='arithmetic' else 70,'body_crc_bits':16,'body_tail_bits':6,'source_sha256':digest(output),'decoder_implicit_zero_reads':padding if family=='arithmetic' else 0,**meta})
            source_rows.append({**base,'method':f'codec_m{m}','m':m,'raw_bits':len(data['raw']),'arithmetic_bits':len(data['arithmetic']),'selected_payload_bits':len(data['payload']),'raw_fallback':int(data['raw_fallback']),'roundtrip_tokens_equal':True})
        latents={'F':F,'Fq':cumulative(vae,scales,10)}
        for m in range(6,10):
            latents[f'prefix_m{m}']=cumulative(vae,scales,m)
            latents[f'VAR_m{m}']=complete_latent(vae,var,scales[:m],int(labels[i]),device)
        names=[];preds=[]
        for name,z in latents.items():
            for decoder_name in ('D0','Dc'):
                pred=original_rgb(vae,z) if decoder_name=='D0' else decoder(z)
                if not torch.isfinite(pred).all():raise FloatingPointError('source decoder')
                names.append(f'{decoder_name}_{name}');preds.append(pred[0].cpu().numpy())
        metrics,*_=quality_metrics(images[i].numpy().astype(np.float32)/255,preds,lp,dino,device)
        for name,metric in zip(names,metrics):
            if not np.isfinite(list(metric.values())).all():raise FloatingPointError('quality')
            source_rows.append({**base,'method':name,**metric})
        rec={'source_id':ids[i],'registration_sha256':regsha,'rows':source_rows,'ledger':costs,'stream_files':streams};write_json(path,rec)
        all_rows.extend(source_rows);ledger.extend(costs);receipts[path.name]=digest(path)
        write_json(folder/'status.json',{'status':'MEASURING_REAL_SOURCE','completed_sources':i+1,'expected_sources':limit,'hardware':safety.last,'seconds':time.time()-started});print('source',a.role,i+1,flush=True)
    if len(all_rows)!=25*limit or len(ledger)!=10*limit:raise RuntimeError('incomplete source table')
    verify_snapshot(code_bindings);verify_snapshot(data_bindings)
    for name,rows in [('source_codec_per_image.csv',all_rows),('source_codec_summary.csv',summarize(all_rows)),('payload_ledger.csv',ledger)]:write_rows(folder/name,rows)
    if a.role=='calibration' and not a.preflight:register(EXP/'quality_targets.json',freeze_targets(all_rows,digest(folder/'source_codec_per_image.csv')))
    write_json(folder/'completion.json',{'status':'REAL_SOURCE_CODEC_COMPLETE','sources':limit,'source_rows':len(all_rows),'ledger_rows':len(ledger),'registration_sha256':regsha,'files':{p.name:digest(p) for p in folder.glob('*.csv')},'source_receipts':receipts,'decoder_sha256':state_sha256(decoder),'seconds':time.time()-started,'synthetic':False,'preflight_only':a.preflight})
    print('COMPLETE',a.role,limit,flush=True)

if __name__=='__main__':
    try:main()
    except ResourceBusy as e:print('SAFE_SOURCE_PAUSE',str(e),flush=True);raise SystemExit(75)
