"""Actual v1 PHY and prefix/VAR cache, never reusing an m8 RX as another m."""
import argparse,time
from pathlib import Path
import numpy as np,torch
from latent_enhancement.runtime import save_torch,write_json,digest,model_paths,require_available
from latent_enhancement_b.common import CACHE,scale_statistics,load_decoder,sample_key_digest
from latent_enhancement.latent import complete_latent
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.progressive import split_prefix,prefix_key
from latent_mechanisms.requalify_predictor import write_rows
from .common import OUT,read,configure_runtime,register,identity_files,Safety,tensor_sha
from .protocol import allocation,transmit_digital,decode_frame,standard_noise,config
from .models import prefix_latent

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['train','calibration'],required=True);p.add_argument('--m',type=int,required=True);p.add_argument('--limit',type=int);p.add_argument('--preflight',action='store_true');p.add_argument('--N',type=int,default=4084);p.add_argument('--nd',type=int);a=p.parse_args()
    configure_runtime();require_available();device=torch.device('cuda:0');cfg=config();ledger=allocation(a.m,a.N,a.nd)
    if a.preflight and (a.role!='calibration' or a.limit!=100):raise ValueError('preflight must use first100 calibration only')
    if not a.preflight and a.limit:raise ValueError('formal caches require complete population')
    folder=OUT/('preflight' if a.preflight else 'cache')/f"N{a.N}_m{a.m}_ND{ledger['ND']}"/a.role;folder.mkdir(parents=True,exist_ok=True)
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device)
    scope={'ledger':ledger,'precision':cfg['precision'],'model_paths':model_paths(),'decoder_state_sha256':state_sha256(decoder),'role':a.role,'snrs_db':cfg['snrs_db'],'seeds':cfg['training_base_seeds'] if a.role=='train' else cfg['calibration_seeds'],'noise':cfg['noise'],'allowed_modes':[a.m],'cache_schema':'source/SNR/noise, F/B_TX/B_RX/G_RX float32, actual decoded class/status'}
    registration={'scope':scope,'bindings':identity_files([__file__])};register(folder/'registration.json',registration)
    scale=scale_statistics(device);safety=Safety();hashes={};rows=[];total=0;started=time.time()
    paths=sorted((CACHE/a.role).glob('shard_*.pt'))
    if a.limit:paths=paths[:1]
    for src in paths:
        require_available()
        receipt=read(src.with_suffix('.json'))
        if digest(src)!=receipt['sha256']:raise RuntimeError('source hash mismatch')
        source=torch.load(src,map_location='cpu',weights_only=True);ids=source['image_ids']
        dst=folder/src.name;meta=dst.with_suffix('.json')
        identity={'scope':scope,'source_sha256':receipt['sha256'],'sample_key_sha256':sample_key_digest(ids,scope['snrs_db'],scope['seeds']),'image_ids':ids}
        if meta.exists():
            saved=read(meta)
            if saved['identity']!=identity or digest(dst)!=saved['sha256']:raise RuntimeError('cache identity/file mismatch')
            hashes[dst.name]=saved['sha256'];total+=len(ids);continue
        count=len(ids);shape=(count,len(scope['snrs_db']),len(scope['seeds']))
        record={'image_ids':ids,'F':source['F'],'B_TX':torch.empty_like(source['F']),'B_RX':torch.zeros(*shape,32,16,16),'G_RX':torch.zeros(*shape,32,16,16),'status':torch.zeros(*shape,3),'label':torch.full(shape,1000,dtype=torch.long),'snrs_db':scope['snrs_db'],'noise_seeds':scope['seeds'],'identity':identity}
        for i,image_id in enumerate(ids):
            if i%10==0 and safety.check():raise RuntimeError('sustained thermal throttle: paused at source boundary')
            source_scales=split_prefix(source['full_tokens'][i].numpy(),10)
            ts=[torch.as_tensor(x[None],device=device) for x in source_scales]
            B=prefix_latent(vae,ts,a.m);record['B_TX'][i]=B[0].cpu()
            # Check the exact ten-scale tokenizer on true F, never truncate its schedule.
            if i==0:
                fresh=vae.quantize.f_to_idxBl_or_fhat(source['F'][i:i+1].to(device),to_fhat=False)
                if any(not torch.equal(fresh[k].cpu(),torch.as_tensor(source_scales[k][None])) for k in range(a.m)):raise RuntimeError('source tokens/frozen quantizer mismatch')
                expected=vae.quantize.f_to_idxBl_or_fhat(source['F'][i:i+1].to(device),to_fhat=True)[a.m-1]
                torch.testing.assert_close(B,expected,atol=2e-5,rtol=1e-5)
            digital=transmit_digital(source_scales,int(source['labels'][i]),ledger)
            memo={};true_tokens=np.concatenate(source_scales[:a.m])
            for si,snr in enumerate(scope['snrs_db']):
                for ni,seed in enumerate(scope['seeds']):
                    # Continuous symbols never affect decoding under the singleton fixed-mode receiver.
                    y=np.concatenate((digital+standard_noise(image_id,seed,ledger,'digital')*10**(-snr/20),np.zeros((ledger['NA'],2))))
                    rx=decode_frame(y,snr,a.N,allowed_modes=(a.m,),digital_allocations={a.m:ledger['ND']});pos=(i,si,ni)
                    record['status'][pos]=torch.tensor([rx['header_ok'],rx['body_crc_ok'],rx['decoded_mode'] if rx['header_ok'] else 0])
                    token_error=1.;prefix_error=float(((B/scale[None,:,None,None])**2).mean());base_error=float(((source['F'][i].to(device)/scale[:,None,None])**2).mean())
                    if rx['header_ok']:
                        key=prefix_key(rx['prefix'],rx['decoded_label'])
                        if key not in memo:
                            pfx=[torch.as_tensor(x[None],device=device) for x in rx['prefix']]
                            brx=prefix_latent(vae,pfx,a.m);grx=complete_latent(vae,var,rx['prefix'],rx['decoded_label'],device)
                            memo[key]=(brx[0].cpu(),grx[0].cpu())
                        brx,grx=memo[key];record['B_RX'][pos]=brx;record['G_RX'][pos]=grx;record['label'][pos]=rx['decoded_label']
                        token_error=float(np.mean(np.concatenate(rx['prefix'])!=true_tokens))
                        prefix_error=float((((brx.to(device)-B[0])/scale[:,None,None])**2).mean())
                        base_error=float((((grx.to(device)-source['F'][i].to(device))/scale[:,None,None])**2).mean())
                    if a.preflight:rows.append({'source_index':i,'image_id':image_id,'m':a.m,'N':a.N,'ND':ledger['ND'],'NA':ledger['NA'],'snr_db':snr,'seed':seed,'header_ok':int(rx['header_ok']),'header_crc_ok':int(rx['header_crc_ok']),'decoded_mode':rx['decoded_mode'],'decoded_label':rx['decoded_label'],'body_crc_ok':int(rx['body_crc_ok']),'token_error_rate_including_erasure':token_error,'prefix_error_normalized':prefix_error,'base_error_normalized':base_error})
            if a.preflight and i%10==9:print('preflight',a.m,i+1,flush=True)
        if any(not torch.isfinite(record[k]).all() for k in ('F','B_TX','B_RX','G_RX','status')):raise FloatingPointError('nonfinite cache')
        save_torch(dst,record);sha=digest(dst);write_json(meta,{'identity':identity,'sha256':sha,'sources':count});hashes[dst.name]=sha;total+=count
        write_json(folder/'status.json',{'status':'BUILDING','sources':total,'elapsed_seconds':time.time()-started,'hardware':safety.last});print('cache',a.m,a.role,total,flush=True)
    expected=a.limit or cfg['data'][a.role]
    if total!=expected:raise RuntimeError('incomplete source population')
    if a.preflight and rows:
        write_rows(folder/'per_frame.csv',rows)
        summary={str(s):{k:float(np.mean([r[k] for r in rows if r['snr_db']==s])) for k in ('header_ok','body_crc_ok','token_error_rate_including_erasure','prefix_error_normalized','base_error_normalized')} for s in scope['snrs_db']}
        write_json(folder/'summary.json',summary)
    write_json(folder/'completion.json',{'status':'REAL_PHY_AND_LATENT_CACHE_COMPLETE','sources':total,'frames':total*len(scope['snrs_db'])*len(scope['seeds']),'scope':scope,'hashes':hashes,'registration_sha256':digest(folder/'registration.json'),'seconds':time.time()-started,'synthetic':False})
if __name__=='__main__':main()
