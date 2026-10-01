"""Calibrate K without development leakage; evaluate frozen partial-scale policies."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import common as c
import numpy as np
import torch
import partial_phy as phy
import partial_receiver as rx
from var_comm.study import seeded_noise

METHODS = ['whole','raster','random','entropy','oracle']

def action_dict(a): return dict(N=a.N,phy=a.phy,m=a.m,q=a.q,order=a.order)
def action_key(a): return json.dumps(action_dict(a),sort_keys=True,separators=(',',':'))
def action_id(a): return f'N{a.N}/{a.phy}/m{a.m}/K{a.q}/{a.order}'
def action_from(d): return phy.Action(**{k:d[k] for k in ['N','phy','m','q','order']})
def method_actions(actions, method):
    return [a for a in actions if a.q == 0 or a.order == method] if method != 'whole' else [a for a in actions if a.q == 0]

def received_key(event):
    return rx.received_cache_key(event)

def noise_for(record, N, seed):
    # Same physical real-noise vector across all orders, K and modulations at this budget.
    return seeded_noise(f'{c.RUN}/M1/N{N}|{record["image_id"]}', seed, (N,2))

def tx_positions(loaded, scales, a, memo):
    if a.q == 0: return None
    prefix=scales[:a.m]
    if a.order in ['raster','random']:
        return rx.order_positions(prefix,a.order,a.q)
    if a.m not in memo:
        memo[a.m] = rx.prefix_logits(loaded['vae'],loaded['var'],prefix,loaded['device'])
    return rx.order_positions(prefix,a.order,a.q,logits=memo[a.m],truth_next=scales[a.m] if a.order=='oracle' else None)

def waveform(loaded, scales, a, memo):
    return phy.transmit(scales,a,positions=tx_positions(loaded,scales,a,memo))

def score_one(loaded,d,i,a,snr,seed,txmemo,render_memo,metric_memo,images=False):
    r=d['records'][i];scales=c.split_tokens(d['T'][i].numpy())
    if action_key(a) not in txmemo['waves']:
        txmemo['waves'][action_key(a)]=waveform(loaded,scales,a,txmemo['logits'])
    wave,ledger=txmemo['waves'][action_key(a)]
    y=wave+noise_for(r,a.N,seed)*10.**(-float(snr)/20)
    event=phy.receive(y,snr,a.N,a.phy)
    key=received_key(event)
    if key not in render_memo:
        render_memo[key]=rx.complete_partial(loaded['vae'],loaded['var'],event,loaded['device'])
    result=render_memo[key]
    latent=result['fhat']
    if latent is not None: latent=latent.detach().cpu()
    lh='erasure' if latent is None else hashlib.sha256(latent.numpy().tobytes()).hexdigest()
    if lh not in metric_memo or (images and metric_memo[lh][1] is None):
        mis=None if len(d['records'])!=100 else d['reference'][c.assets.mismatch_permutation()[i]]
        metric_memo[lh]=c.quality(loaded,r,latent,d['F'][i],d['reference'][i],mis,images)
    metrics,image=metric_memo[lh]
    row=dict(ledger)
    row.update(source_id=r['image_id'],source_index=i,preprocessing_id=r['preprocessing_id'],
        N=a.N,phy_family=a.phy,snr_db=snr,noise_seed=seed,action_id=action_id(a),m=a.m,q=a.q,order=a.order,
        next_scale_len=phy.SIZES[a.m]**2,**metrics)
    if len(d['records'])==100:row['mismatch_source_id']=d['records'][c.assets.mismatch_permutation()[i]]['image_id']
    row.update(phy.event_fields(event))
    row['partial_used']=bool(result['diagnostics'].get('partial_used',False))
    row['header_exact']=bool(event['header_ok'] and event['decoded_m']==a.m and event['decoded_q']==a.q and event['decoded_order']==a.order)
    # Truth enters offline diagnostics only, after the receiver has returned.
    rp=event.get('prefix',[])
    row['prefix_exact']=bool(row['header_ok'] and len(rp)==a.m and all(np.array_equal(x,y) for x,y in zip(rp,scales[:a.m])))
    row['prefix_false_accept']=bool(row['prefix_crc_ok'] and not row['prefix_exact'])
    row['order_agreement']=''
    row['received_partial_exact']=''
    row['known_tokens_fixed']=bool(result['diagnostics']['known_tokens_fixed'])
    if row['partial_used'] and row['header_exact'] and a.q>0:
        tp=tx_positions(loaded,scales,a,txmemo['logits'])
        rp=np.asarray(result['diagnostics']['known_positions'],dtype=np.int64)
        row['order_agreement']=bool(np.array_equal(tp,rp))
        row['received_partial_exact']=bool(row['order_agreement'] and np.array_equal(np.asarray(result['diagnostics']['known_values']),scales[a.m][rp]))
        row['tx_position_sha256']=hashlib.sha256(np.asarray(tp,dtype='<u2').tobytes()).hexdigest()
        row['rx_position_sha256']=hashlib.sha256(np.asarray(rp,dtype='<u2').tobytes()).hexdigest()
    else:
        row['tx_position_sha256']='';row['rx_position_sha256']=''
    row['waveform_sha256']=hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
    row['observation_sha256']=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
    return row,image

def summarize(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['N'],r['phy_family'],r['snr_db'],r['action_id'])].append(r)
    out=[]
    for key,rr in sorted(groups.items()):
        a=rr[0];z={k:a[k] for k in ['N','phy_family','snr_db','action_id','m','q','order']}
        for k in ['psnr_db','lpips_alex','dino_cosine','E']:
            z[k]=float(np.mean([float(r[k]) for r in rr]))
        z['failure_fraction']=float(np.mean([not r['header_ok'] or not r['body_crc_ok'] for r in rr]))
        z['rows']=len(rr);out.append(z)
    return out

def rank(group):
    # Same registered selection principle as earlier digital work; K0 is always in the candidate set.
    ref=min(group,key=lambda r:(r['failure_fraction'],-r['m'],-r['q'],r['action_id']))
    feasible=[r for r in group if r['psnr_db']>=ref['psnr_db']-.25]
    return sorted(feasible,key=lambda r:(r['lpips_alex'],r['failure_fraction'],r['action_id'])),ref

def legal_grid():
    return [a for N in [512,1024] for fam in phy.PHY_FAMILIES for a in phy.action_grid(N,fam)]

def calibration(loaded,screen):
    stage='m1_screen' if screen else 'm1_calibration'
    d=c.data('calibration',loaded)
    inputs=[] if screen else [c.RESULT/'m1_shortlist.json',c.OUT/'m1_screen_complete.json']
    if not screen and c.read(c.OUT/'m1_screen_complete.json')['policy_sha256']!=c.sha(c.RESULT/'m1_shortlist.json'):raise RuntimeError('Screen shortlist hash changed')
    identity=c.registration(loaded,d,stage,inputs)
    grid=legal_grid();mapping={action_id(a):a for a in grid}
    if screen:
        choices={(N,f,s):[a for a in grid if a.N==N and a.phy==f] for N in [512,1024] for f in phy.PHY_FAMILIES for s in c.SNRS}
        count=200;seeds=[4101]
    else:
        shortlist=c.read(c.RESULT/'m1_shortlist.json')
        if shortlist['development_read']: raise RuntimeError('Screen read development')
        choices={}
        for cell in shortlist['cells']:
            key=(cell['N'],cell['phy_family'],cell['snr_db'])
            choices.setdefault(key,{})
            for aid in cell['action_ids']: choices[key][aid]=mapping[aid]
        choices={k:list(v.values()) for k,v in choices.items()};count=1000;seeds=c.CAL_SEEDS
    sums=defaultdict(lambda:dict(n=0,psnr_db=0.,lpips_alex=0.,dino_cosine=0.,E=0.,failure_fraction=0.))
    first={};written=0;start=time.time()
    for i in range(count):
        c.check();p=c.OUT/stage/'cells'/f'{i:04d}.json'
        if p.exists():
            saved=c.read(p)
            if saved['identity']!=identity: raise RuntimeError('Calibration resume binding mismatch')
            rows=saved['rows']
        else:
            rows=[];tm=dict(waves={},logits={});rm={};mm={}
            for (N,f,snr),acts in choices.items():
                for a in acts:
                    for seed in seeds:
                        c.check();r,_=score_one(loaded,d,i,a,snr,seed,tm,rm,mm)
                        rows.append(r)
            c.seal(p,dict(identity=identity,rows=rows))
        for r in rows:
            key=(r['N'],r['phy_family'],r['snr_db'],r['action_id']);first[key]=r
            v=sums[key];v['n']+=1
            for met in ['psnr_db','lpips_alex','dino_cosine','E']: v[met]+=float(r[met])
            v['failure_fraction']+=float(not r['header_ok'] or not r['body_crc_ok'])
        written+=len(rows)
        c.status(stage,sources=i+1,total=count,rows=written,elapsed_seconds=time.time()-start)
    summary=[]
    for key,v in sorted(sums.items()):
        r=first[key];z={k:r[k] for k in ['N','phy_family','snr_db','action_id','m','q','order']}
        z.update({k:val/v['n'] for k,val in v.items() if k!='n'});z['rows']=v['n'];summary.append(z)
    c.csv_rows(c.RESULT/(stage+'_summary.csv'),summary)
    cells=[]
    for N in [512,1024]:
        for f in phy.PHY_FAMILIES:
            for snr in c.SNRS:
                for method in METHODS:
                    group=[r for r in summary if (r['N'],r['phy_family'],r['snr_db'])==(N,f,snr) and (r['q']==0 or r['order']==method) and (method!='whole' or r['q']==0)]
                    if not screen:
                        allowed=next(x['action_ids'] for x in shortlist['cells'] if (x['N'],x['phy_family'],x['snr_db'],x['method'])==(N,f,snr,method))
                        group=[r for r in group if r['action_id'] in allowed]
                    ranked,ref=rank(group)
                    if screen:
                        whole_ranked,_=rank([r for r in group if r['q']==0])
                        aids=sorted({r['action_id'] for r in ranked[:2]}|{whole_ranked[0]['action_id']})
                        cells.append(dict(N=N,phy_family=f,snr_db=snr,method=method,action_ids=aids,reliability_reference=ref))
                    else:
                        winner=ranked[0]
                        cells.append(dict(N=N,phy_family=f,snr_db=snr,method=method,action=action_dict(mapping[winner['action_id']]),calibration=winner,reliability_reference=ref))
    dest='m1_shortlist.json' if screen else 'm1_policy.json'
    c.seal(c.RESULT/dest,dict(development_read=False,cells=cells,registration_sha256=identity,sources=count,noise_seeds=seeds))
    c.assert_frozen(loaded)
    c.write(c.OUT/(stage+'_complete.json'),dict(status='COMPLETE',sources=count,rows=written,development_read=False,policy_sha256=c.sha(c.RESULT/dest),training_updates=0))

def import_references(d,loaded):
    rows=[];bindings={}
    for N,date in [(512,'20260930'),(1024,'20261001')]:
        folder=c.ROOT/(f'results/extreme_bandwidth_{date}_R1' if N==512 else f'results/extreme_bandwidth_{date}_R1_N{N}')
        path=folder/'per_frame.csv';cfg=c.read(folder/'config.json')
        # The final study identity is nested; enforce exact model hash occurrence and exact row source/preprocess keys.
        encoded=json.dumps(cfg,sort_keys=True)
        for name,h in loaded['identity']['models'].items():
            if h not in encoded: raise RuntimeError('Historical frozen asset incompatible: '+name)
        permutations=[]
        def collect(obj):
            if isinstance(obj,dict):
                for key,val in obj.items():
                    if key=='mismatch_permutation':permutations.append(val)
                    else:collect(val)
            elif isinstance(obj,list):
                for val in obj:collect(val)
        collect(cfg)
        if not permutations or any(x!=c.assets.mismatch_permutation() for x in permutations):raise RuntimeError('Historical DINO mismatch reference differs')
        index={r['image_id']:r['preprocessing_id'] for r in d['records']}
        for r in c.read_csv(path):
            if r['method'] not in [f'P{N}','D_U_QPSK','D_U_16QAM']:continue
            if r['source_id'] not in index or r['preprocessing_id']!=index[r['source_id']]:raise RuntimeError('Historical image identities differ')
            targets=['QPSK','16QAM'] if r['method']==f'P{N}' else [r['method'].removeprefix('D_U_')]
            for fam in targets:
                row=dict(r);row.update(N=N,phy_family=fam,method=f'P{N}' if r['method']==f'P{N}' else 'legacy_policy',historical_reference=True)
                idx=int(row['source_index'])
                if not 0<=idx<len(d['records']) or d['records'][idx]['image_id']!=row['source_id']:raise RuntimeError('Historical source index differs')
                row['mismatch_source_id']=d['records'][c.assets.mismatch_permutation()[idx]]['image_id']
                rows.append(row)
        bindings[str(path)]=c.sha(path)
    if len(rows)!=12000:raise RuntimeError('Historical P/digital reference rows incomplete')
    c.write(c.RESULT/'m1_reference_identity.json',dict(bindings=bindings,rows=len(rows),P_twice_for_plot_facets=True))
    return rows

def evaluate(loaded):
    policy=c.read(c.RESULT/'m1_policy.json');done=c.read(c.OUT/'m1_calibration_complete.json')
    if done['policy_sha256']!=c.sha(c.RESULT/'m1_policy.json') or policy['development_read']:raise RuntimeError('Unfrozen policy')
    d=c.data('development',loaded);identity=c.registration(loaded,d,'m1_development',[c.RESULT/'m1_policy.json',c.OUT/'m1_calibration_complete.json']);allrows=[];start=time.time()
    for i,r in enumerate(d['records']):
        c.check();p=c.OUT/'m1_development/cells'/f'{i:03d}.json'
        if p.exists():
            cell=c.read(p)
            if cell['identity']!=identity:raise RuntimeError('Development resume mismatch')
            allrows.extend(cell['rows']);continue
        rows=[];tm=dict(waves={},logits={});rm={};mm={}
        for N in [512,1024]:
            for fam in phy.PHY_FAMILIES:
                for snr in c.SNRS:
                    choices={x['method']+'_policy':action_from(x['action']) for x in policy['cells'] if (x['N'],x['phy_family'],x['snr_db'])==(N,fam,snr)}
                    ent=choices['entropy_policy']
                    for order in ['raster','random','oracle']:
                        try: a=phy.Action(N,fam,ent.m,ent.q,order)
                        except ValueError:continue
                        choices[order+'_at_entropy']=a
                    for seed in c.DEV_SEEDS:
                        save=i in [0,25,50,75] and snr in [4,13] and seed==2001
                        for method,a in choices.items():
                            row,image=score_one(loaded,d,i,a,snr,seed,tm,rm,mm,save)
                            row.update(method=method,historical_reference=False);rows.append(row)
                            if save:c.save_image(c.RESULT/'examples/m1'/f'N{N}_{fam}_s{snr}_src{i:03d}_{method}.png',image)
        c.seal(p,dict(identity=identity,rows=rows));allrows.extend(rows)
        c.status('m1_development',sources=i+1,total=100,rows=len(allrows),elapsed_seconds=time.time()-start)
    allrows.extend(import_references(d,loaded));c.csv_rows(c.RESULT/'m1_per_frame.csv',allrows)
    c.assert_frozen(loaded)
    c.write(c.OUT/'m1_development_complete.json',dict(status='COMPLETE',sources=100,rows=len(allrows),policy_sha256=c.sha(c.RESULT/'m1_policy.json'),training_updates=0))

def rate_curves(loaded):
    # Dense source-only diagnostic, not paid noisy-channel performance and not policy tuning.
    d=c.data('development',loaded);c.read(c.OUT/'m1_development_complete.json');identity=c.registration(loaded,d,'m1_rate_curve',[c.OUT/'m1_development_complete.json'])
    rows=[]
    for i,r in enumerate(d['records']):
        p=c.OUT/'m1_rate_curve/cells'/f'{i:03d}.json'
        if p.exists():
            saved=c.read(p)
            if saved['identity']!=identity:raise RuntimeError('Rate-curve identity')
            rows.extend(saved['rows']);continue
        scales=c.split_tokens(d['T'][i].numpy());part=[]
        for m in [5,6]:
            logits=rx.prefix_logits(loaded['vae'],loaded['var'],scales[:m],loaded['device'])
            L=phy.SIZES[m]**2
            for q in sorted(set(range(0,L,4))|{L}):
                for order in ['raster','random','entropy','oracle']:
                    if q==L:
                        latent=c.legacy.complete_latent(loaded['vae'],loaded['var'],scales[:m+1],1000,loaded['device']).cpu()
                    elif q==0:
                        latent=c.legacy.complete_latent(loaded['vae'],loaded['var'],scales[:m],1000,loaded['device']).cpu()
                    else:
                        a=phy.Action(1024,'16QAM',m,q,order)
                        positions=rx.order_positions(scales[:m],order,q,logits=logits,truth_next=scales[m] if order=='oracle' else None)
                        wave,_=phy.transmit(scales,a,positions=positions)
                        event=phy.receive(wave,100.,1024,'16QAM')
                        if not event['header_ok'] or not event['prefix_crc_ok'] or not event['partial_usable']:
                            raise RuntimeError('Source-only curve PHY roundtrip failed')
                        if not all(np.array_equal(x,y) for x,y in zip(event['prefix'],scales[:m])) or not np.array_equal(event['partial_values'],scales[m][positions]):
                            raise RuntimeError('Source-only curve silently changed source tokens')
                        received=rx.complete_partial(loaded['vae'],loaded['var'],event,loaded['device'])
                        if received['diagnostics']['known_positions']!=positions.tolist():raise RuntimeError('Source-only curve order mismatch')
                        latent=received['fhat'].cpu()
                    metrics,_=c.quality(loaded,r,latent,d['F'][i],d['reference'][i],d['reference'][c.assets.mismatch_permutation()[i]])
                    part.append(dict(source_id=r['image_id'],source_index=i,m=m,q=q,order=order,source_bits=12*(sum(p*p for p in phy.SIZES[:m])+q),mask_bits=L if order=='oracle' and 0<q<L else 0,wireless_claim=False,**metrics))
        c.seal(p,dict(identity=identity,rows=part));rows.extend(part);c.status('m1_rate_curve',sources=i+1,total=100)
    c.csv_rows(c.RESULT/'m1_rate_curve.csv',rows);c.write(c.OUT/'m1_rate_curve_complete.json',dict(status='COMPLETE',rows=len(rows),wireless_claim=False))

def timing(loaded):
    d=c.data('development',loaded);policy=c.read(c.RESULT/'m1_policy.json');rows=[]
    # Warm the complete physical receive path and GPU kernels before timing.
    for cell in policy['cells']:
        a=action_from(cell['action']);snr=cell['snr_db'];r=d['records'][0]
        for _ in range(3):
            wave,_=waveform(loaded,c.split_tokens(d['T'][0].numpy()),a,{})
            y=wave+noise_for(r,a.N,2001)*10.**(-snr/20)
            result=rx.complete_partial(loaded['vae'],loaded['var'],phy.receive(y,snr,a.N,a.phy),loaded['device'])
            if result['fhat'] is not None:loaded['decoder'](result['fhat'].to(loaded['device'])).cpu()
    torch.cuda.synchronize()
    for i in [0,11,22,33,44,55,66,77,88,99]:
        r=d['records'][i]
        for cell in policy['cells']:
            a=action_from(cell['action']);snr=cell['snr_db']
            c.check();torch.cuda.synchronize();t=time.perf_counter()
            x=torch.as_tensor(r['pixels'][None],dtype=torch.float32,device=loaded['device'])/127.5-1
            f=loaded['vae'].quant_conv(loaded['vae'].encoder(x))
            tok=torch.cat(loaded['vae'].quantize.f_to_idxBl_or_fhat(f,to_fhat=False),1)[0].cpu().numpy()
            wave,ledger=waveform(loaded,c.split_tokens(tok),a,{})
            torch.cuda.synchronize();tx=(time.perf_counter()-t)*1000
            if not np.array_equal(tok,d['T'][i].numpy()):raise RuntimeError('Timed online TX differs from registered cached tokens')
            y=wave+noise_for(r,a.N,2001)*10.**(-snr/20)
            torch.cuda.synchronize();t=time.perf_counter()
            event=phy.receive(y,snr,a.N,a.phy);res=rx.complete_partial(loaded['vae'],loaded['var'],event,loaded['device'])
            latent=res['fhat'];out=np.full((3,256,256),.5,np.float32) if latent is None else loaded['decoder'](latent.to(loaded['device']))[0].cpu().numpy()
            torch.cuda.synchronize();rt=(time.perf_counter()-t)*1000
            row=dict(ledger);row.update(source_id=r['image_id'],source_index=i,N=a.N,phy_family=a.phy,snr_db=snr,method=cell['method']+'_policy',action_id=action_id(a),tx_ms=tx,rx_ms=rt,cache_used=False,online_cached_tokens_equal=True,warmup_calls_per_policy=3,training_updates=0)
            rows.append(row)
        c.status('m1_timing',sources=i+1,total=100,rows=len(rows))
    c.csv_rows(c.RESULT/'m1_timing.csv',rows);c.write(c.OUT/'m1_timing_complete.json',dict(status='COMPLETE',rows=len(rows),cache_used=False))

def main(stage):
    loaded=c.setup()
    with torch.inference_mode():
        if stage=='screen':calibration(loaded,True)
        elif stage=='calibration':calibration(loaded,False)
        elif stage=='development':evaluate(loaded)
        elif stage=='rate_curves':rate_curves(loaded)
        elif stage=='timing':timing(loaded)
        else:raise ValueError(stage)
    c.assert_frozen(loaded)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);args=p.parse_args()
    try:main(args.stage)
    except c.assets.old.b.ResourceBusy as error:
        c.write(c.OUT/'resource_wait.json',dict(stage=args.stage,error=str(error),time=time.time()));raise SystemExit(75)
