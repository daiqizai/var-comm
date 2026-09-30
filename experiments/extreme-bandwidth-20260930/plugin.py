"""Frozen selected P512 evaluation for the extreme bandwidth N512 study. Calibration precedes development."""
import os,sys,json,time,hashlib,csv,traceback,argparse,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'experiments/rx-posterior-step1-20260929'
sys.path.insert(0,str(OLD))
import run_preflight as environment
import rx_v3_common as old
import importlib.util
def load_local_module(name,filename):
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=(HERE/filename).resolve():
            raise RuntimeError('Local module name collision: '+name)
        return module
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
rx=load_local_module('extreme_bw_frozen_receiver','receiver.py')
import numpy as np
import torch
from torch.nn import functional as fn
from token_efficiency.execution import Cell,apply_channel,execute,waveform_sha
from token_efficiency.digital_grid import population
from latent_enhancement_b.common import scale_statistics
from var_comm.quality import dino_features
from PIL import Image,ImageDraw

RUN_ROOT=ROOT/'outputs/EXTREME-BW-20260930-R1'
OUT=RUN_ROOT/'plugin'
SELECTED_PATH=RUN_ROOT/'training/p512_2026093001/selected_P512.json'
SELECTED_RECORD=None
RESULT=ROOT/'results/extreme_bandwidth_20260930_R1'
SNRS=[1,4,7,13,19]
GRID=[0.,.25,.5,.75,1.,1.5]
CAL_SEEDS=[4101,4102,4103]
DEV_SEEDS=[2001,2002,2003]
TIMING=[0,11,22,33,44,55,66,77,88,99]
EXAMPLES=[0,25,50,75]
BATCH=4
# The legacy Cell registration only accepts historical budgets. Its shared
# apply_channel/execute functions accept this explicit N512 continuous cell.
from dataclasses import dataclass
@dataclass(frozen=True)
class ContinuousCell:
    family: str = 'continuous'
    N: int = 512
    m: object = None
    mcs: str = 'QPSK'
    @property
    def name(self): return 'P512'
CELL=ContinuousCell()

def sha(p):return old.sha(p)
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):old.b.write_json(p,obj)
def status(stage,**kw):
    write(OUT/'status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
def check():old.b.boundary()
def seal(p,obj):
    if p.exists():
        assert read(p)==obj,('immutable identity changed',str(p))
    else:write(p,obj)
def save_tensor(p,obj):
    temporary=p.with_suffix('.tmp');torch.save(obj,temporary);os.replace(temporary,p)
    write(Path(str(p)+'.json'),dict(sha256=sha(p)))
def load_tensor(p):
    assert sha(p)==read(Path(str(p)+'.json'))['sha256']
    return torch.load(p,map_location='cpu',weights_only=True)
def csv_rows(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    rows=[{('lambda' if k=='lambda_' else k):v for k,v in row.items()} for row in rows]
    keys=list(dict.fromkeys(k for row in rows for k in row))
    temporary=p.with_name(p.name+'.tmp')
    with temporary.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    os.replace(temporary,p)
def dump_hashes(models):return {k:old.b.state_sha256(v) for k,v in models.items()}
def batches(n,size=BATCH):
    for i in range(0,n,size):yield slice(i,min(n,i+size))
def mom(e):
    e=e.double();return dict(mean=e.mean((0,2,3)).cpu().tolist(),
        variance=e.var((0,2,3),correction=0).cpu().tolist(),
        mse=e.square().mean((0,2,3)).cpu().tolist())

def setup():
    global SELECTED_RECORD
    load_for_evaluation=load_local_module('_extreme_bw_train','train.py').load_for_evaluation
    SELECTED_RECORD=read(SELECTED_PATH)
    assert SELECTED_RECORD['N']==512
    load_assets=load_local_module('extreme_bw_visual_assets','assets.py').setup
    loaded=load_assets()
    vae,var,stats,static=loaded['vae'],loaded['var'],loaded['stats'],loaded['static']
    dec=loaded['decoder']
    p,meta=load_for_evaluation(SELECTED_PATH,scale_statistics(),torch.device('cuda:0'))
    registration=meta.get('registration')
    if registration is None:
        registration=read(SELECTED_PATH.parent/'registration.json')
    assert old.b.state_sha256(dec)==registration['decoder_state_sha256']
    assert p.uses==512 and SELECTED_RECORD['step']<=40000
    lp,dino,qmeta=loaded['lpips'],loaded['dino'],loaded['identity']['quality']
    models=dict(vae=vae,var=var,decoder=dec,P512=p,lpips=lp,dino=dino)
    for model in models.values():model.eval().requires_grad_(False)
    identity=dict(models=dump_hashes(models),base=meta,selected=SELECTED_RECORD,
      selected_path=str(SELECTED_PATH),selected_sha256=sha(SELECTED_PATH),quality=qmeta,
      stats_sha256=sha(old.b.OUT/'calibration_statistics.pt'))
    return models,stats,static,identity

def base_model_id():
    assert SELECTED_RECORD is not None
    return 'P512_step'+str(SELECTED_RECORD['step'])+'_'+SELECTED_RECORD['checkpoint_sha256'][:12]

def calibration_sources(stats):
    F,Fq,T,ids,bindings=old.b.load_calibration()
    records,pixelbindings=population('calibration')
    assert ids==[r['image_id'] for r in records] and ids[:200]==stats['subset_ids']
    return F[:200],Fq[:200],T[:200],records[:200],dict(latents=bindings,pixels=pixelbindings)

@torch.no_grad()
def selfcheck(models,stats,static,F,Fq,T,records):
    vae,var,dec,p=models['vae'],models['var'],models['decoder'],models['P512']
    f=F[:1].cuda();truth=old.b.split(T[:1].cuda());z=f+.01
    q=rx.infer(vae,var,f,'A1',[0.]*10,static,lam=0.,details=False)['fhat']
    assert torch.equal(q,Fq[:1].cuda()),'clean official quantization parity'
    a1=rx.infer(vae,var,z,'A1',[.1]*10,static,lam=0.,details=False)
    for prior in ['A2','V']:
        other=rx.infer(vae,var,z,prior,[.1]*10,static,lam=0.,details=False)
        assert torch.equal(other['fhat'],a1['fhat'])
        assert all(torch.equal(x,y) for x,y in zip(a1['tokens'],other['tokens']))
    tests=[]
    for i in range(2):
        record=records[i];f=F[i:i+1].cuda()
        encoded=vae.quant_conv(vae.encoder(torch.as_tensor(record['pixels'][None],device='cuda',dtype=torch.float32)/127.5-1))
        np.testing.assert_allclose(encoded.cpu(),f.cpu(),rtol=1e-5,atol=2e-5)
        for snr in SNRS:
            wave=p.transmit(f)[0].cpu().numpy();y=apply_channel(wave,snr,record['image_id'],4101,CELL)
            z=p.receive(torch.tensor(y[None],device='cuda'),torch.tensor([snr],device='cuda',dtype=torch.float32))
            image=dec(z)[0].cpu().numpy()
            native,info=execute(record,CELL,snr,4101,vae,var,dec,'cuda',p)
            np.testing.assert_allclose(image,native,atol=2e-5,rtol=1e-5)
            np.testing.assert_allclose(wave,info['waveform'],atol=2e-5,rtol=1e-5)
            assert rx.bypass(z) is z and torch.equal(dec(rx.bypass(z)),dec(z))
            tests.append(dict(source_id=record['image_id'],snr_db=snr,
                P512_native_max_abs=float(np.abs(image-native).max()),actual_E=float(np.square(wave,dtype=np.float64).sum())))
    # Scoring truth is isolated: changing it cannot change RX tokens or the cumulative estimate.
    truth=old.b.split(T[i:i+1].cuda())
    ref=rx.infer(vae,var,z,'V',[.1]*10,static,lam=1.,clean=f,truth=truth)
    alternative_truth=[(x+17)%4096 for x in truth]
    alt=rx.infer(vae,var,z,'V',[.1]*10,static,lam=1.,clean=f+.123,truth=alternative_truth)
    assert torch.equal(ref['fhat'],alt['fhat']) and all(torch.equal(x,y) for x,y in zip(ref['tokens'],alt['tokens']))
    result=dict(passed=True,selected_sha256=sha(SELECTED_PATH),P512_original_chain=tests,clean_lambda0_official=True,
      A2_V_lambda0_equals_A1=True,true_bypass_exact=True,scoring_branch_cannot_change_RX=True)
    seal(OUT/'selfcheck.json',result)
    return result

@torch.no_grad()
def observations(F,records,seeds,p,role):
    path=OUT/(role+'_observations.pt')
    expected=dict(config_sha256=sha(RESULT/'plugin_config.json'),role=role,seeds=seeds,
      source_ids=[r['image_id'] for r in records],preprocessing_ids=[r['preprocessing_id'] for r in records])
    if path.exists():
        data=load_tensor(path);assert data['identity']==expected
        csv_rows(RESULT/(role+'_channel_ledger.csv'),data['rows']);return data
    Z=[];truth_indices=[];metadata=[]
    for i,record in enumerate(records):
        check();wave=p.transmit(F[i:i+1].cuda())[0].cpu().numpy()
        E=float(np.square(wave,dtype=np.float64).sum())
        assert wave.shape==(512,2) and np.isclose(E,1024,atol=.02,rtol=1e-5)
        for snr in SNRS:
            for seed in seeds:
                y=apply_channel(wave,snr,record['image_id'],seed,CELL)
                z=p.receive(torch.tensor(y[None],device='cuda'),torch.tensor([snr],dtype=torch.float32,device='cuda'))
                Z.append(z[0].cpu());truth_indices.append(i)
                metadata.append(dict(source_id=record['image_id'],source_index=i,snr_db=snr,
                  noise_seed=seed,N=512,E=E,waveform_sha256=waveform_sha(wave),observation_sha256=waveform_sha(y),
                  preprocessing_id=record['preprocessing_id']))
        status(role+'_observations',sources=i+1,total=len(records))
    data=dict(Z=torch.stack(Z),source_indices=torch.tensor(truth_indices),rows=metadata,identity=expected)
    save_tensor(path,data);csv_rows(RESULT/(role+'_channel_ledger.csv'),metadata)
    return data

def error_statistics(F,observed,stats):
    output={};rows=[]
    for snr in SNRS:
        indices=[i for i,r in enumerate(observed['rows']) if r['snr_db']==snr]
        ii=observed['source_indices'][indices];e=observed['Z'][indices].double()-F[ii].double()
        zs=(e/stats['sigma'].double());tau=mom(zs)
        v=[]
        for k,pn in enumerate(old.b.PATCH_NUMS):
            ek=fn.interpolate(e,size=(pn,pn),mode='area') if k<9 else e
            vk=float(ek.var(correction=0));v.append(max(0.,vk));channel=mom(ek)
            rows.append(dict(snr_db=snr,type='P_error_scale',scale=k+1,channel='all',
              mean=float(ek.mean()),variance=vk,mse=float(ek.square().mean()),count=ek.numel()))
            for c in range(32):rows.append(dict(snr_db=snr,type='P_error_scale',scale=k+1,channel=c,
                mean=channel['mean'][c],variance=channel['variance'][c],mse=channel['mse'][c],count=ek[:,c].numel()))
        for c in range(32):rows.append(dict(snr_db=snr,type='P_error_standardized',scale='all',channel=c,
          mean=tau['mean'][c],variance=tau['variance'][c],mse=tau['mse'][c],count=zs[:,c].numel()))
        output[str(snr)]=dict(variance_by_scale=v,tau_variance=tau['variance'],tau_statistics=tau,indices=indices)
    if not (RESULT/'plugin_selected_policy.json').exists():csv_rows(RESULT/'plugin_error_stats.csv',rows)
    return output,rows

@torch.no_grad()
def images_metrics(Z,source_indices,records,models,dino_reference=None,mismatch=None,save_images=False):
    allrows=[];images=[];features=[]
    for sl in batches(len(Z)):
        check();target=torch.tensor(np.stack([records[int(i)]['pixels'] for i in source_indices[sl]]),dtype=torch.float32,device='cuda')/255.
        reconstructed=models['decoder'](Z[sl].cuda())
        psnr=-10*torch.log10((target-reconstructed).square().flatten(1).mean(1))
        lp=models['lpips'](target*2-1,reconstructed*2-1).flatten()
        for j in range(len(target)):
            allrows.append(dict(psnr_db=float(psnr[j]),lpips_alex=float(lp[j])))
        if dino_reference is not None:
            embedding=dino_features(models['dino'],reconstructed)
            ii=source_indices[sl].long().cuda();cos=fn.cosine_similarity(embedding,dino_reference[ii],dim=1)
            mism=fn.cosine_similarity(embedding,dino_reference[mismatch[ii.cpu()].cuda()],dim=1)
            for j in range(len(target)):allrows[sl.start+j].update(dino_cosine=float(cos[j]),dino_mismatched=float(mism[j]))
        if save_images:images.extend(reconstructed.cpu())
    return allrows,images

def candidate_key(prior,lam):return 'A1' if lam==0 or prior=='A1' else prior+'_lambda'+str(lam)
def candidate_list():return [('A1',0.)]+[(p,l) for p in ['A2','V'] for l in GRID if l>0]

@torch.no_grad()
def calibrate(F,T,records,obs,stats,static,models,error,errrows,config):
    selected=RESULT/'plugin_selected_policy.json'
    if selected.exists():
        policy=read(selected);assert policy['config_sha256']==sha(RESULT/'plugin_config.json')
        return policy
    vae,var=models['vae'],models['var'];candidates=[];levels={}
    for snr in SNRS:
        e=error[str(snr)];ind=e['indices'];ii=obs['source_indices'][ind];Z=obs['Z'][ind];clean=F[ii]
        tau=np.asarray(e['tau_variance']);base,_=images_metrics(Z,ii,records,models)
        basemean={m:float(np.mean([r[m] for r in base])) for m in ['psnr_db','lpips_alex']}
        csv_rows(OUT/f'calibration_P512_{snr}.csv',base)
        candidates.append(dict(snr_db=snr,method='P512',lambda_='',policy_action='BYPASS',feasible=True,**basemean))
        level={k:v for k,v in e.items() if k!='indices'};level['methods']={};data={}
        level['psnr_drop_cap_db']=.2 if snr==13 else .3
        for prior,lam in candidate_list():
            check();key=candidate_key(prior,lam);path=OUT/'calibration_candidates'/f'{snr}_{key}.pt';path.parent.mkdir(exist_ok=True)
            if path.exists():Q=load_tensor(path)['Fq']
            else:
                estimates=[]
                for sl in batches(len(Z)):
                    check();result=rx.infer(vae,var,Z[sl].cuda(),prior,e['variance_by_scale'],static,lam=lam,details=False)
                    estimates.append(result['fhat'].cpu());status('calibration_candidates',snr_db=snr,candidate=key,frames=sl.stop,total=len(Z))
                Q=torch.cat(estimates);save_tensor(path,dict(Fq=Q))
            ep=(Z.double()-clean.double())/stats['sigma'].double()
            eq=(clean.double()-Q.double())/stats['sigma'].double();rho=mom(eq);r=np.asarray(rho['variance'])
            denominator=tau+r;alpha=np.divide(tau,denominator,out=np.zeros(32),where=denominator>0)
            fused=rx.fusion(Z,Q,torch.tensor(alpha,dtype=torch.float32).reshape(1,32,1,1))
            metrics,_=images_metrics(fused,ii,records,models)
            mean={m:float(np.mean([r[m] for r in metrics])) for m in ['psnr_db','lpips_alex']}
            # Every source has exactly three paired noises, so this is source-first averaging algebraically.
            cap=.2 if snr==13 else .3;feasible=mean['psnr_db']-basemean['psnr_db']>=-cap
            cross=(ep*eq).mean((0,2,3)).tolist()
            for c in range(32):errrows.append(dict(snr_db=snr,type='candidate_standardized',scale='all',channel=c,
                method=prior,lambda_=lam,mean=rho['mean'][c],variance=rho['variance'][c],mse=rho['mse'][c],cross_second_moment=cross[c],count=eq[:,c].numel()))
            data[key]=dict(lambda_=lam,raw_feasible=feasible,rho=rho['variance'],rho_statistics=rho,alpha=alpha.tolist(),
                calibration=mean,cross_second_moment=cross,cache=str(path))
            candidates.append(dict(snr_db=snr,method=prior,lambda_=lam,policy_action='CORRECT',feasible=feasible,**mean))
            csv_rows(RESULT/'plugin_calibration_candidates.csv',candidates);csv_rows(RESULT/'plugin_error_stats.csv',errrows)
        for prior in ['A1','A2','V']:
            choices=[data[candidate_key(prior,l)] for l in ([0.] if prior=='A1' else GRID)]
            if prior!='A1':
                zero={**data['A1']};candidates.append(dict(snr_db=snr,method=prior,lambda_=0.,policy_action='CORRECT',feasible=zero['raw_feasible'],**zero['calibration']))
            feasible=[d for d in choices if d['raw_feasible']]
            chosen=min(feasible or choices,key=lambda d:(d['calibration']['lpips_alex'],d['lambda_']))
            action='CORRECT' if feasible and chosen['calibration']['lpips_alex']<basemean['lpips_alex'] else 'BYPASS'
            level['methods'][prior]={**chosen,'raw_selected_lambda':chosen['lambda_'],'policy_action':action,
                'raw_feasible':bool(feasible),'diagnostic':not bool(feasible),
                'calibration_psnr_delta_db':chosen['calibration']['psnr_db']-basemean['psnr_db'],
                'calibration_psnr_drop_cap_db':level['psnr_drop_cap_db']}
            candidates.append(dict(snr_db=snr,method=prior,lambda_='',policy_action='BYPASS',feasible=True,**basemean))
        level['alpha_common']=data['A1']['alpha'];level['V_lambda1']=data['V_lambda1.0'];level['P512']=basemean;levels[str(snr)]=level
        status('calibration_selection',snr_db=snr,complete_levels=len(levels))
    csv_rows(RESULT/'plugin_calibration_candidates.csv',candidates)
    policy=dict(config_sha256=sha(RESULT/'plugin_config.json'),development_read=False,levels=levels)
    seal(selected,policy);seal(OUT/'calibration_complete.json',dict(policy_sha256=sha(selected),config_sha256=sha(RESULT/'plugin_config.json'),development_read=False))
    return policy

def render_example(path,source,images):
    path.parent.mkdir(parents=True,exist_ok=True);grid=Image.new('RGB',(256*5,276),'white');draw=ImageDraw.Draw(grid)
    for i,(name,image) in enumerate([('Source',source),*images]):
        pixels=np.clip(np.asarray(image)*255,0,255).astype(np.uint8).transpose(1,2,0)
        grid.paste(Image.fromarray(pixels),(i*256,20));draw.text((i*256+5,3),name,fill='black')
    grid.save(path)

def save_image(path,image):
    path.parent.mkdir(parents=True,exist_ok=True)
    pixels=np.clip(np.asarray(image)*255,0,255).astype(np.uint8).transpose(1,2,0)
    Image.fromarray(pixels).save(path)

@torch.no_grad()
def development(models,stats,static,policy):
    # Development is first opened here, after immutable calibration selections.
    assert read(OUT/'calibration_complete.json')['policy_sha256']==sha(RESULT/'plugin_selected_policy.json')
    records,bindings=population('development');assert len(records)==100
    dev_identity=dict(config_sha256=sha(RESULT/'plugin_config.json'),policy_sha256=sha(RESULT/'plugin_selected_policy.json'),
        bindings=bindings,source_ids=[r['image_id'] for r in records],preprocessing_ids=[r['preprocessing_id'] for r in records])
    seal(OUT/'development_identity.json',dev_identity)
    vae,var,dec,p=models['vae'],models['var'],models['decoder'],models['P512']
    F=[];T=[];reference=[]
    for record in records:
        check();image=torch.tensor(record['pixels'][None],dtype=torch.float32,device='cuda')/127.5-1
        f=vae.quant_conv(vae.encoder(image));F.append(f[0].cpu())
        T.append(torch.cat(vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=False),1)[0].cpu())
        reference.append(dino_features(models['dino'],(image+1)/2)[0])
    F=torch.stack(F);T=torch.stack(T);reference=torch.stack(reference)
    permutation=torch.tensor(read(RESULT/'plugin_config.json')['mismatch_permutation'],dtype=torch.long)
    obs=observations(F,records,DEV_SEEDS,p,'development');allrows=[];tokenrows=[]
    for source in range(100):
        check();cellpath=OUT/'development_cells'/f'{source:03d}.json';cellpath.parent.mkdir(exist_ok=True)
        if cellpath.exists():
            saved=read(cellpath);assert saved['identity']==dev_identity
            allrows.extend(saved['rows']);tokenrows.extend(saved['token_rows']);continue
        rows=[];tokrows=[]
        for snr in SNRS:
            level=policy['levels'][str(snr)];ind=[i for i,r in enumerate(obs['rows']) if r['source_index']==source and r['snr_db']==snr]
            Z=obs['Z'][ind];ii=obs['source_indices'][ind];f=F[ii].cuda();truth=old.b.split(T[ii].cuda());candidates={}
            estimates={'P512':Z};post_for={};methodinfo={}
            for prior in ['A1','A2','V']:
                selected=level['methods'][prior];lam=selected['raw_selected_lambda'];key=candidate_key(prior,lam)
                if key not in candidates:
                    result=rx.infer(vae,var,Z.cuda(),prior,level['variance_by_scale'],static,lam=lam,clean=f,truth=truth,details=True)
                    candidates[key]=result
                result=candidates[key];Q=result['fhat'].cpu();post_for[prior]=Q
                alpha=torch.tensor(selected['alpha'],dtype=torch.float32).reshape(1,32,1,1)
                common=torch.tensor(level['alpha_common'],dtype=torch.float32).reshape(1,32,1,1)
                for suffix,value in [('policy',Z if selected['policy_action']=='BYPASS' else rx.fusion(Z,Q,alpha)),('raw_fused',rx.fusion(Z,Q,alpha)),('common',rx.fusion(Z,Q,common)),('tok',Q)]:
                    name=prior+'_'+suffix;estimates[name]=value;methodinfo[name]=(prior,lam,selected['policy_action'] if suffix=='policy' else 'DIAGNOSTIC')
                tf=rx.tf_metrics(vae,var,Z.cuda(),truth,prior,level['variance_by_scale'],static,lam=lam)
                for mode,metrics in [('CL',result['metrics']),('TF',tf)]:
                    for k,m in enumerate(metrics):
                        for j,seed in enumerate(DEV_SEEDS):
                            per_image={name[:-10]:values[j] for name,values in m.items() if name.endswith('_per_image')}
                            tokrows.append(dict(source_id=records[source]['image_id'],source_index=source,snr_db=snr,noise_seed=seed,method=prior,lambda_=lam,
                               mode=mode,scale=k+1,token_count=m['token_count'],**per_image))
            key='V_lambda1.0'
            if key not in candidates:candidates[key]=rx.infer(vae,var,Z.cuda(),'V',level['variance_by_scale'],static,lam=1.,details=False)
            q1=candidates[key]['fhat'].cpu();a1=torch.tensor(level['V_lambda1']['alpha'],dtype=torch.float32).reshape(1,32,1,1)
            estimates['V_lambda1']=rx.fusion(Z,q1,a1);methodinfo['V_lambda1']=('V',1.,'CORRECT')
            # Reuse identical tensors/outputs, especially all true bypasses.
            rendered={};preview={}
            for name,value in estimates.items():
                identical=next((other for other in rendered if torch.equal(value,estimates[other])),None)
                if identical is None:
                    metric,images=images_metrics(value,ii,records,models,reference,permutation,save_images=True)
                    rendered[name]=(metric,images)
                else:rendered[name]=rendered[identical]
                metric,images=rendered[name];preview[name]=images[0].numpy()
                prior,lam,action=methodinfo.get(name,('P512','', 'BYPASS'))
                Q=post_for.get(prior,Z) if name!='V_lambda1' else q1
                baseerr=(F[ii].double()-Z.double()).square().flatten(1).sum(1)
                posterr=(F[ii].double()-Q.double()).square().flatten(1).sum(1)
                finalerr=(F[ii].double()-value.double()).square().flatten(1).sum(1)
                for j,r in enumerate(metric):rows.append(dict(**obs['rows'][ind[j]],base_model_id=base_model_id(),
                    in_training_range=snr in SNRS,method=name,system='continuous_receiver_plugin',decoder_id='Dc',
                    class_condition='unconditional',phy_family='continuous',energy_constraint='per_frame_2N',N_header=0,N_body=512,
                    source_bits='',coded_bits='',effective_code_rate='',action_m='',modulation='',header_ok='',body_crc_ok='',
                    training_seed=2026093001,model_id=base_model_id(),energy=obs['rows'][ind[j]]['E'],rx_ms='',output_type=name.split('_')[-1] if name!='P512' else 'base',
                    lambda_=lam,policy_action=action,dino_fail=r['dino_cosine']<.6,lpips_fail=r['lpips_alex']>.35,
                    latent_valid=True,decoder_applied=True,
                    latent_sq_err_base=float(baseerr[j]),latent_sq_err_post=float(posterr[j]),latent_sq_err_final=float(finalerr[j]),latent_sq_error=float(finalerr[j]),**r))
            if source in EXAMPLES and snr in [4,13]:
                render_example(RESULT/'examples/plugin'/f'source_{source:03d}_snr_{snr}.png',records[source]['pixels']/255.,
                    [(n,preview[n]) for n in ['P512','A1_policy','A2_policy','V_policy']])
                save_image(RESULT/'examples/plugin'/f'source_{source:03d}_snr_{snr}_Source_seed2001.png',records[source]['pixels']/255.)
                for name in ['P512','V_policy']:
                    save_image(RESULT/'examples/plugin'/f'source_{source:03d}_snr_{snr}_{name}_seed2001.png',preview[name])
        seal(cellpath,dict(rows=rows,token_rows=tokrows,identity=dev_identity));allrows.extend(rows);tokenrows.extend(tokrows)
        csv_rows(RESULT/'plugin_per_frame.csv',allrows);csv_rows(RESULT/'plugin_token_accuracy.csv',tokenrows)
        status('development',sources=source+1,total=100,rows=len(allrows))
    csv_rows(RESULT/'plugin_per_frame.csv',allrows);csv_rows(RESULT/'plugin_token_accuracy.csv',tokenrows)
    return records,F,obs,allrows

@torch.no_grad()
def timing(models,static,policy,records,F,obs):
    path=RESULT/'plugin_timing.csv'
    if path.exists():
        try:
            with path.open() as f:existing=list(csv.DictReader(f))
            grid={(int(r['source_index']),int(r['snr_db']),r['method'],int(r['repeat'])) for r in existing}
            expected={(i,s,m,r) for i in TIMING for s in SNRS for m in ['P512','V_raw','V_policy','V_lambda1'] for r in range(2)}
            if len(existing)==len(expected) and grid==expected:return
        except (ValueError,KeyError):pass
    rows=[];p=models['P512'];dec=models['decoder'];vae=models['vae'];var=models['var']
    def receive(y,snr,kind):
        torch.cuda.synchronize();start=time.perf_counter()
        Z=p.receive(torch.tensor(y[None],device='cuda'),torch.tensor([snr],dtype=torch.float32,device='cuda'))
        torch.cuda.synchronize();Pdone=time.perf_counter()
        level=policy['levels'][str(snr)];selected=level['methods']['V']
        fixed_lambda1=kind=='V_lambda1'
        lam=1. if fixed_lambda1 else selected['raw_selected_lambda']
        alpha=level['V_lambda1']['alpha'] if fixed_lambda1 else selected['alpha']
        runvar=kind in ('V_raw','V_lambda1') or (kind=='V_policy' and selected['policy_action']!='BYPASS')
        if runvar:
            Q=rx.infer(vae,var,Z,'V',level['variance_by_scale'],static,lam=lam,details=False)['fhat']
            Z=rx.fusion(Z,Q,torch.tensor(alpha,device='cuda',dtype=torch.float32).reshape(1,32,1,1))
        torch.cuda.synchronize();Qdone=time.perf_counter();image=dec(Z).cpu().numpy()
        torch.cuda.synchronize();finish=time.perf_counter()
        return dict(rx_ms=1000*(finish-start),P_receive_ms=1000*(Pdone-start),correction_ms=1000*(Qdone-Pdone),
                    decode_ms=1000*(finish-Qdone),ran_VAR=runvar and lam>0,
                    raw_selected_lambda='' if kind=='P512' else lam,
                    policy_action='BYPASS' if kind=='P512' else 'CORRECT' if kind in ('V_raw','V_lambda1') else selected['policy_action'])
    for i in TIMING:
        wave=p.transmit(F[i:i+1].cuda())[0].cpu().numpy()
        for method in ['P512','V_raw','V_policy','V_lambda1']:
            for _ in range(3):check();receive(apply_channel(wave,13,records[i]['image_id'],2001,CELL),13,method)
            for snr in SNRS:
                for repeat in range(2):
                    check();y=apply_channel(wave,snr,records[i]['image_id'],2001,CELL)
                    rows.append(dict(source_id=records[i]['image_id'],source_index=i,snr_db=snr,noise_seed=2001,
                          method=method,repeat=repeat,endpoints='CPU waveform -> CPU float RGB; noise outside RX',**receive(y,snr,method)))
        status('timing',sources=TIMING.index(i)+1,total=10)
    csv_rows(path,rows)

def config(identity,records,bindings):
    rng=np.random.default_rng(20260930)
    while True:
        permutation=rng.permutation(100)
        if np.all(permutation!=np.arange(100)):break
    source_files=[*HERE.glob('*.py'),HERE/'EXECUTION_PLAN.md',HERE/'protocol.json',OLD/'probe.py',OLD/'rx_v3_common.py',OLD/'run_preflight.py',
        ROOT/'src/var_comm/quality.py',ROOT/'src/var_comm/next_scale_prior.py',
        ROOT/'experiments/token_channel_efficiency_20260923/src/token_efficiency/execution.py']
    return dict(scope='N512_FROZEN_RECEIVER_PLUGIN',historical_git='07cafae0e4bb9469a37723d4d27260895d3c28d3',
      identity=identity,calibration_sources=[r['image_id'] for r in records],calibration_bindings=bindings,
      snrs=SNRS,main_snrs=SNRS,training_snrs=SNRS,lambda_grid=GRID,
      calibration_seeds=CAL_SEEDS,development_seeds=DEV_SEEDS,N=512,E=1024,
      mismatch_permutation=permutation.tolist(),mismatch_seed=20260930,bootstrap_seed=20260930,bootstrap_repeats=10000,
      timing_indices=TIMING,warmup_repeats=3,timing_repeats=2,examples_indices=EXAMPLES,examples_snrs=[4,13],
      variance_floor=1e-12,beta=0,statistical_zero_tolerance=1e-12,
      variance_reduction='population/source/noise/spatial/channel; channel diagnostics separately',
      zero_fusion_denominator_alpha=0,lambda0='nearest-codeword; A2/V aliases of A1; no VAR call',
      bypass='return exactly Fhat_P512 and reuse P512 image',fusion='per-method calibration alpha; common alpha=A1',
      noise_rule="shared original apply_channel: VAR-CONTINUOUS-512|source_id, seed, (512,2), float32 sigma=10**(-snr/20); one AWGN only",
      no_development_tuning=True,exploratory_reused_development=True,evaluation_training_updates=0,
      checkpoint_selected_on='full_1000_calibration_5SNR_3noise_before_receiver_calibration',
      receiver_calibration_sources=200,development_sources=100,selected_updates=identity['selected']['step'],
      receiver_origin=dict(path=str(ROOT/'experiments/rx-posterior-step2-A-20260930/receiver.py'),
          sha256=sha(ROOT/'experiments/rx-posterior-step2-A-20260930/receiver.py'),
          identical=sha(HERE/'receiver.py')==sha(ROOT/'experiments/rx-posterior-step2-A-20260930/receiver.py')),
      H_R_rule='same LPIPS branch or same DINO+specificity branch must exceed P512/A1/A2 controls; bypass excluded; PSNR cap .3 (.2 at 13)',
      psnr_drop_caps_db={str(s):.2 if s==13 else .3 for s in SNRS},
      source_bindings={str(p):sha(p) for p in source_files})

@torch.no_grad()
def main(preflight=False):
    OUT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():
        done=read(OUT/'completion.json')
        assert done['selected_sha256']==sha(SELECTED_PATH)
        return
    status('loading_models');models,stats,static,identity=setup();F,Fq,T,records,bindings=calibration_sources(stats)
    cfg=config(identity,records,bindings)
    if preflight or not (OUT/'selfcheck.json').exists():selfcheck(models,stats,static,F,Fq,T,records)
    assert read(OUT/'selfcheck.json')['passed']
    assert read(OUT/'selfcheck.json')['selected_sha256']==sha(SELECTED_PATH)
    if preflight:status('PREFLIGHT_PASS');return
    seal(RESULT/'plugin_config.json',cfg)
    obs=observations(F,records,CAL_SEEDS,models['P512'],'calibration')
    errors,errrows=error_statistics(F,obs,stats)
    policy=calibrate(F,T,records,obs,stats,static,models,errors,errrows,cfg)
    devrecords,devF,devobs,rows=development(models,stats,static,policy)
    timing(models,static,policy,devrecords,devF,devobs)
    assert dump_hashes(models)==identity['models'],'frozen weights changed'
    assert sha(SELECTED_PATH)==identity['selected_sha256'],'selected checkpoint descriptor changed'
    for p,h in cfg['source_bindings'].items():assert sha(p)==h,'bound source changed'
    write(OUT/'completion.json',dict(status='N512_PLUGIN_EVALUATION_COMPLETE',metric_rows=len(rows),
       selected_sha256=identity['selected_sha256'],
       config_sha256=sha(RESULT/'plugin_config.json'),policy_sha256=sha(RESULT/'plugin_selected_policy.json'),
       frozen_weights_unchanged=True,finished=time.time(),evaluation_training_updates=0,
       selected_updates=identity['selected']['step']))
    status('COMPLETE',evaluation_training_updates=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--selected-json',type=Path,default=SELECTED_PATH)
    args=parser.parse_args();SELECTED_PATH=args.selected_json.resolve()
    try:main(args.preflight_only)
    except old.b.ResourceBusy as error:
        status('PAUSED_SAFE_BOUNDARY',reason=str(error));raise SystemExit(75)
    except Exception:
        write(OUT/('failure_'+str(int(time.time()))+'.json'),dict(traceback=traceback.format_exc(),time=time.time()))
        status('FAILED',traceback=traceback.format_exc());traceback.print_exc();raise
