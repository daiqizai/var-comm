"""Versioned calibration and hard gates for the user's posterior variance revision."""
import run_preflight  # environment paths only; no launcher executes on import
import probe as base
from probe import torch, np, fn, Path, json, time, os, traceback
from probe import digest, write_json, read, PATCH_NUMS, ResourceBusy
OUT=base.OUT/'revision_v2'
DESIGN=Path(__file__).with_name('design_v2.json')

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def seal(path,data):
    path=Path(path)
    if path.exists():raise RuntimeError('immutable cell already exists: '+str(path))
    atomic_json(path,data)
    Path(str(path)+".sha256").write_text(digest(path)+"\n")

def progress(stage,**extra):
    atomic_json(OUT/'status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**extra))

def parameters():
    return [(b,l) for b in read(DESIGN)['beta_grid'] for l in read(DESIGN)['lambda_grid']]

def variance(stats,eta,beta):
    return stats['unit_noise_var'].double()*eta**2+beta*stats['s2'].double()

def choose(scores):
    """Maximum proper score; ascending beta/lambda wins exact ties."""
    assert len(scores)==len(parameters()) and np.isfinite(scores).all()
    return int(np.argmax(scores))

def load_inputs():
    cfg=read(DESIGN);base.configure_runtime();base.require_available();base.SAFETY=base.Safety()
    lease=read(base.OUT/'lease_status.json');assert time.time()-lease['time']<40
    ident=lease['process'];stat=Path('/proc')/str(ident['pid'])/'stat'
    assert stat.read_text().split(') ')[1].split()[19]==str(ident['start_ticks'])
    assert (Path('/proc')/str(ident['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode()==ident['cmdline']
    F,Fq,T,ids,bindings=base.load_calibration()
    meta=read(base.OUT/'calibration_statistics.json')
    assert bindings==meta['cache_bindings'] and digest(base.OUT/'calibration_statistics.pt')==meta['sha256']
    assert meta['design_sha256']==digest(base.CONFIG)
    stats=torch.load(base.OUT/'calibration_statistics.pt',map_location='cpu',weights_only=True)
    assert stats['subset_ids']==ids[:200]
    registration=read(OUT/'registration.json')
    for p,h in registration['bindings'].items():assert digest(p)==h,('binding changed',p)
    paths=base.model_paths()
    for p,h in read(base.OUT/'models.json')['files'].items():assert digest(p)==h
    vae,var=base.load_models(paths,torch.device('cuda:0'))
    assert base.state_sha256(vae)==read(base.OUT/'models.json')['vae_state_sha256']
    assert base.state_sha256(var)==read(base.OUT/'models.json')['var_state_sha256']
    static=[base.frequency_log_probs(c,cfg['static_pseudocount']).float().cuda() for c in stats['counts']]
    return cfg,F,Fq,T,ids,stats,vae,var,static

def checked_cell(path):
    assert Path(str(path)+".sha256").read_text().strip()==digest(path), ("sealed cell hash",str(path))
    obj=read(path)
    assert obj['registration_sha256']==digest(OUT/'registration.json')
    return obj

@torch.no_grad()
def difficulty_pass(name,snrs,inputs):
    cfg,F,Fq,T,ids,stats,vae,var,static=inputs
    etas=torch.tensor([10**(-s/20) for s in snrs],device='cuda',dtype=torch.float32)
    totals=np.zeros(len(snrs),dtype=np.float64);cell_hashes={}
    for i in range(200):
        base.boundary();path=OUT/name/f'{i:04d}.json'
        if path.exists():cell=checked_cell(path)
        else:
            truth=base.split(T[i:i+1].cuda());rest=F[i:i+1].cuda().clone()
            selected={}
            for k in range(10):
                if k in (7,8,9):selected[k]=rest.clone()
                rest.sub_(base.contribution(vae.quantize,truth[k],k))
            accuracies=np.zeros(len(snrs),dtype=np.float64)
            for seed in cfg['calibration_noise_seeds']:
                n=(stats['sigma']*base.noise(ids[i],seed)).cuda()
                for k in (7,8,9):
                    for start in range(0,len(snrs),cfg['difficulty_eta_batch']):
                        e=etas[start:start+cfg['difficulty_eta_batch']]
                        r=selected[k]+e[:,None,None,None]*n
                        d,_=base.distance(vae.quantize,r,k)
                        a=d.argmin(-1).eq(truth[k]).double().mean(-1).cpu().numpy()
                        accuracies[start:start+len(e)]+=a/9
            cell=dict(registration_sha256=digest(OUT/'registration.json'),source_id=ids[i],
                      source_index=i,snrs_db=snrs,accuracy=accuracies.tolist(),seeds=cfg['calibration_noise_seeds'],
                      scales=[8,9,10],aggregation='equal scales, then noise seeds, then sources')
            seal(path,cell)
        assert cell['snrs_db']==snrs and cell['source_id']==ids[i]
        totals+=np.asarray(cell['accuracy'])/200;cell_hashes[str(path)]=digest(path)
        progress(name,sources=i+1,total=200)
    return dict(snrs_db=snrs,accuracy=totals.tolist(),cells=cell_hashes)

@torch.no_grad()
def difficulty(inputs):
    cfg=inputs[0];p=OUT/'noise_levels.json'
    if p.exists():return checked_cell(p)
    coarse=difficulty_pass('difficulty_coarse',cfg['difficulty_coarse_snrs_db'],inputs)
    best=[int(np.argmin(np.abs(np.asarray(coarse['accuracy'])-t))) for t in cfg['difficulty_targets']]
    fine=sorted(set(round(coarse['snrs_db'][j]+d/10,1) for j in best for d in range(-10,11)))
    refined=difficulty_pass('difficulty_fine',fine,inputs)
    choices=[]
    for target in cfg['difficulty_targets']:
        j=int(np.argmin(np.abs(np.asarray(refined['accuracy'])-target)))
        choices.append(dict(target_accuracy=target,snr_db=fine[j],eta=10**(-fine[j]/20),
                            achieved_accuracy=refined['accuracy'][j],absolute_target_error=abs(refined['accuracy'][j]-target)))
    assert len({x['snr_db'] for x in choices})==4
    result=dict(registration_sha256=digest(OUT/'registration.json'),levels=choices,coarse=coarse,refined=refined,
                source_role='calibration_only',development_accessed=False)
    seal(p,result);return result

@torch.no_grad()
def calibrate(inputs,levels):
    cfg,F,Fq,T,ids,stats,vae,var,static=inputs
    grid=parameters();sums=np.zeros((4,3,len(grid)),dtype=np.float64);hashes={}
    for i in range(200):
        base.boundary();path=OUT/'proper_score_cells'/f'{i:04d}.json'
        if path.exists():cell=checked_cell(path)
        else:
            truth=base.split(T[i:i+1].cuda())
            canonical=var(torch.tensor([1000],device='cuda'),vae.quantize.idxBl_to_var_input(truth)).float()
            logpriors=[fn.log_softmax(x,dim=-1) for x in base.split(canonical)]
            clean=F[i:i+1].cuda().clone();rests={}
            for k in range(10):
                if k>=6:rests[k]=clean.clone()
                clean.sub_(base.contribution(vae.quantize,truth[k],k))
            scores=np.zeros((4,3,len(grid)),dtype=np.float64)
            for seed in cfg['calibration_noise_seeds']:
                n=(stats['sigma']*base.noise(ids[i],seed)).cuda()
                for level,lev in enumerate(levels['levels']):
                    eta=lev['eta']
                    for k in range(6,10):
                        d,_=base.distance(vae.quantize,rests[k]+eta*n,k)
                        for pidx,prior in enumerate(['A1','A2','V']):
                            lp=0 if prior=='A1' else (static[k][None,None] if prior=='A2' else logpriors[k])
                            for g,(beta,lam) in enumerate(grid):
                                v=float(variance(stats,eta,beta)[k]);assert v>0
                                logits=-d/(2*v)+lam*lp
                                proper=(logits.gather(-1,truth[k][...,None]).squeeze(-1)-torch.logsumexp(logits,dim=-1)).double().mean()
                                scores[level,pidx,g]+=float(proper)/12
            cell=dict(registration_sha256=digest(OUT/'registration.json'),source_id=ids[i],source_index=i,
                      noise_levels_sha256=digest(OUT/'noise_levels.json'),scores=scores.tolist(),
                      axes=['level','prior(A1,A2,V)','grid(beta,lambda)'],grid=grid,
                      aggregation='true-prefix log probability, equal scales 7-10 then 3 seeds then sources')
            seal(path,cell)
        assert cell['source_id']==ids[i] and cell['noise_levels_sha256']==digest(OUT/'noise_levels.json')
        sums+=np.asarray(cell['scores'])/200;hashes[str(path)]=digest(path)
        progress('PROPER_SCORE_CALIBRATION',sources=i+1,total=200)
    choices=[]
    for level,lev in enumerate(levels['levels']):
        row=dict(**lev,parameters={})
        for pidx,prior in enumerate(['A1','A2','V']):
            j=choose(sums[level,pidx]);b,l=grid[j]
            row['parameters'][prior]=dict(beta=b,lambda_=l,grid_index=j,mean_true_log_probability=float(sums[level,pidx,j]))
        choices.append(row)
    cfg_final=dict(design=cfg,registration_sha256=digest(OUT/'registration.json'),
                   noise_levels_sha256=digest(OUT/'noise_levels.json'),levels=choices,
                   complete_grid_scores=sums.tolist(),grid=grid,proper_score_cell_sha256=hashes,
                   frozen_time=time.time(),development_accessed=False,holdout_accessed=False,
                   fixed=dict(beta=0,lambda_=1),decision_version='calibrated',
                   A1_note='positive scalar likelihood variance leaves MAP unchanged; calibrated beta/lambda only affect log-probability reporting')
    p=OUT/'frozen_config.json'
    if p.exists():
        old=checked_cell(p)
        assert old['levels']==choices and old['complete_grid_scores']==sums.tolist()
        return old
    seal(p,cfg_final);return cfg_final

@torch.no_grad()
def gates(inputs,config):
    cfg,F,Fq,T,ids,stats,vae,var,static=inputs
    settings=[('fixed',0,1)]+[(f'calibrated_level_{j}',x['parameters']['V']['beta'],x['parameters']['V']['lambda_']) for j,x in enumerate(config['levels'])]
    eta=10**(-cfg['engineering_snr_db']/20);rows=[];hashes={}
    for i in cfg['engineering_calibration_indices']:
        base.boundary();path=OUT/'engineering_cells'/f'{i:04d}.json'
        if path.exists():cell=checked_cell(path)
        else:
            f=F[i:i+1].cuda();truth=base.split(T[i:i+1].cuda());q=vae.quantize
            official=q.f_to_idxBl_or_fhat(f,to_fhat=False)
            # Exact zero-noise MAP limit has no division by zero and ignores finite priors.
            limit=q.f_to_idxBl_or_fhat(f,to_fhat=True)[-1]
            got,idx,_,_=base.infer(vae,var,f,truth,'A1','CL',torch.ones(10),static,lam=0)
            assert all(torch.equal(a,b) for a,b in zip(idx,official)) and torch.equal(got,limit)
            torch.testing.assert_close(limit,Fq[i:i+1].cuda(),rtol=1e-5,atol=2e-5)
            canonical=var(torch.tensor([1000],device='cuda'),q.idxBl_to_var_input(truth)).float()
            _,_,_,sequential=base.infer(vae,var,f,truth,'V','TF',variance(stats,eta,0),static,lam=1)
            err=float((canonical-torch.cat(sequential,1)).abs().max())
            assert err<=cfg['official_forward_absolute_tolerance']
            checks=[]
            for seed in cfg['calibration_noise_seeds']:
                z=f+(eta*stats['sigma']*base.noise(ids[i],seed)).cuda()
                for name,beta,lam in settings:
                    for mode in ['TF','CL']:
                        _,pred,metrics,_=base.infer(vae,var,z,truth,'V',mode,variance(stats,eta,beta),static,lam=lam,details=True)
                        checks.append(dict(version=name,mode=mode,seed=seed,beta=beta,lambda_=lam,
                                          token_correct=int(torch.cat(pred,1).eq(torch.cat(truth,1)).sum()),
                                          token_total=680,scale_metrics=metrics))
            cell=dict(registration_sha256=digest(OUT/'registration.json'),source_id=ids[i],source_index=i,
                      frozen_config_sha256=digest(OUT/'frozen_config.json'),zero_noise_nearest_tokens_exact=True,
                      zero_noise_Fq_bitwise_equal=True,official_TF_logits_max_abs=err,
                      snr_db=cfg['engineering_snr_db'],checks=checks)
            seal(path,cell)
        assert cell['frozen_config_sha256']==digest(OUT/'frozen_config.json')
        rows.extend(cell['checks']);hashes[str(path)]=digest(path);progress('REAL_FIXED_AND_CALIBRATED_HIGH_SNR_GATE',sources=len(hashes),total=len(cfg['engineering_calibration_indices']))
    summary=[]
    for name,b,l in settings:
        for mode in ['TF','CL']:
            subset=[r for r in rows if r['version']==name and r['mode']==mode]
            acc=sum(r['token_correct'] for r in subset)/sum(r['token_total'] for r in subset)
            summary.append(dict(version=name,mode=mode,beta=b,lambda_=l,token_accuracy=acc,passed=acc>=.99))
    passed=all(r['passed'] for r in summary)
    result=dict(status='REVISED_CALIBRATION_AND_HIGH_SNR_GATE_PASS' if passed else 'REVISED_HARD_GATE_FAILED_NO_DEVELOPMENT',
                passed=passed,synthetic=False,summary=summary,cells=hashes,engineering_snr_db=cfg['engineering_snr_db'],
                threshold=.99,frozen_config_sha256=digest(OUT/'frozen_config.json'),development_accessed=False,
                holdout_accessed=False,time=time.time())
    assert base.state_sha256(vae)==read(base.OUT/'models.json')['vae_state_sha256']
    assert base.state_sha256(var)==read(base.OUT/'models.json')['var_state_sha256']
    seal(OUT/'calibration_completion.json',result)
    if not passed:seal(OUT/'blocked.json',result)
    return result

@torch.no_grad()
def main():
    inputs=load_inputs()
    if (OUT/'calibration_completion.json').exists():
        result=read(OUT/'calibration_completion.json')
        assert result['frozen_config_sha256']==digest(OUT/'frozen_config.json')
        for path,h in result['cells'].items():assert digest(path)==h
        progress(result['status'],resumed_completed_boundary=True)
        return
    levels=difficulty(inputs)
    config=calibrate(inputs,levels)
    result=gates(inputs,config)
    assert base.state_sha256(inputs[6])==read(base.OUT/'models.json')['vae_state_sha256']
    assert base.state_sha256(inputs[7])==read(base.OUT/'models.json')['var_state_sha256']
    progress(result['status'],completion=str(OUT/'calibration_completion.json'))
if __name__=='__main__':
    try:main()
    except ResourceBusy as e:
        progress('SAFE_REVISED_PROBE_PAUSE',reason=str(e));raise SystemExit(75)
    except Exception:
        atomic_json(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
