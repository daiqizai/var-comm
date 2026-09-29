"""Calibration-only v3 B. Never reads development; all decisions sealed before it."""
import rx_v3_b_helpers as h
from rx_v3_b_helpers import c,torch,np,Path,read,sha,seal,cell,OUT
import time,traceback
@torch.no_grad()
def main():
    reg=c.verify_registration('B_calibration');cfg=h.algorithm()
    assert cell(OUT/'limit_check.json')['passed']
    if (OUT/'B_calibration_completion.json').exists():return
    vae,var,stats,static=c.start();F,Fq,T,ids,bound=c.b.load_calibration()
    assert ids[:200]==stats['subset_ids']
    grid=[];proper=[];bindings={}
    for i in range(200):
        c.b.boundary();p=OUT/'B_grid_cells'/f'{i:03d}.json';d=cell(p)
        if d is None:
            g,q=h.calibration_grid(vae,var,F[i:i+1].cuda(),c.b.split(T[i:i+1].cuda()),ids[i],stats,static,cfg)
            d=dict(source_index=i,source_id=ids[i],registration_sha256=reg,grid=g.tolist(),probability_only=q.tolist(),synthetic=False)
            seal(p,d)
        assert d['source_id']==ids[i] and d['registration_sha256']==reg
        grid.append(d['grid']);proper.append(d['probability_only']);bindings[str(p)]=sha(p)
        c.status('B_calibration',phase='TF_accuracy_and_probability_grid',sources=i+1,total=200)
    selections=h.select_arrays(np.asarray(grid),np.asarray(proper),cfg)
    selection_obj=dict(algorithm_sha256=sha(h.ALGORITHM),registration_sha256=reg,levels=selections,
                       calibration_sources=ids[:200],noise_seeds=cfg['calibration_noise_seeds'],
                       grid_cells=bindings,development_read=False)
    p=OUT/'B_selection.json'
    if p.exists():assert cell(p)==selection_obj
    else:seal(p,selection_obj)
    del grid,proper
    sums={};squares={};counts={};rho_cells={}
    for i in range(200):
        c.b.boundary();p=OUT/'B_rho_cells'/f'{i:03d}.json';d=cell(p)
        if d is None:
            f=F[i:i+1].cuda();truth=c.b.split(T[i:i+1].cuda());rows=[]
            for selection in selections:
                if selection['role']!='decision':continue
                eta=selection['eta'];vs=c.variances(stats,eta)
                for seed in cfg['calibration_noise_seeds']:
                    c.b.boundary();Z=c.observe(f,stats,ids[i],seed,eta);dedup={}
                    for prior,version,lam in h.profiles(selection):
                        ident=(prior,lam)
                        if ident not in dedup:
                            result=c.infer(vae,var,Z,truth,prior,'CL',vs,static,lam,clean=f,details=True)
                            err=((f.double()-result['fhat'].double())/stats['sigma'].cuda().double())
                            data=dict(sum=err.sum((0,2,3)).tolist(),sum_sq=err.square().sum((0,2,3)).tolist(),count=256,
                                      path_accuracy=[m['path_accuracy'] for m in result['metrics']],latent_squared_error=result['latent_error'])
                            if i==0 and selection==selections[0] and seed==cfg['calibration_noise_seeds'][0] and prior=='V':
                                test=c.infer(vae,var,Z,truth,prior,'CL',vs,static,lam,clean=f+.123,details=True)
                                assert torch.equal(test['fhat'],result['fhat']) and all(torch.equal(x,y) for x,y in zip(test['tokens'],result['tokens']))
                                data['clean_diagnostic_branch_cannot_change_RX']=True
                            dedup[ident]=data
                        rows.append(dict(level=selection['name'],noise_seed=seed,prior=prior,version=version,lambda_=lam,**dedup[ident]))
            d=dict(source_index=i,source_id=ids[i],registration_sha256=reg,selection_sha256=sha(OUT/'B_selection.json'),rows=rows,synthetic=False)
            seal(p,d)
        assert d['source_id']==ids[i] and d['registration_sha256']==reg and d['selection_sha256']==sha(OUT/'B_selection.json')
        for row in d['rows']:
            k=row['level']+'/'+h.key(row['prior'],row['version'])
            sums[k]=sums.get(k,np.zeros(32))+np.array(row['sum']);squares[k]=squares.get(k,np.zeros(32))+np.array(row['sum_sq'])
            counts[k]=counts.get(k,0)+row['count']
        rho_cells[str(p)]=sha(p);c.status('B_calibration',phase='CL_fusion_variance',sources=i+1,total=200)
    rho={}
    for k in sums:
        variance=squares[k]/counts[k]-(sums[k]/counts[k])**2
        assert np.isfinite(variance).all() and (variance>=0).all() and counts[k]==200*3*256
        rho[k]=dict(variance=variance.tolist(),mean=(sums[k]/counts[k]).tolist(),count_per_channel=counts[k])
    frozen=dict(status='CALIBRATION_ONLY_FROZEN_BEFORE_B_DEVELOPMENT',algorithm=cfg,algorithm_sha256=sha(h.ALGORITHM),
                registration_sha256=reg,selection_sha256=sha(OUT/'B_selection.json'),levels=selections,rho=rho,
                calibration_statistics_sha256=sha(c.b.OUT/'calibration_statistics.pt'),calibration_cache_bindings=bound,
                rho_cells=rho_cells,development_read=False,time=time.time())
    seal(OUT/'B_frozen_config.json',frozen)
    assert c.b.state_sha256(vae)==read(c.b.OUT/'models.json')['vae_state_sha256'] and c.b.state_sha256(var)==read(c.b.OUT/'models.json')['var_state_sha256']
    c.verify_registration('B_calibration')
    seal(OUT/'B_calibration_completion.json',dict(status='REAL_V3_B_CALIBRATION_COMPLETE',registration_sha256=reg,
         frozen_config_sha256=sha(OUT/'B_frozen_config.json'),grid_sources=200,rho_sources=200,development_read=False,time=time.time()))
if __name__=='__main__':
    try:main()
    except c.b.ResourceBusy as exc:c.status('B_calibration',status='SAFE_SOURCE_PAUSE',reason=str(exc));raise SystemExit(75)
    except Exception:c.write(OUT/'B_calibration_failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
