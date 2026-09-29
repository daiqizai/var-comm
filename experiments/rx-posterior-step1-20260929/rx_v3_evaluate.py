"""v3 B development: frozen calibration; real frozen models; full source pairing."""
import rx_v3_b_helpers as h
from rx_v3_b_helpers import c,torch,np,Path,read,sha,seal,cell,OUT
import time,traceback
@torch.no_grad()
def main():
    reg=c.verify_registration('B_evaluation')
    if (OUT/'B_completion.json').exists():return
    assert cell(OUT/'limit_check.json')['passed']
    assert cell(OUT/'A_completion.json')['source_A_approximate_check_passed']
    cal=cell(OUT/'B_calibration_completion.json');frozen=cell(OUT/'B_frozen_config.json')
    assert cal['frozen_config_sha256']==sha(OUT/'B_frozen_config.json') and frozen['development_read'] is False
    assert frozen['algorithm_sha256']==sha(h.ALGORITHM)
    cfg=frozen['algorithm'];vae,var,stats,static=c.start()
    decoder=c.b.load_decoder(vae,torch.device('cuda:0'));dcsha=c.b.state_sha256(decoder)
    assert dcsha==cell(OUT/'A_completion.json')['decoder_state_sha256']
    lp,dino,qidentity=c.load_quality();assert qidentity==cell(OUT/'quality_identity.json')
    # Development access occurs strictly after the sealed calibration/config checks above.
    from latent_enhancement_eval.runner import load_targets
    targets=load_targets()
    population=[dict(source_id=r['target']['image_id'],index=r['index'],rgb_sha256=r['rgb_sha256'],source_npz_sha256=r['source_npz_sha256']) for r in targets]
    assert population==cell(OUT/'A_population.json')['records']
    access=dict(frozen_config_sha256=sha(OUT/'B_frozen_config.json'),registration_sha256=reg,population=population)
    if (OUT/'B_population.json').exists():assert cell(OUT/'B_population.json')==access
    else:seal(OUT/'B_population.json',access)
    hashes={};nrows=0
    for rec in targets:
        c.b.boundary();i=rec['index'];sid=rec['target']['image_id'];p=OUT/'B_cells'/f'{i:03d}.json';old=cell(p)
        if old is not None:
            assert old['source_identity']==population[i] and old['registration_sha256']==reg and old['frozen_config_sha256']==sha(OUT/'B_frozen_config.json')
            hashes[str(p)]=sha(p);nrows+=len(old['rows']);continue
        F,truth,Fq=c.dev_source(vae,rec);source=rec['pixels'].astype(np.float32)/255;lpv=h.canonical(vae,var,truth);rows=[]
        for selection in frozen['levels']:
            eta=selection['eta'];vs=c.variances(stats,eta)
            for seed in cfg['development_noise_seeds']:
                c.b.boundary();Z=c.observe(F,stats,sid,seed,eta)
                tf=h.tf_metrics(vae,var,F,truth,Z,stats,static,selection,lpv)
                shared=dict(level=selection['name'],snr_equiv_db=selection['snr_db'],level_role=selection['role'],eta=eta,noise_seed=seed,
                            source_id=sid,source_index=i,population='original_development_100',preprocessing_sha256=rec['rgb_sha256'])
                for profile,data in tf.items():
                    rows.append(dict(**shared,mode='TF',profile=profile,**data))
                if selection['role']!='decision':continue
                dedup={}
                for prior,version,lam in h.profiles(selection):
                    ident=(prior,lam);profile=h.key(prior,version)
                    if ident not in dedup:
                        dg=selection['probability_only'].get(prior)
                        result=c.infer(vae,var,Z,truth,prior,'CL',vs,static,lam,clean=F,
                            diag_variance=c.variances(stats,eta,dg['beta']) if dg else None,
                            diag_lam=dg['lambda'] if dg else None)
                        dedup[ident]=dict(fhat=result['fhat'],scales=result['metrics'],probability_only=result['diagnostic_metrics'],latent_squared_error=result['latent_error'])
                    result=dedup[ident]
                    rhorec=frozen['rho'][selection['name']+'/'+profile];rho=torch.tensor(rhorec['variance'],device='cuda',dtype=torch.float32).reshape(1,32,1,1)
                    fused=c.fusion(result['fhat'],Z,stats,rho,eta)
                    images=decoder(torch.cat([result['fhat'],fused],0)).cpu().numpy()
                    quality=c.quality(source,list(images),lp,dino)
                    rows.append(dict(**shared,mode='CL',profile=profile,prior=prior,version=version,lambda_=lam,beta=0,
                         scales=result['scales'],probability_only=result['probability_only'],latent_squared_error=result['latent_squared_error'],
                         fused_latent_squared_error=float((F.double()-fused.double()).square().sum()),
                         token_image=quality[0],fused_image=quality[1]))
        obj=dict(source_identity=population[i],source_id=sid,source_index=i,registration_sha256=reg,
                 frozen_config_sha256=sha(OUT/'B_frozen_config.json'),synthetic=False,rows=rows)
        seal(p,obj);hashes[str(p)]=sha(p);nrows+=len(rows);c.status('B_evaluation',sources=i+1,total=100)
    assert c.b.state_sha256(vae)==read(c.b.OUT/'models.json')['vae_state_sha256'] and c.b.state_sha256(var)==read(c.b.OUT/'models.json')['var_state_sha256']
    assert c.b.state_sha256(decoder)==dcsha and c.b.state_sha256(lp)==qidentity['lpips_state'] and c.b.state_sha256(dino)==qidentity['dino_state']
    c.verify_registration('B_evaluation')
    seal(OUT/'B_completion.json',dict(status='REAL_V3_B_DEVELOPMENT_COMPLETE',sources=100,rows=nrows,cells=hashes,
         synthetic=False,registration_sha256=reg,frozen_config_sha256=sha(OUT/'B_frozen_config.json'),time=time.time(),
         note='Original training stays stopped; no automatic step 2; M3 is image utility, token metrics are mechanism evidence.'))
if __name__=='__main__':
    try:main()
    except c.b.ResourceBusy as exc:c.status('B_evaluation',status='SAFE_SOURCE_PAUSE',reason=str(exc));raise SystemExit(75)
    except Exception:c.write(OUT/'B_evaluation_failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
