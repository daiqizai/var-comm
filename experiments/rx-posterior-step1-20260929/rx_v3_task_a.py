"""Task A is independent of token hard gates. Original development population only."""
import rx_v3_common as c
from rx_v3_common import torch,np,Path,read,sha,seal,cell,OUT
import traceback,time
@torch.no_grad()
def limit_check(vae,var,stats,static):
    p=OUT/'limit_check.json'
    if p.exists():return cell(p)
    F,Fq,T,ids,bound=c.b.load_calibration();rows=[];passed=True
    for i in read(c.DESIGN)['engineering_calibration_indices']:
        c.b.boundary();f=F[i:i+1].cuda();truth=c.b.split(T[i:i+1].cuda())
        official=vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=False)
        expected=vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=True)[-1]
        canonical=var(torch.tensor([1000],device='cuda'),vae.quantize.idxBl_to_var_input(truth)).float()
        for mode in ['TF','CL']:
            a=c.infer(vae,var,f,truth,'A1',mode,torch.zeros(10),static,lam=1,details=False)
            v=c.infer(vae,var,f,truth,'V',mode,torch.zeros(10),static,lam=1,details=False)
            identical=all(torch.equal(x,y) for x,y in zip(a['tokens'],v['tokens']))
            official_equal=all(torch.equal(x,y) for x,y in zip(v['tokens'],official))
            fq_equal=torch.equal(v['fhat'],expected) and torch.equal(v['fhat'],a['fhat'])
            logit_error=float((torch.cat(v['logits'],1)-canonical).abs().max())
            ok=identical and official_equal and fq_equal and logit_error<=.0002
            passed=passed and ok
            rows.append(dict(source_id=ids[i],source_index=i,mode=mode,eta=0,beta=0,lambda_=1,
                V_A1_all_tokens_bitwise_equal=identical,V_official_all_tokens_equal=official_equal,
                Fq_bitwise_equal=fq_equal,token_count=680,official_TF_logits_max_abs=logit_error,passed=ok))
    result=dict(status='EXACT_ZERO_NOISE_LIMIT_CHECK_PASS' if passed else 'EXACT_LIMIT_CHECK_FAILED',passed=passed,
        checks=rows,development_used=False,calibration_cache_sha256=bound,design_sha256=sha(c.DESIGN),time=time.time())
    seal(p,result);return result
@torch.no_grad()
def main():
    reg=c.verify_registration('A');cfg=read(c.DESIGN)
    if (OUT/'A_completion.json').exists():return
    vae,var,stats,static=c.start()
    # Fast independent code check within the same GPU reservation; it is not a gate on Task A.
    limit=limit_check(vae,var,stats,static)
    decoder=c.b.load_decoder(vae,torch.device('cuda:0'));dc_sha=c.b.state_sha256(decoder)
    lp,dino,qidentity=c.load_quality()
    from latent_enhancement_eval.runner import load_targets
    targets=load_targets()
    source_identity=[dict(source_id=r['target']['image_id'],index=r['index'],rgb_sha256=r['rgb_sha256'],
                         source_npz_sha256=r['source_npz_sha256']) for r in targets]
    if not (OUT/'A_population.json').exists():seal(OUT/'A_population.json',dict(records=source_identity,registration_sha256=reg))
    else:assert cell(OUT/'A_population.json')['records']==source_identity
    qpath=OUT/'quality_identity.json'
    if qpath.exists():assert cell(qpath)==qidentity
    else:seal(qpath,qidentity)
    mu=stats['mu'].cuda();sigma=stats['sigma'].cuda();rho=stats['rho'].cuda();hashes={};reference=[]
    for rec in targets:
        c.b.boundary();i=rec['index'];sid=rec['target']['image_id'];p=OUT/'A_cells'/f'{i:03d}.json';old=cell(p)
        if old is not None:
            assert old['registration_sha256']==reg and old['source_id']==sid
            reference.append(old['source_A_Dc_true_Fq']);hashes[str(p)]=sha(p);continue
        F,truth,Fq=c.dev_source(vae,rec);source=rec['pixels'].astype(np.float32)/255
        clean=decoder(Fq)[0].cpu().numpy();reference_q=c.quality(source,[clean],lp,dino)[0];reference.append(reference_q)
        rows=[]
        x=(F-mu)/sigma
        for snr in cfg['oracle_snrs_db']:
            eta=10**(-snr/20)
            for seed in cfg['development_noise_seeds']:
                c.b.boundary();Z=c.observe(F,stats,sid,seed,eta);z=(Z-mu)/sigma
                B1=mu+sigma*z/(1+eta**2);O1=c.fusion(Fq,Z,stats,rho,eta)
                images=decoder(torch.cat([B1,O1],0)).cpu().numpy();metrics=c.quality(source,list(images),lp,dino)
                for method,m,latent in zip(['B1','O1'],metrics,[B1,O1]):
                    rows.append(dict(task='A',method=method,source_id=sid,source_index=i,
                        population='original_development_100',preprocessing_sha256=rec['rgb_sha256'],
                        snr_equiv_db=snr,eta=eta,noise_seed=seed,latent_squared_error=float((F.double()-latent.double()).square().sum()),
                        standardized_signal_energy=float(x.double().square().sum()),
                        standardized_noise_energy=float((z.double()-x.double()).square().sum()),**m))
        obj=dict(registration_sha256=reg,source_id=sid,source_index=i,rows=rows,source_A_Dc_true_Fq=reference_q,
                 source_identity=source_identity[i],synthetic=False)
        seal(p,obj);hashes[str(p)]=sha(p);c.status('A',sources=i+1,total=100)
    source_A={k:float(np.mean([r[k] for r in reference])) for k in reference[0]}
    # Approximate values supplied by the execution sheet; exact source/model/preprocessing identities above remain primary.
    reference_pass=abs(source_A['psnr_db']-23.46)<=.1 and abs(source_A['lpips_alex']-.099)<=.003
    assert c.b.state_sha256(vae)==read(c.b.OUT/'models.json')['vae_state_sha256']
    assert c.b.state_sha256(var)==read(c.b.OUT/'models.json')['var_state_sha256']
    assert c.b.state_sha256(decoder)==dc_sha and c.b.state_sha256(lp)==qidentity['lpips_state'] and c.b.state_sha256(dino)==qidentity['dino_state']
    c.verify_registration('A')
    seal(OUT/'A_completion.json',dict(status='REAL_ORACLE_AND_LMMSE_TASK_A_COMPLETE',synthetic=False,
        sources=100,rows=4200,cells=hashes,registration_sha256=reg,limit_check_passed=limit['passed'],
        source_A_reproduction=source_A,source_A_approximate_check_passed=reference_pass,
        source_A_tolerance=dict(psnr_db=.1,lpips_alex=.003),decoder_state_sha256=dc_sha,
        note='Task A runs independently of token consistency gates; task B still requires its own frozen calibration and exact-limit pass.',
        time=time.time()))
if __name__=='__main__':
    try:main()
    except c.b.ResourceBusy as exc:c.status('A',status='SAFE_SOURCE_PAUSE',reason=str(exc));raise SystemExit(75)
    except Exception:c.write(OUT/'A_failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
