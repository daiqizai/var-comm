"""Calibration-only real-weight parity and timing; no scientific M2 output."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time
import traceback

import wrapper as w
import acceleration


class NumericalMismatch(Exception):
    def __init__(self,case,field):
        self.case=case;self.field=field
        super().__init__(f'{case}: strict equality failed for {field}')


def tensor_sha(tensor):
    a=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def kv_closed(var,case):
    for index,block in enumerate(var.blocks):
        attn=block.attn
        if not hasattr(attn,'caching') or not hasattr(attn,'cached_k') or not hasattr(attn,'cached_v'):
            raise RuntimeError('Unsupported original VAR cache interface')
        if attn.caching or attn.cached_k is not None or attn.cached_v is not None:
            raise NumericalMismatch(case,f'kv_cache_closed_block{index}')


def compare(torch,original,candidate,case):
    if len(original['tokens'])!=len(candidate['tokens']):raise NumericalMismatch(case,'token_scale_count')
    for k,(a,b) in enumerate(zip(original['tokens'],candidate['tokens'])):
        if not torch.equal(a,b):raise NumericalMismatch(case,f'tokens_scale{k}')
    if not torch.equal(original['fhat'],candidate['fhat']):raise NumericalMismatch(case,'fhat')
    if original['diagnostics']!=candidate['diagnostics']:raise NumericalMismatch(case,'diagnostics')
    if original['inference']!=candidate['inference']:raise NumericalMismatch(case,'inference_label')


def timed(torch,device,fn):
    torch.cuda.synchronize(device);started=time.perf_counter()
    result=fn()
    torch.cuda.synchronize(device)
    return result,time.perf_counter()-started


def assess(rows,strict=True,threshold=1.10):
    """Aggregate interleaved all-case throughput; never drop slow cases."""
    original=sum(sum(row['original_seconds']) for row in rows)
    candidate=sum(sum(row['accelerated_seconds']) for row in rows)
    if not rows or not math.isfinite(original) or not math.isfinite(candidate) or min(original,candidate)<=0:
        raise RuntimeError('Complete positive engineering timings required')
    speedup=original/candidate
    qualified=bool(strict and speedup>=threshold)
    return dict(status='QUALIFIED' if qualified else 'USE_ORIGINAL',
        selected_implementation='accelerated' if qualified else 'original',
        speedup=speedup,strict_equality_passed=bool(strict),
        original_total_seconds=original,accelerated_total_seconds=candidate,
        minimum_speedup=threshold,decision_metric='sum legacy seconds / sum accelerated seconds; all registered cases and repeats',
        reason='strict equality and aggregate speedup>=1.10' if qualified else 'aggregate speedup below1.10')


def record_exception(receipt,error,common,out,reason='program/runtime error'):
    if w.is_resource_busy(error,common):
        waiting=dict(error=str(error),time=time.time(),exit_code=75)
        receipt.update(status='RUNNING',selected_implementation=None,resource_wait=waiting,
            reason='incomplete qualification; waiting for original safe-resource boundary')
        w.write(Path(out)/'resource_wait.json',dict(stage='benchmark',status='WAITING_FOR_SAFE_RESOURCE',**waiting))
        raise SystemExit(75) from error
    receipt.update(status='FAILED',reason=reason,error=str(error),traceback=traceback.format_exc())
    raise error


def run():
    out=w.ROOT/w.OUT_REL;out.mkdir(parents=True,exist_ok=True)
    path=out/w.QUALIFICATION;start=time.time();own=w.source_bindings()
    receipt=dict(status='RUNNING',selected_implementation=None,development_read=False,training_updates=0,
        scientific_result=False,engineering_fixture=True,source_bindings=own,
        scope='first4 original calibration sources; no development; no original M2 writes',
        statistics_sources=4,noise_seed=4101,repeats=2,latency_allcases=[],strict_equality_passed=False)
    w.write(path,receipt)
    loaded=None;common=None
    try:
        parent=w.require_m1();receipt['m1_publication_bindings']=parent
        active=w.original_gpu_workers()
        if active:raise RuntimeError('Original GPU workers still active: '+str(active))
        common,rx,runner=w.load_original()
        torch=rx.torch
        base=common.source_bindings();receipt['base_source_bindings']=base
        w.verify(own);w.verify(base);w.verify(parent)
        common.check();loaded=common.setup();common.assert_frozen(loaded)
        receipt['model_identity']=loaded['identity']
        receipt['numerical_backend']=w.numerical_backend(torch,loaded['device'])
        data=common.data('calibration',loaded)
        if len(data['records'])!=1000:raise RuntimeError('Original complete1000 calibration population required')
        receipt['calibration_bindings']=data['bindings'];receipt['calibration_population_loaded']=1000
        records=data['records'][:4];F=data['F'][:4];T=data['T'][:4]
        receipt['fixtures']=[dict(source_id=r['image_id'],source_index=i,preprocessing_id=r['preprocessing_id'],
            pixels_sha256=hashlib.sha256(r['pixels'].tobytes()).hexdigest(),
            F_sha256=tensor_sha(F[i]),T_sha256=tensor_sha(T[i])) for i,r in enumerate(records)]
        receipt['fixture_transform']='original RGB uint8 CHW256 / original frozen VAE latent+10-scale tokens; no new pixels'
        tokens=runner.split_batch(T)
        with torch.no_grad():
            prefix_all=[t.to(loaded['device']) for t in tokens[:4]]
            residual=F.to(loaded['device'])-rx.prefix_latent(loaded['vae'].quantize,prefix_all)
            pca=rx.fit_channel_pca(residual)
            static=runner.prior_tables(tokens)
            states={}
            for g,ch in ((4,8),(8,32)):
                projection=rx.Projection(g,ch,pca['axes'].to(loaded['device']))
                stats=rx.fit_structural_stats(loaded['vae'].quantize,F,tokens,projection,check=common.check,batch=4)
                operators=rx.Operators(loaded['vae'].quantize,projection,check=common.check)
                for k in range(rx.PREFIX_M,len(rx.PATCH_NUMS)):operators.get(k)
                states[projection.name]=(projection,stats,operators)
            # Engineering cache lives only in the speed output directory.
            tinycache=out/'engineering_statistics.pt'
            torch.save(dict(pca=pca,static=static,statistics={name:value[1] for name,value in states.items()},
                calibration_source_ids=[r['image_id'] for r in records],scientific_result=False),tinycache)
            receipt['engineering_cache_bindings']={str(tinycache):w.sha(tinycache)}
            receipt['manifest']=dict(format_version=1,model_identity=loaded['identity'],
                numerical_backend=receipt['numerical_backend'],calibration_bindings=data['bindings'],
                fixture_source_ids=[r['image_id'] for r in records],
                fixture_statistics_scope='PCA, mean/variance and static counts fitted on these4 sources for engineering only',
                engineering_cache_bindings=receipt['engineering_cache_bindings'],
                receiver_inputs='received prefix/measurement/projector/statistics/prior; no truth or target in infer')
            import json
            receipt['manifest_sha256']=hashlib.sha256(json.dumps(receipt['manifest'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            allcases=[]
            for i,record in enumerate(records):
                for name,(projection,stats,operators) in states.items():
                    for snr in ('clean',7):
                        for prior in ('VAR','STATIC','LIKELIHOOD','UNGUIDED'):
                            allcases.append((i,record,name,projection,stats,operators,snr,prior,1.))
            # Small predeclared sensitivity cases, independent of results.
            projection,stats,operators=states['g4_c8']
            for prior in ('VAR','STATIC'):
                for lam in (.25,2.):allcases.append((0,records[0],'g4_c8',projection,stats,operators,7,prior,lam))
            receipt['expected_cases']=len(allcases);receipt['case_grid']='4sources x2projections x(clean,7dB) x4priors +4 lambda sensitivity cases'
            for case_index,(i,record,name,projection,stats,operators,snr,prior,lam) in enumerate(allcases):
                common.check();w.verify(own)
                case=f'source{i}/{name}/{snr}/{prior}/lambda{lam}'
                prefix=[t[i:i+1].to(loaded['device']) for t in tokens[:4]]
                pre=rx.prefix_latent(loaded['vae'].quantize,prefix)
                obs,variance,info=runner.measure(F[i:i+1].to(loaded['device']),pre,projection,record,snr,4101)
                kwargs=dict(noise_variance=variance,prior=prior,lam=lam,static=static,operators=operators,check=common.check)
                def legacy():return rx.infer(loaded['vae'],loaded['var'],prefix,obs,projection,stats,**kwargs)
                def accelerated():return acceleration.infer(rx,loaded['vae'],loaded['var'],prefix,obs,projection,stats,**kwargs)
                # Warm both implementations for this exact source/profile.
                a=legacy();kv_closed(loaded['var'],case+'/warm_original')
                b=accelerated();kv_closed(loaded['var'],case+'/warm_accelerated');compare(torch,a,b,case+'/warm')
                row=dict(case=case,source_index=i,source_id=record['image_id'],projection=name,snr_db=snr,
                    prior=prior,lambda_value=lam,noise_seed=0 if snr=='clean' else 4101,
                    original_seconds=[],accelerated_seconds=[],tokens_equal=True,fhat_equal=True,diagnostics_equal=True,
                    kv_closed=True,token_sha256=[tensor_sha(t) for t in a['tokens']],fhat_sha256=tensor_sha(a['fhat']),
                    observation_sha256=tensor_sha(obs),noise_variance=variance,measurement=info)
                for repeat in range(2):
                    results={}
                    for label,fn in ([('original',legacy),('accelerated',accelerated)] if repeat==0 else
                                     [('accelerated',accelerated),('original',legacy)]):
                        value,seconds=timed(torch,loaded['device'],fn);kv_closed(loaded['var'],case+'/'+label)
                        row[label+'_seconds'].append(seconds);results[label]=value
                    compare(torch,results['original'],results['accelerated'],case+f'/repeat{repeat}')
                receipt['latency_allcases'].append(row)
                w.write(out/'benchmark_status.json',dict(status='RUNNING',cases_complete=case_index+1,
                    expected_cases=len(allcases),last_case=case,elapsed_seconds=time.time()-start))
        if len(receipt['latency_allcases'])!=receipt['expected_cases']:raise RuntimeError('Engineering case inventory incomplete')
        common.assert_frozen(loaded);kv_closed(loaded['var'],'final')
        receipt.update(assess(receipt['latency_allcases']))
        receipt.update(weights_unchanged=True,no_gradients=True,all_stage_kv_closed=True)
        w.verify(own);w.verify(base);w.verify(parent);w.verify(receipt['engineering_cache_bindings'])
    except NumericalMismatch as error:
        try:
            if loaded is not None:common.assert_frozen(loaded)
            w.verify(own);w.verify(receipt['base_source_bindings']);w.verify(receipt['m1_publication_bindings'])
        except Exception as fallback_error:
            record_exception(receipt,fallback_error,common,out,reason='fallback integrity check failed')
        receipt.update(status='USE_ORIGINAL',selected_implementation='original',strict_equality_passed=False,
            speedup=None,reason='numerical mismatch',mismatch=dict(case=error.case,field=error.field),
            weights_unchanged=True,no_gradients=True)
    except Exception as error:
        record_exception(receipt,error,common,out)
    finally:
        receipt['elapsed_seconds']=time.time()-start;w.write(path,receipt)
    return receipt


if __name__=='__main__':run()
