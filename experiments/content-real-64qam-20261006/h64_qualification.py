"""Bound H qualification: new CPU packet decodes are all ledger precharged.

CLI: python -B h64_qualification.py --config qualification_config.json
This command requires a sealed registration binding the config and all sources.
No source-model inference, scientific calibration, GPU, or experiment selection.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_name(p.name+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,p)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def generator(*parts):
    import numpy as np
    data=json.dumps(['H64_QUALIFICATION_V1',2026100601,*parts],sort_keys=True).encode()
    seed=int.from_bytes(hashlib.sha256(data).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(seed))


def wilson(k,n,z=3.2905267314919255):
    require(n>0 and 0<=k<=n,'Invalid binomial counts')
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,c-w),min(1.,c+w)


def validate_config(config_path):
    cfg=read(config_path);reg=read(cfg['registration'])
    require(reg['status'] in ('REGISTERED','H_REGISTERED','CONTENT_REAL_64QAM_REGISTERED',
                             'H_EXECUTION_REVISION_REGISTERED'), 'Registration is not sealed')
    bindings=dict(reg['input_bindings']);bindings.update(reg['source_bindings'])
    require(bindings.get(str(Path(config_path).absolute()))==sha(config_path),'Qualification config is not registration-bound')
    for path,checksum in bindings.items():
        require(sha(path)==checksum,'Registered input/source changed: '+path)
    for name in ('h64_qualification.py','h64_phy.py','h64_catalog.py','h64_backend.py','h64_source.py'):
        source=str(Path(__file__).absolute().with_name(name))
        require(bindings.get(source)==sha(source),'H source absent from registration: '+source)
    budget=read(cfg['budget_registration'])
    require(budget['status']=='FROZEN' and budget['branch']=='H' and budget['total_cap']==200000,
            'Independent H budget registration is not frozen')
    limits=budget['phase_limits']
    require(limits['qualification']==2000 and sum(limits.values())<=200000,
            'Registered H qualification and total caps differ')
    require(type(limits.get('development')) is int and limits['development']>0,
            'Development reserve missing')
    require(1<=len(cfg['cpu_affinity'])<=2 and len(set(cfg['cpu_affinity']))==len(cfg['cpu_affinity']),
            'CPU qualification permits at most two cores')
    required=[cfg['ledger_module'],cfg['reference_qualification'],cfg['budget_registration'],
              str(Path(cfg['legacy_runtime'])/'ldpc_backend.py'),
              str(Path(cfg['legacy_runtime'])/'uep_phy.py'),
              str(Path(cfg['legacy_runtime'])/'uep_common.py'),
              str(Path(cfg['root'])/'src/var_comm/scale_channel.py'),
              str(Path(cfg['root'])/'src/var_comm/token_trellis.cpp')]
    for path in required:
        require(bindings.get(path)==sha(path),'Required old/new source dependency is not bound: '+path)
    return cfg,reg,sha(cfg['registration'])


def configure_cpu(cfg):
    require(sys.platform.startswith('linux'),'Actual packet qualification requires registered Linux worker')
    require(set(cfg['cpu_affinity']).issubset(os.sched_getaffinity(0)),'Requested CPU affinity unavailable')
    os.sched_setaffinity(0,set(cfg['cpu_affinity']))
    os.setpriority(os.PRIO_PROCESS,0,15)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='2'
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    require(torch.get_num_threads()==2 and torch.get_num_interop_threads()==2,'CPU thread settings differ')


def mathematical_qualification(symbols=100000):
    import numpy as np
    from h64_phy import pam,modulate,demap,integer_bits,exact_uncoded_ser,exact_ser_from_decision_regions,logsumexp
    rows=[]
    for q in (4,6):
        labels=np.stack([integer_bits(i,q) for i in range(1<<q)])
        constellation=modulate(labels,q)[:,0,:].astype(np.float64)
        require(abs(float(np.mean(np.square(constellation).sum(-1)))-2)<1e-6,'Uniform constellation energy differs from Es2')
        require(np.array_equal((demap(constellation[None],19,q)>0).astype(np.uint8).reshape(-1,q),labels),
                'Noiseless QAM bit map/demap mismatch')
        ordered=np.argsort(pam(q))
        require(all(bin(int(a)^int(b)).count('1')==1 for a,b in zip(ordered[:-1],ordered[1:])), 'PAM labels are not Gray')
        test_y=np.array([[[.13,-.47],[1.81,-1.35]]])
        metric=-.5*10**1.3*np.square(test_y[0,:,None,:]-constellation[None,:,:]).sum(-1)
        full_llr=np.stack([logsumexp(metric[:,labels[:,j]==1],-1)-logsumexp(metric[:,labels[:,j]==0],-1)
                           for j in range(q)],-1).reshape(1,-1)
        require(np.allclose(demap(test_y,13,q),full_llr,rtol=2e-6,atol=3e-6),'Component LLR differs from full QAM likelihood')
        for snr in (13,16,19):
            rng=generator('uncoded_SER',q,snr)
            sent=rng.integers(0,1<<q,symbols)
            wave=constellation[sent]
            noise=rng.standard_normal(wave.shape)*10**(-snr/20)
            received=wave+noise
            # Euclidean symbol decisions, independent of the bit-marginal LLR test.
            axis=np.argmin(np.square(received[:,:,None]-pam(q)),axis=-1)
            decoded=(axis[:,0]<<(q//2))+axis[:,1]
            errors=int(np.sum(decoded!=sent));estimate=errors/symbols
            theory=exact_uncoded_ser(snr,q)
            integrated=exact_ser_from_decision_regions(snr,q)
            tolerance=5*math.sqrt(theory*(1-theory)/symbols)+1e-3
            require(abs(theory-integrated)<1e-12,'Exact QAM formula and mapped decision regions disagree')
            require(abs(estimate-theory)<=tolerance,'Uncoded QAM SER exceeds registered five-sigma plus 1e-3 tolerance')
            rows.append(dict(q=q,snr_db=snr,symbols=symbols,errors=errors,observed_SER=estimate,
                exact_SER=theory,decision_region_SER=integrated,absolute_tolerance=tolerance,
                average_frame_Es=float(np.square(wave).sum()/symbols),mean_constellation_Es=2,
                result='PASS',theory_scope='UNIFORM_UNCODED_COHERENT_SQUARE_QAM_ONLY'))
    return rows


def physical_qualification(backend,header,ledger,buckets,boundary=lambda:None,
                           noiseless_per_layout=8,blocks_per_point=64):
    import numpy as np
    from h64_catalog import make_profile
    from h64_phy import transmit_body,receive_body,receive_header
    phase='qualification';noiseless=[];headers=[];curve=[]
    # Eight paid actual header decodes include zero and top-of-field IDs.
    header_ids=(0,1,2,127,1023,2047,4094,4095)
    public_header_test_book={str(pid):dict(qualification_only=True) for pid in header_ids}
    for pid in header_ids:
        boundary()
        wave=header.transmit(pid)
        event=receive_header(header,wave,19,public_header_test_book,ledger,phase,f'HQUAL:header:{pid}')
        require(event['header_ok'] and event['profile_id']==pid,'Noiseless paid header failed')
        headers.append(dict(profile_id=pid,accepted=True,decode_calls=1))
    for b in buckets:
        if b['admission']!='ADMITTED': continue
        # Dummy random source bits test the PHY, not arithmetic parse/model quality.
        profile=make_profile(b,6,0,'arithmetic',['QUALIFICATION_RANDOM_PAYLOAD_ONLY'])
        key=profile['profile_key'];capacity=profile['source_capacity']
        for j in range(noiseless_per_layout):
            boundary()
            rng=generator('noiseless',key,j)
            payload=(np.zeros(capacity,dtype=np.uint8) if j==0 else np.ones(capacity,dtype=np.uint8)
                     if j==1 else rng.integers(0,2,capacity,dtype=np.uint8))
            counter=100000+8*buckets.index(b)+j
            wave,meta=transmit_body(backend,payload,profile,counter,'HQUAL')
            outcome=receive_body(backend,wave,profile,30,counter,ledger,phase,f'HQUAL:noiseless:{key}:{j}','HQUAL')
            require(outcome['crc_accepted'] and outcome['parser_accepted'] and outcome['payload']==payload.tolist(),
                    'Noiseless LDPC/interleaving/payload roundtrip failed')
            noiseless.append(dict(profile_key=key,q=b['q'],nominal_rate=b['nominal_rate'],sample=j,
                actual_energy=meta['actual_energy'],k=b['k'],n=b['n'],layout_id=b['layout']['layout_id'],correct=True))
        for snr in (13,16,19):
            counts=dict(correct=0,rejected=0,undetected=0,crc_accepted=0,parser_invalid=0)
            for j in range(blocks_per_point):
                boundary()
                rng=generator('coded_curve',key,snr,j)
                payload=rng.integers(0,2,capacity,dtype=np.uint8)
                counter=200000+10000*buckets.index(b)+100*snr+j
                wave,_=transmit_body(backend,payload,profile,counter,'HQUAL')
                noisy=(wave+rng.standard_normal(wave.shape).astype(np.float32)*10**(-snr/20)).astype(np.float32)
                event=receive_body(backend,noisy,profile,snr,counter,ledger,phase,f'HQUAL:curve:{key}:{snr}:{j}','HQUAL')
                accepted=bool(event['crc_accepted'])
                correct=accepted and event['parser_accepted'] and event['payload']==payload.tolist()
                counts['correct']+=int(correct);counts['crc_accepted']+=int(accepted)
                counts['rejected']+=int(not accepted);counts['undetected']+=int(accepted and not correct)
                counts['parser_invalid']+=int(accepted and not event['parser_accepted'])
            low,high=wilson(counts['correct'],blocks_per_point)
            row=dict(profile_key=key,q=b['q'],nominal_rate=b['nominal_rate'],snr_db=snr,
                blocks=blocks_per_point,**counts,success_low_99_9=low,success_high_99_9=high,
                empirical_BLER=1-counts['correct']/blocks_per_point,
                theory_scope='NO_EXACT_LDPC_BLER_THEORY_CLAIM',weak_performance_is_implementation_failure=False)
            curve.append(row)
    # Reject only a statistically resolved wrong-direction trend, not a weak link.
    reversals=[]
    for key in sorted({r['profile_key'] for r in curve}):
        rows=sorted([r for r in curve if r['profile_key']==key],key=lambda r:r['snr_db'])
        for low,high in zip(rows[:-1],rows[1:]):
            if low['success_low_99_9']>high['success_high_99_9']:
                reversals.append(dict(profile_key=key,lower_snr=low['snr_db'],higher_snr=high['snr_db']))
    return dict(noiseless=noiseless,headers=headers,coded_curve=curve,reversals=reversals,
                actual_decode_calls=len(headers)+len(noiseless)+sum(x['blocks'] for x in curve),
                coded_trend_rule='adjacent_SNR_nonoverlap_of_99.9pct_Wilson_success_intervals',
                source_model_executed=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg,reg,registration_sha=validate_config(args.config)
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    if (out/'failure.json').exists(): raise RuntimeError('Existing qualification failure requires separately registered recovery')
    if (out/'completion.json').exists():
        done=read(out/'completion.json')
        require(done['status']=='H_PHY_QUALIFICATION_PASS' and done['registration_sha256']==registration_sha,
                'Completed qualification status/registration differs')
        require(done['budget_registration_sha256']==sha(cfg['budget_registration']),'Completed budget identity differs')
        for path,checksum in done['outputs'].items(): require(sha(path)==checksum,'Completed output changed')
        print(json.dumps(done));return
    started=time.time()
    try:
        configure_cpu(cfg)
        spec=importlib.util.spec_from_file_location('h64_registered_budget_ledger',cfg['ledger_module'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        budget_registration=read(cfg['budget_registration'])
        ledger=module.BudgetLedger(cfg['ledger'],sha(cfg['budget_registration']),'H',budget_registration['phase_limits'])
        sys.path.insert(0,cfg['legacy_runtime'])
        from uep_phy import Header
        from h64_backend import H64Backend
        from h64_catalog import all_buckets,catalogue
        backend=H64Backend(reference_qualification=read(cfg['reference_qualification']),
                            legacy_backend_path=Path(cfg['legacy_runtime'])/'ldpc_backend.py')
        buckets=all_buckets(backend);catalog=catalogue(buckets)
        write(out/'catalogue.json',catalog)
        math_rows=mathematical_qualification(100000)
        write(out/'uncoded_ser_qualification.json',dict(status='PASS',rows=math_rows))
        def boundary():
            stop_paths=[out/'STOP',out.parent/'STOP']
            if cfg.get('stop_file'): stop_paths.append(Path(cfg['stop_file']))
            require(not any(p.exists() for p in stop_paths),'STOP requested before next packet')
        physical=physical_qualification(backend,Header(cfg['root']),ledger,buckets,boundary)
        write(out/'physical_qualification.json',physical)
        require(not physical['reversals'],'Statistically resolved coded-SNR trend reversal; preserve and diagnose')
        require(any(b['q']==6 and b['admission']=='ADMITTED' for b in buckets),'No qualified 64QAM layout remains')
        budget=ledger.assert_quiescent()
        require(budget['phase_charged']['qualification']<=2000,'Qualification budget exceeded')
        outputs={str(out/name):sha(out/name) for name in ('catalogue.json','uncoded_ser_qualification.json','physical_qualification.json')}
        done=dict(status='H_PHY_QUALIFICATION_PASS',registration_sha256=registration_sha,
            budget_registration_sha256=sha(cfg['budget_registration']),
            config_sha256=sha(args.config),backend_identity=backend.identity,
            actual_decode_calls=physical['actual_decode_calls'],budget=budget,GPU_used=False,
            source_model_executed=False,science_calibration_executed=False,
            elapsed_seconds=time.time()-started,outputs=outputs,
            source_bindings=reg['source_bindings'],input_bindings=reg['input_bindings'])
        write(out/'completion.json',done);print(json.dumps(done))
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAILED',registration_sha256=registration_sha,
            error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.time()-started,automatic_retry=False))
        raise


if __name__=='__main__':main()
