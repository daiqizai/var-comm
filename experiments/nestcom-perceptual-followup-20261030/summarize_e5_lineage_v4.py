"""Summarize completed, direct boundary timings under the exact v8 role map.

No model, PHY, timing replay, bootstrap or historical component summation.
"""
from followup_common import *
from collections import defaultdict
import math, traceback


def closure(role):
    control = L/(role+'_control')
    wait = read(control/'actual_wait.json')
    owner = read(control/'owner.json')['owner']
    assert wait['returncode'] == 0
    assert ident(wait['child']['pid']) is None and ident(owner['pid']) is None
    assert not (L/role/'failure.json').exists()
    result = closed(role)
    return dict(role=role, actual_wait=wait, owner_gone=True, child_gone=True,
                registration=pin(D/(role+'_registration.json')),
                completion=pin(L/role/'completion.json')), result


def aggregate(group, ids, joined):
    data = defaultdict(list)
    assert set(joined) == {(sid, rep) for sid in ids for rep in range(5)}
    for (sid, rep), pair in joined.items():
        assert set(pair) == {'tx', 'rx'}
        phase = 'warmup' if rep < 2 else 'measured'
        assert pair['tx']['phase'] == pair['rx']['phase'] == phase
        assert isinstance(pair['rx']['success'], bool)
        if phase == 'warmup':
            continue
        data[sid].append(dict(TX=pair['tx']['seconds'], RX=pair['rx']['seconds'],
                             total=pair['tx']['seconds']+pair['rx']['seconds'],
                             success=pair['rx']['success']))
    assert set(data) == set(ids) and all(len(v) == 3 for v in data.values())
    per_source = [dict(group=group['name'], source_id=sid,
                       **{d:float(np.mean([x[d] for x in data[sid]])) for d in ['TX','RX','total']},
                       successes=sum(x['success'] for x in data[sid]), measured_calls=3) for sid in ids]
    result = []
    for direction in ['TX', 'RX', 'total']:
        vals = [x[direction] for sid in ids for x in data[sid]]
        successful = [x[direction] for sid in ids for x in data[sid] if x['success']]
        result.append(dict(group=group['name'],method=group['method'],N=group['N'],snr_db=group['snr_db'],
            grid=group.get('grid',''),direction=direction,sources=len(ids),measured=len(vals),warmup=2*len(ids),
            equal_source_mean_seconds=float(np.mean([v[direction] for v in per_source])),
            call_median_seconds=float(np.median(vals)),call_p95_seconds=float(np.quantile(vals,.95,method='linear')),
            successful_RX_calls=len(successful),failed_RX_calls=len(vals)-len(successful),
            success_conditional_RX_mean_seconds=float(np.mean(successful)) if direction=='RX' and successful else '',
            all_failed_in_denominator=True))
    return per_source, result


def main(request, expected):
    assert sha(request) == expected
    r=read(request);[checked(v) for v in r['code']]
    os.sched_setaffinity(0,set(r['cpu_affinity']))
    output=Path(r['output']);output.mkdir(exist_ok=False)
    try:
        control_evidence, _ = closure('E5_last_two_v1')
        plan=Path(r['plan']);cfg=cr(r['config']);mapping=cr(r['role_map'])
        groups=cr(r['groups']);sources=cr(r['sources']);ids=[s['source_id'] for s in sources]
        assert len(groups)==16 and len(ids)==len(set(ids))==32 and cfg['source_order']==ids
        assert len(mapping)==32 and len({(v['group_index'],v['direction']) for v in mapping})==32
        assert cfg['cpu_cores']==2 and cfg['batch']==1 and cfg['warmups']==2 and cfg['measurements']==3
        allrows=[];summary=[];source_means=[];evidence=[];calls=[];fit=[]
        for gi,g in enumerate(groups):
            joined={}
            for direction in ['tx','rx']:
                role=next(v['role'] for v in mapping if v['group_index']==gi and v['direction']==direction)
                prefix='E5_arpc3' if gi==9 else 'E5_bpg2' if gi==12 else 'E5_dep5'
                assert role==f'{prefix}_g{gi:02d}_{direction}'
                proof,done=closure(role);reg=read(D/(role+'_registration.json'))
                assert reg['config']==r['config'] and reg['sources']==r['sources'] and reg['group']==g
                assert done['rows']==160 and done['warmups']==64 and done['measured']==96
                assert done['ledger']['counts']==[dict(kind='timing_trial',status='COMPLETE',count=160)]
                assert done['ledger']['unresolved']==0 and done['cached_scientific_calls']==0
                env=read(L/role/'environment.json')
                assert env['cpu_affinity']==cfg['cpu_affinity'] and env['batch']==1
                assert env['threads']==6 and env['interop']==2 and env['registered_UUID']==cfg['GPU_UUID']
                rr=read(L/role/'rows.json');assert len(rr)==160
                assert [(v['source_id'],v['repetition']) for v in rr]==[(sid,rep) for sid in ids for rep in range(5)]
                for v in rr:
                    assert v['group']==g['name'] and v['direction']==direction
                    assert math.isfinite(v['seconds']) and v['seconds']>0
                    allrows.append(v);key=(v['source_id'],v['repetition'])
                    pair=joined.setdefault(key,{})
                    assert direction not in pair;pair[direction]=v
                    if direction=='tx' and g['method'] in ['WHOLE','Token_NeST','Ada']:
                        witness=read(L/role/'witnesses'/(v['event']+'.json'))
                        nc=256 if g['method']=='Ada' else len(g['endpoints'])
                        assert witness['terminated_candidates']==nc
                        cdf_steps=256 if g['method']=='Ada' else max(m+int(k>0) for m,k in g['endpoints'])
                        fit.append(dict(group=g['name'],source_id=v['source_id'],repetition=v['repetition'],phase=v['phase'],
                            fresh_source_probability_traversals=1,cdf_construction_steps=cdf_steps,
                            independently_terminated_candidates=nc,actual_candidate_length_tests=nc,
                            selected_endpoint_descending_rank=witness['length_checks'],endpoint=str(witness['endpoint']),
                            count_basis='audited bound code plus per-trial witness; full fits-list scan, not early stopping'))
                if direction=='rx':
                    txrole=next(v['role'] for v in mapping if v['group_index']==gi and v['direction']=='tx')
                    assert reg['received_symbols']==pin(L/txrole/'received_symbols.json')
                if gi==9:
                    service=read(L/role/'phy_service_actual_wait.json')
                    assert service['returncode']==0 and service['final_identity'] is None
                    assert service['report']['result'][direction.upper()]==160
                    proof['PHY_service_actual_wait']=service
                proof.update(rows=pin(L/role/'rows.json'),environment=env,ledger=done['ledger'])
                evidence.append(proof)
                for kind,count in done['in_memory_calls'].items():
                    calls.append(dict(group=g['name'],direction=direction,kind=kind,count=count,
                                      includes_warmups=True,timing_trials=160,counter_scope='worker-stage instrumented calls; not exhaustive operator count'))
            src,stats=aggregate(g,ids,joined);source_means.extend(src);summary.extend(stats)
        assert len(allrows)==5120 and len(source_means)==512 and len(summary)==48
        # Write only after every input and every actual process closure passes.
        dest=D/'e5';dest.mkdir(exist_ok=False)
        for name,data in [('call_timings.csv',allrows),('source_means.csv',source_means),('summary.csv',summary),
                          ('model_call_accounting.csv',calls),('source_fitting_accounting.csv',fit)]:
            csvout(dest/name,data)
        save(dest/'execution_evidence.json',dict(controller=control_evidence,stages=evidence))
        save(dest/'identity.json',dict(role_map=r['role_map'],config=r['config'],sources=r['sources'],groups=r['groups'],
            source_order=ids,groups_count=16,stages=32,trials=5120,warmup=2048,measured=3072,
            matched_token_group='Token_NeST_N1024_S10',CPU_cores=2,math_threads=6,batch=1,
            directions_measured_directly=True,total_definition='paired TX wall + RX wall; no channel generation or propagation',
            source_cache_used_in_timing=False,historical_latency_rows=0,original_failures_preserved=True,
            original_deadline_unchanged=True,extension_scope=r['extension_scope'],
            timing_repeats_use_same_registered_noise=True,summary_uses_no_quality_results=True))
        save(output/'completion.json',dict(status='COMPLETE',rows=5120,groups=16,summary=pin(dest/'summary.csv'),
             new_model_calls=0,new_PHY_calls=0,new_timing_calls=0,summary_calls=1))
        print('E5_ACTUAL_TIMING_COMPLETE',5120,48,flush=True)
    except BaseException:
        save(output/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main(*sys.argv[1:])
