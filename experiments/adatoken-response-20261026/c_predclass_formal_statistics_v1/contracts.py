"""Metadata gates and baseline normalization; no scientific computation."""
import math

METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
SNRS=(4,10,19)
COUNTS={'common500':500,'kodak24':24}
BASELINE_METHODS={'common500':{'EC_VAR_PARTIAL','P100000','Swin200000','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024','ARPC_ADAPTED'},
 'kodak24':{'NEST_PARTIAL','NEST_WHOLE','P100000','Swin200000','BPG_ADAPTIVE','ARPC_ADAPTED'}}


def check_closed(wait,done,registration_descriptor,registration,role,gone):
    assert wait['actual_wait']is True and type(wait['returncode'])is int and wait['returncode']==0
    assert all(wait[k]is False for k in('timed_out','terminated','killed'))and wait['metadata_error']is None
    assert wait['registration']==registration_descriptor
    assert wait['argv']==registration['stages'][role]['argv']+['--registration',registration_descriptor['path'],'--expected-registration',registration_descriptor['sha256']]
    assert gone(wait['process'])and gone(wait['parent_process'])
    assert done['status']=='COMPLETE'and done['registration_sha256']==registration_descriptor['sha256']
    assert done['population']==registration['population']and done['rows']==COUNTS[done['population']]*9
    limits=registration['limits'][role];seen=set()
    for kind,status,count in done['counts']:
        assert kind not in seen and kind in limits and status=='COMPLETE'
        assert type(count)is int and 0<count<=limits[kind];seen.add(kind)
    kind=registration['metric_kinds'][role]
    assert kind in('three','clip')
    required=('load_lpips','load_dreamsim')if kind=='three'else('load_clip',)
    assert all([k,'COMPLETE',1]in done['counts']for k in required)
    assert done['new_reconstruction']==done['new_PHY']==0


def sources_match(plans,population,historical_population,historical_frames):
    count=COUNTS[population];t,c=plans
    assert t['population']==c['population']==population
    assert t['frozen_policy']==c['frozen_policy']
    assert t['sources']==c['sources']
    source=sorted(t['sources'],key=lambda s:s['source_index'])
    assert len(source)==count and [s['source_index']for s in source]==list(range(count))
    assert len({s['source_id']for s in source})==count
    original=historical_population['common500'if population=='common500'else'sources']
    original=sorted(original,key=lambda s:s['source_index'])
    assert len(original)==count and [s['source_index']for s in original]==list(range(count))
    proof=[]
    for actual,old in zip(source,original):
        assert actual['source_id']==old['source_id']
        key='preprocessing_id'if population=='common500'else'pixels_chw_uint8_sha256'
        assert actual['preprocessing_id']==old[key]
        if population=='common500':assert actual['archive']==old['archive']
        proof.append(dict(source_index=actual['source_index'],source_id=actual['source_id'],
          uint8_pixels_sha256=actual['preprocessing_id'],reference_float_sha256=actual['reference_sha256']))
    method='EC_VAR_PARTIAL'if population=='common500'else'NEST_PARTIAL'
    ref_field='reference_sha256'if population=='common500'else'reference_array_sha256'
    rows=[r for r in historical_frames if r['method']==method]
    assert len(rows)==count*9
    seen=set();noise_sets={}
    for row in rows:
        i,snr,noise=int(row['source_index']),int(row['snr_db']),int(row['noise_seed'])
        assert i in range(count)and snr in SNRS
        assert row['source_id']==source[i]['source_id']and row[ref_field]==source[i]['reference_sha256']
        if population=='kodak24':assert row['array_hash_convention']=='dtype_shape_cbytes'
        else:assert int(row['N'])==1024
        key=i,snr,noise;assert key not in seen;seen.add(key)
        noise_sets.setdefault((i,snr),set()).add(noise)
    assert set(noise_sets)=={(i,s)for i in range(count)for s in SNRS}
    assert all(len(v)==3 for v in noise_sets.values())
    # The historical NeST noise namespace need not equal Ada's namespace.
    return source,proof


def normalize_nest(population,sources,source_tables):
    count=COUNTS[population];result=[]
    for table in source_tables:
        for r in table:
            metric=r['metric']
            if population=='common500'and 'family'in r:
                if r['population']!='common500'or r['family']!='PARTIAL'or metric not in METRICS[:3]:continue
                mean=r['source_mean'];noise=3
            elif population=='common500':
                if r['method']!='EC_VAR_PARTIAL'or metric!='clip_image_cosine':continue
                mean=r['source_mean'];noise=int(r['noise_count'])
            else:
                if r['method']!='NEST_PARTIAL'or metric not in METRICS:continue
                mean=r['mean'];noise=int(r['noise_count'])
            i,s=int(r['source_index']),int(r['snr_db']);assert i in range(count)and s in SNRS and noise==3
            if 'source_id'in r:assert r['source_id']==sources[i]['source_id']
            assert math.isfinite(float(mean))
            result.append(dict(population=population,method='NEST_PARTIAL',source_index=i,source_id=sources[i]['source_id'],
              snr_db=s,metric=metric,source_mean=float(mean),noise_count=3))
    keys=[(r['source_index'],r['snr_db'],r['metric'])for r in result]
    assert len(keys)==count*12 and len(set(keys))==len(keys)
    assert set(keys)=={(i,s,m)for i in range(count)for s in SNRS for m in METRICS}
    return result


def score_plan_match(scores,plan,metric_kind):
    key=lambda v:(int(v['source_index']),int(v['snr_db']),int(v['noise_seed']))
    frames=plan['frames'];index={key(v):v for v in frames}
    assert len(index)==len(frames)==len(scores)
    assert len({key(v)for v in scores})==len(scores)and set(index)=={key(v)for v in scores}
    fields=('source_id','population','method','received_prefix','wire_status','source_status','accepted','pixel_sha256')
    for s in scores:
        v=index[key(s)]
        for name in fields:assert str(s[name])==str(v[name]),name
        if v['reuse_old']:
            assert metric_kind=='three'and s['score_origin']=='EXACT_FROZEN_SOURCE_PIXEL_PAIR_REUSE'
            for m in METRICS[:3]:assert float(s[m])==float(v['old_values'][m])
        else:assert s['score_origin']in('NEW_NATIVE_PAIR','EXACT_PAIR_REUSE')


def baseline_summary(population,rows):
    methods=BASELINE_METHODS[population];count=COUNTS[population];result=[]
    for r in rows:
        if r['metric']not in METRICS:continue
        assert r['method']in methods and int(r['source_count'])==count
        direction=r['direction'if population=='common500'else'favorable_direction']
        assert direction==('higher'if r['metric']in('psnr_db','clip_image_cosine')else'lower')
        # Carry full original decimal strings through export. No old CI rerun.
        for k in('mean','ci_low','ci_high'):assert math.isfinite(float(r[k]))
        assert float(r['ci_low'])<=float(r['ci_high'])
        result.append(dict(population=population,method=r['method'],snr_db=int(r['snr_db']),metric=r['metric'],
          direction=direction,source_count=count,logical_frames=count*3,mean=r['mean'],ci_low=r['ci_low'],ci_high=r['ci_high'],
          statistics_origin='EXISTING_FROZEN_SUMMARY'))
    keys=[(r['method'],r['snr_db'],r['metric'])for r in result]
    assert len(keys)==len(set(keys))and set(keys)=={(m,s,k)for m in methods for s in SNRS for k in METRICS}
    return result
