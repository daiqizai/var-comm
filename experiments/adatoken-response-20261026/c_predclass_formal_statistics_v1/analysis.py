"""Formal source-paired statistics only; no model, image, PHY or random draws."""
import hashlib
import numpy as np

METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
SNRS=(4,10,19)
POPULATIONS={'common500':(500,(6201,6202,6203)),'kodak24':(24,(2001,2002,2003))}
DIRECTION={'psnr_db':'higher','lpips_alex':'lower','dreamsim':'lower','clip_image_cosine':'higher'}
BANK_RAW_SHA={'common500':'0c33c7302272b4c97c41ee96c5a66dae95194c2f1dc58a3fc50e32d60b42a0f7',
 'kodak24':'552e5b5aebdfa237ac3a72f9c1329dbdcff67335f4f0ee1fa9d1dc0275ed42d8'}


# Verbatim existing b_statistics.interval, checked by AST in qualification.
def interval(v,bank):
 valid=np.isfinite(v);samples=v[bank];counts=np.isfinite(samples).sum(axis=1);assert np.all(counts>0)
 draws=np.nansum(samples,axis=1)/counts;lo,hi=np.quantile(draws,[.025,.975],method='linear');return dict(mean=float(np.mean(v[valid])),ci_low=float(lo),ci_high=float(hi),source_count=int(valid.sum()))


def source_order(sources,population):
    count,_=POPULATIONS[population]
    assert len(sources)==count and {s['source_index']for s in sources}==set(range(count))
    ordered=sorted(sources,key=lambda s:s['source_index'])
    assert len({s['source_id']for s in ordered})==count
    return ordered


def validate_bank(bank,population):
    count,_=POPULATIONS[population]
    assert bank.dtype==np.int64 and bank.shape==(10000,count)
    assert int(bank.min())>=0 and int(bank.max())<count
    assert hashlib.sha256(bank.tobytes(order='C')).hexdigest()==BANK_RAW_SHA[population]


def frame_key(v):
    return int(v['source_index']),int(v['snr_db']),int(v['noise_seed'])


def merge_scores(three,clip,sources,population):
    count,noises=POPULATIONS[population];ordered=source_order(sources,population)
    expected={(i,s,n)for i in range(count)for s in SNRS for n in noises}
    by_kind=[]
    for rows in(three,clip):
        keys=[frame_key(v)for v in rows]
        assert len(keys)==count*9 and len(set(keys))==len(keys)and set(keys)==expected
        by_kind.append({frame_key(v):v for v in rows})
    result=[]
    for key in sorted(expected):
        t,c=(table[key]for table in by_kind);i,s,n=key
        for v in(t,c):
            assert v['source_id']==ordered[i]['source_id']and v['population']==population
            assert v['method']=='ADA_PREDCLASS_FIXEDFRAME'
            assert v['source_status']in('ACCEPTED_SOURCE','FIXED_GRAY_SOURCE_REJECT','FIXED_GRAY_PHY_REJECT')
            assert str(v['accepted'])in('True','False')
            assert (str(v['accepted'])=='True')==(v['source_status']=='ACCEPTED_SOURCE')
            assert 0<=int(v['received_prefix'])<=256 and len(v['pixel_sha256'])==64
        for k in('source_id','population','method','received_prefix','wire_status','source_status','accepted','pixel_sha256'):
            assert str(t[k])==str(c[k]),'Metric files refer to different actual outputs: '+k
        values={m:float(t[m])for m in METRICS[:3]}
        values['clip_image_cosine']=float(c['clip_image_cosine'])
        assert all(np.isfinite(v)for v in values.values())
        result.append(dict(population=population,source_index=i,source_id=t['source_id'],snr_db=s,noise_seed=n,
          method='ADA_PREDCLASS_FIXEDFRAME',received_prefix=int(t['received_prefix']),wire_status=t['wire_status'],
          source_status=t['source_status'],accepted=str(t['accepted'])=='True',pixel_sha256=t['pixel_sha256'],
          three_score_origin=t['score_origin'],clip_score_origin=c['score_origin'],**values))
    return result


def summarize(frames,sources,nest_source_means,bank,population):
    validate_bank(bank,population);count,noises=POPULATIONS[population];ordered=source_order(sources,population)
    expected={(i,s,n)for i in range(count)for s in SNRS for n in noises}
    assert len(frames)==count*9 and {frame_key(v)for v in frames}==expected
    fi={frame_key(v):v for v in frames};ni={}
    for row in nest_source_means:
        i,s,m=int(row['source_index']),int(row['snr_db']),row['metric'];key=i,s,m
        assert key not in ni and i in range(count)and s in SNRS and m in METRICS
        assert row['source_id']==ordered[i]['source_id']and row['population']==population
        assert row['method']=='NEST_PARTIAL'and int(row['noise_count'])==3
        value=float(row['source_mean']);assert np.isfinite(value);ni[key]=value
    assert set(ni)=={(i,s,m)for i in range(count)for s in SNRS for m in METRICS}
    means=[];summary=[];paired=[];pair_source=[]
    for s in SNRS:
        for m in METRICS:
            values=[];reference=[]
            for i in range(count):
                sub=[fi[i,s,n]for n in noises]
                assert all(v['source_id']==ordered[i]['source_id']for v in sub)
                raw=np.asarray([float(v[m])for v in sub],np.float64);assert np.isfinite(raw).all()
                value=float(np.mean(raw));values.append(value);reference.append(ni[i,s,m])
                means.append(dict(population=population,method='ADA_PREDCLASS_FIXEDFRAME',source_index=i,
                  source_id=ordered[i]['source_id'],snr_db=s,metric=m,source_mean=value,noise_count=3))
            a=np.asarray(values,np.float64);b=np.asarray(reference,np.float64);delta=b-a
            summary.append(dict(population=population,method='ADA_PREDCLASS_FIXEDFRAME',snr_db=s,metric=m,
              direction=DIRECTION[m],logical_frames=count*3,**interval(a,bank)))
            paired.append(dict(population=population,method='NEST_PARTIAL',reference='ADA_PREDCLASS_FIXEDFRAME',
              comparison='NEST_minus_ADA',snr_db=s,metric=m,delta_definition='NeST minus Ada',
              favorable_delta='positive'if DIRECTION[m]=='higher'else'negative',pairing_unit='source image',**interval(delta,bank)))
            for i,d in enumerate(delta):
                pair_source.append(dict(population=population,source_index=i,source_id=ordered[i]['source_id'],snr_db=s,
                  metric=m,nest_source_mean=float(b[i]),ada_source_mean=float(a[i]),nest_minus_ada=float(d)))
    acceptance=[]
    for s in SNRS:
        counts={}
        for v in frames:
            if v['snr_db']!=s:continue
            k=v['wire_status'],v['source_status'];counts[k]=counts.get(k,0)+1
        assert sum(counts.values())==count*3
        for (wire,status),n in sorted(counts.items()):
            acceptance.append(dict(population=population,snr_db=s,wire_status=wire,source_status=status,
              logical_frames=n,total_logical_frames=count*3))
    assert len(summary)==len(paired)==12 and len(means)==len(pair_source)==count*12
    return dict(source_means=means,summary=summary,paired_differences=paired,paired_source_values=pair_source,acceptance=acceptance)


def capacity_summary(choices,sources,population):
    count,_=POPULATIONS[population];ordered=source_order(sources,population)
    keys=[(int(v['source_index']),int(v['snr_db']))for v in choices]
    assert len(keys)==count*3 and len(set(keys))==len(keys)and set(keys)=={(i,s)for i in range(count)for s in SNRS}
    for v in choices:
        assert v['source_id']==ordered[v['source_index']]['source_id']and v['population']==population
        assert 0<=v['prefix']<=256 and v['explicit_null']==(v['prefix']==0)
        assert 0<=v['arithmetic_bits']<=v['arithmetic_capacity_bits']and v['arithmetic_capacity_bits']>0
        assert v['paid_class_bits']==(0 if v['explicit_null']else 10)
        assert v['paper32_prefix']in range(0,257,32)and v['paper32_prefix']<=v['prefix']
        assert 0<=v['paper32_arithmetic_bits']<=v['arithmetic_capacity_bits']
        assert v['container_bits']==(2 if v['explicit_null']else v['arithmetic_bits']+10)
    summary=[];distribution=[]
    for s in SNRS:
        rows=[v for v in choices if v['snr_db']==s];counts={}
        for v in rows:counts[v['prefix']]=counts.get(v['prefix'],0)+1
        for prefix,n in sorted(counts.items()):
            distribution.append(dict(population=population,snr_db=s,transmitted_prefix=prefix,source_count=n,logical_frames=n*3,total_sources=count))
        summary.append(dict(population=population,snr_db=s,source_count=count,logical_frames=count*3,
          mean_transmitted_tokens=float(np.mean([v['prefix']for v in rows])),
          mean_arithmetic_utilization=float(np.mean([v['arithmetic_bits']/v['arithmetic_capacity_bits']for v in rows])),
          mean_terminated_arithmetic_bits=float(np.mean([v['arithmetic_bits']for v in rows])),
          mean_container_bits=float(np.mean([v['container_bits']for v in rows])),
          mean_paid_class_bits=float(np.mean([v['paid_class_bits']for v in rows])),
          mean_descending_length_checks=float(np.mean([v['length_checks']for v in rows])),
          mean_paper32_tokens=float(np.mean([v['paper32_prefix']for v in rows])),
          mean_paper32_arithmetic_utilization=float(np.mean([v['paper32_arithmetic_bits']/v['arithmetic_capacity_bits']for v in rows])),
          explicit_null_sources=sum(v['explicit_null']for v in rows),length_field_bits=13,body_CRC_bits=16,
          protected_header_symbols=68,body_symbols=956,total_complex_symbols=1024,
          paper32_role='Capacity diagnostic only; no paper32 quality inference or policy reselection'))
    return summary,distribution
