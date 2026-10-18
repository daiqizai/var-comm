"""Finite fixed-source grouping and exact existing baseline version selection."""
import csv,hashlib,json,math,statistics
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction','dists','dreamsim','ms_ssim')
BASELINES=('EC_VAR_PARTIAL','P100000','Swin200000','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024')
DIRECTIONS={m:('lower' if m in ('lpips_alex','dists','dreamsim') else 'higher') for m in METRICS}
def group(rows,sources,grayhash):
    source={s['source_id']:s for s in sources};groups={s:{} for s in source};seen=set()
    assert len(source)==100 and len(rows)==900
    fields=['key','source_id','source_index','snr_db','candidate_id','noise_seed','target_endpoint','endpoint','actual_status','receiver_status']
    for row in rows:
        sid=row['source_id'];key=(sid,int(row['snr_db']),int(row['noise_seed']))
        assert sid in source and key not in seen and key[1] in [4,10,19] and key[2] in [6201,6202,6203];seen.add(key)
        assert row['source_index']==source[sid]['source_index']
        sha=row['image_float_sha256'] if row['image'] is not None else grayhash
        items=groups[sid]
        if sha not in items:items[sha]={'state':sha,'image':row['image'],'image_float_sha256':sha,'frames':[]}
        items[sha]['frames'].append({k:row[k] for k in fields})
    result=[]
    for s in sorted(sources,key=lambda x:x['rank']):
        pairs=list(groups[s['source_id']].values());assert sum(len(p['frames']) for p in pairs)==9
        result.append({'source_id':s['source_id'],'source_index':s['source_index'],'rank':s['rank'],'archive':s['archive'],'pairs':pairs})
    assert sum(len(s['pairs']) for s in result)<=800
    return result
def baseline_rows(paths,sources):
    indexes={int(s['source_index']):s['source_id'] for s in sources};assert len(indexes)==100
    values={}
    def add(method,snr,metric,index,value,origin):
        k=(method,int(snr),metric,int(index));assert k not in values and math.isfinite(float(value))
        values[k]={'method':method,'snr_db':int(snr),'metric':metric,'source_index':int(index),'source_id':indexes[int(index)],'source_mean':float(value),'noise_count':3,'origin':origin}
    with open(paths['main'],newline='') as f:
        for r in csv.DictReader(f):
            m,s,i=r['method'],int(r['snr_db']),int(r['source_index'])
            if m not in BASELINES or s not in [4,10,19] or i not in indexes or r['metric'] not in METRICS[:4] or (m=='EC_VAR_PARTIAL' and s==19):continue
            assert r['source_id']==indexes[i] and int(r['noise_count'])==3
            add(m,s,r['metric'],i,r['mean'],'FINAL_COMMON500_FOUR')
    with open(paths['supplement'],newline='') as f:
        for r in csv.DictReader(f):
            m,s,i=r['method'],int(r['snr_db']),int(r['source_index'])
            if m not in BASELINES or s not in [4,10,19] or i not in indexes or r['metric'] not in METRICS[4:] or (m=='EC_VAR_PARTIAL' and s==19):continue
            assert int(r['noise_count'])==3
            add(m,s,r['metric'],i,r['source_mean'],'FINAL_COMMON500_SUPPLEMENT')
    with open(paths['t1c'],newline='') as f:
        for r in csv.DictReader(f):
            i=int(r['source_index'])
            if r['population']!='common500' or r['family']!='PARTIAL' or i not in indexes:continue
            assert r['metric'] in METRICS
            add('EC_VAR_PARTIAL',19,r['metric'],i,r['new_mean'],'T1C_M10_REVISED_19DB')
    expected={(m,s,k,i) for m in BASELINES for s in [4,10,19] for k in METRICS for i in indexes}
    assert set(values)==expected and len(values)==8400
    return list(values.values())
def means(rows,population):
    groups={};seen=set()
    for r in rows:
        key=(r['source_id'],int(r['snr_db']),int(r['noise_seed']))
        assert key not in seen and key[0] in population and key[1] in [4,10,19] and key[2] in [6201,6202,6203];seen.add(key)
        groups.setdefault(key[:2],[]).append(r)
    result=[]
    for sid in sorted(population):
        for snr in [4,10,19]:
            group=groups[sid,snr];assert len(group)==3 and {int(r['noise_seed']) for r in group}=={6201,6202,6203}
            for m in METRICS:
                v=statistics.mean(float(r[m]) for r in group);assert math.isfinite(v)
                result.append({'method':'ARPC_ADAPTED','source_id':sid,'source_index':int(group[0]['source_index']),'snr_db':snr,'metric':m,'source_mean':v,'noise_count':3,'origin':'ACTUAL_ARPC_FIXED_EVALUATION'})
    assert len(seen)==900 and len(result)==2100
    return result
