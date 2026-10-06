"""Registered cross-modulation shortlist and source-balanced final selection.

Pure functions only. The execution caller authenticates all population, model,
packet and score files. Development/holdout rows are rejected for selection.
"""
import math

SNRS=(1,4,7,10,13,19)
NOISE=(4101,4102,4103)
PRIMARY='dinov2_vitl14_cosine'

def require(ok,msg):
    if not ok: raise ValueError(msg)

def rank(scores):
    require(bool(scores) and all(type(v) in (float,int) and math.isfinite(v) for v in scores.values()),'Finite complete scores required')
    return sorted(scores,key=lambda k:(-scores[k],k))

def shortlist(profiles,proxy_scores):
    pool={p['candidate_id']:p for p in profiles}
    require(len(pool)==len(profiles),'Unique representative candidates required')
    require(set(proxy_scores)==set(SNRS),'All six registered SNRs required')
    cells=[]
    for snr in SNRS:
        scores=proxy_scores[snr]
        require(set(scores)==set(pool),'No missing proxy candidate may be dropped')
        w=rank({k:v for k,v in scores.items() if pool[k]['K']==0})
        p=rank({k:v for k,v in scores.items() if pool[k]['K']>0})
        require(len(w)>=3 and len(p)>=2,'Insufficient registered distinct actions')
        cells.extend([dict(family='WHOLE',snr_db=snr,candidate_ids=w[:3]),
                      dict(family='PARTIAL',snr_db=snr,candidate_ids=[w[0]]+p[:2],fixed_whole_fallback=w[0])])
    return dict(cells=cells,modulation_coverage_quota=False,selection_role='calibration',
                scope='REGISTERED_FINITE_SHORTLIST_NOT_EXACT_FULL_GRID_OPTIMUM')

def score_proxy(profiles,source_ids,gray,clean,probabilities):
    require(len(source_ids)==len(set(source_ids))==200,'Original construction200 required')
    require(set(gray)==set(source_ids) and set(clean)==set(source_ids),'Complete same-source quality required')
    result={}
    for snr in SNRS:
        ph=probabilities[('header',snr)]
        require(0<=ph<=1,'Invalid empirical header probability')
        scores={}
        for p in profiles:
            body=p['groups'][0]['phy_key'];pb=probabilities[(body,snr)]
            require(0<=pb<=1,'Invalid empirical body probability')
            state=p['m'],p['K']
            values=[]
            for sid in source_ids:
                require(state in clean[sid],'Missing exact clean state; no extrapolation')
                g,c=gray[sid],clean[sid][state]
                require(math.isfinite(g) and math.isfinite(c),'Nonfinite clean or gray quality')
                values.append(g+ph*pb*(c-g))
            scores[p['candidate_id']]=math.fsum(values)/200
        result[snr]=scores
    return result

def final_select(short,source_ids,rows):
    require(len(source_ids)==len(set(source_ids))==1000,'Original complete1000 calibration required')
    expected={(c['snr_db'],k) for c in short['cells'] for k in c['candidate_ids']}
    require(len(short['cells'])==12 and {(c['family'],c['snr_db']) for c in short['cells']}=={(f,s) for f in ('WHOLE','PARTIAL') for s in SNRS},'Frozen six-SNR family grid required')
    values={}
    for row in rows:
        require(row['population_role']=='calibration','Development/holdout must never select policies')
        point=row['snr_db'],row['candidate_id']
        i=row['source_index'];seed=row['noise_seed'];v=row[PRIMARY]
        require(type(i) is int and 0<=i<1000 and row['source_id']==source_ids[i] and point in expected and seed in NOISE,'Wrong registered score identity')
        key=(point,i,seed)
        require(key not in values and math.isfinite(v),'Duplicate/nonfinite metric')
        values[key]=float(v)
    require(len(values)==len(expected)*1000*3,'Incomplete actual calibration cannot select')
    means={p:math.fsum(math.fsum(values[p,i,n] for n in NOISE)/3 for i in range(1000))/1000 for p in expected}
    winners=[]
    for cell in short['cells']:
        s=cell['snr_db'];scores={k:means[s,k] for k in cell['candidate_ids']};winner=rank(scores)[0]
        winners.append(dict(family=cell['family'],snr_db=s,candidate_id=winner,mean_DINO_L=scores[winner],candidate_scores=scores))
    return dict(status='POLICIES_FROZEN_ON_CALIBRATION1000',objective=PRIMARY,noise_reduction='source_mean_first',
                winners=winners,development_used=False,holdout_used=False,tie_epsilon=0)
