from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from var_comm.study import paired_interval

ROOT=Path(__file__).resolve().parents[5]; VAR=ROOT; BASE=VAR/'outputs/VAR-LATENT-ENHANCEMENT-20260917'; FIXED=BASE/'development_eval_v1'; POLICY=BASE/'followup/digital_policy_v2/policies.json'; OUT=BASE/'followup/digital_policy_development_v1'

def read(path):
 with Path(path).open(newline='',encoding='utf-8') as h:return list(csv.DictReader(h))
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r));
 with path.open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def _value(row, *names, default=None):
 for name in names:
  if name in row and row[name] not in ('', None):
   return row[name]
 return default

def _source_key(row):
 value=_value(row,'source_index','source_id','source')
 if value is None: raise KeyError('row has no source identity')
 return int(value)

def _snr_key(row):
 value=_value(row,'snr_db','snr')
 if value is None: raise KeyError('row has no SNR identity')
 return float(value)

def _seed_key(row):
 # Keep the full noise/seed identity while joining rows.  A missing seed is
 # an error: silently collapsing it recreates the old last-row-wins bug.
 value=_value(row,'seed','noise_seed','noise_id')
 if value is None: raise KeyError('row has no seed/noise identity')
 return str(value)

def _preprocess_key(row):
 return str(_value(row,'preprocessing_id','preprocess_id','preprocessing',default=''))

def _mean_metric(rows, name):
 values=[float(r[name]) for r in rows]
 if not values: raise ValueError(f'empty metric group: {name}')
 return float(np.mean(values))

def _aggregate_source_snr(rows):
 """Aggregate seed rows only after checking complete pairing dimensions."""
 by_seed={}
 for row in rows:
  key=(_source_key(row),_snr_key(row),_seed_key(row),_preprocess_key(row))
  by_seed.setdefault(key,[]).append(row)
 # A complete source/SNR/seed/preprocessing identity must occur exactly once.
 # Averaging duplicate rows hides a broken join and can silently change the
 # paired estimate, so reject both exact and conflicting duplicates.
 seed_rows=[]
 for (source,snr,seed,preprocess),values in sorted(by_seed.items()):
  if len(values)!=1:
   raise RuntimeError(f'duplicate complete pairing key: source={source}, snr={snr}, seed={seed}, preprocessing={preprocess}, rows={len(values)}')
  row=dict(values[0]);row.update({'source_index':source,'snr_db':snr,'seed':seed,'preprocessing_id':preprocess})
  seed_rows.append(row)
 by_pair={}
 for row in seed_rows:
  by_pair.setdefault((_source_key(row),_snr_key(row),_preprocess_key(row)),[]).append(row)
 result=[]
 for (source,snr,preprocess),values in sorted(by_pair.items()):
  if len(values)!=3: raise RuntimeError(f'incomplete seed dimension for source={source}, snr={snr}, preprocessing={preprocess}: {len(values)}')
  row=dict(values[0]);row.update({'source_index':source,'snr_db':snr,'preprocessing_id':preprocess,'seed_count':len(values),
                                   'seed_keys':','.join(sorted(str(_seed_key(v)) for v in values))})
  for name in ('psnr_db','lpips_alex','dino_cosine'):
   row[name]=_mean_metric(values,name)
  result.append(row)
 return result
def main():
 policies=json.loads(POLICY.read_text());fixed=read(FIXED/'per_frame.csv');out=OUT;out.mkdir(parents=True,exist_ok=True);selected=[]
 for family in ('raw','arithmetic'):
  for budget in (3572,4084):
   for renderer in ('D0','Dc'):
    for rule in ('reliability','quality','goodput'):
     rows=[]
     for row in fixed:
      row_renderer='Dc' if row['method'].endswith('_Dc') else 'D0'
      if row['family']!=family or int(row['N'])!=budget or row_renderer!=renderer:continue
      mode=int(policies['actions'][family][str(budget)][renderer][str(float(row['snr_db']))][rule])
      if int(row['sent_mode'])!=mode:continue
      copy=dict(row);copy.update({'renderer':renderer,'policy':rule,'selected_mode':mode});rows.append(copy)
     if len(rows)!=1500:raise RuntimeError(f'incomplete selected policy {family} {budget} {renderer} {rule}: {len(rows)}')
     selected.extend(rows)
 write(OUT/'per_frame.csv',selected)
 # Preserve source/SNR/seed/preprocessing dimensions before averaging seeds.
 source_seed={}
 for row in selected:
  key=(row['family'],int(row['N']),row['renderer'],row['policy'],_source_key(row),_snr_key(row),_seed_key(row),_preprocess_key(row))
  source_seed.setdefault(key,[]).append(row)
 source_snr={}
 for key,values in source_seed.items():
  family,budget,renderer,rule,source,snr,seed,preprocess=key
  source_snr.setdefault((family,budget,renderer,rule,source,snr,preprocess),[]).append(values[0] if len(values)==1 else {**values[0],**{name:_mean_metric(values,name) for name in ('psnr_db','lpips_alex','dino_cosine')}})
 summary=[]
 for key,values in sorted(source_snr.items()):
  family,budget,renderer,rule,source,snr,preprocess=key
  if len(values)!=3: raise RuntimeError(f'incomplete selected seed dimension for {family}/{budget}/{renderer}/{rule}/{source}/{snr}: {len(values)}')
  summary.append({'family':family,'budget':budget,'renderer':renderer,'policy':rule,'source_index':source,'snr_db':snr,
                  'preprocessing_id':preprocess,'seed_count':len(values),'seed_keys':','.join(sorted(str(_seed_key(r)) for r in values)),
                  'psnr_db':_mean_metric(values,'psnr_db'),'lpips_alex':_mean_metric(values,'lpips_alex'),'dino_cosine':_mean_metric(values,'dino_cosine'),'selected_mode':values[0]['selected_mode']})
 write(OUT/'per_source_snr.csv',summary)
 overall=[]
 for family in ('raw','arithmetic'):
  for budget in (3572,4084):
   for renderer in ('D0','Dc'):
    for rule in ('reliability','quality','goodput'):
     vals=[r for r in summary if r['family']==family and int(r['budget'])==budget and r['renderer']==renderer and r['policy']==rule]
     overall.append({'family':family,'budget':budget,'renderer':renderer,'policy':rule,'psnr_db':float(np.mean([float(r['psnr_db']) for r in vals])),'lpips_alex':float(np.mean([float(r['lpips_alex']) for r in vals])),'dino_cosine':float(np.mean([float(r['dino_cosine']) for r in vals]))})
 write(OUT/'summary.csv',overall)
 # paired vs corresponding B fixed methods
 pairs=[]
 for row in overall:
  b='m8_plus_latent_512' if row['budget']==3572 else 'm8_plus_latent_1024'; fixed_b=[r for r in fixed if r['method']==b]
  a=np.array([[float(x['psnr_db']),float(x['lpips_alex']),float(x['dino_cosine'])] for x in summary if x['family']==row['family'] and int(x['budget'])==row['budget'] and x['renderer']==row['renderer'] and x['policy']==row['policy']])
  bsummary=_aggregate_source_snr(fixed_b)
  bvals={(int(x['source_index']),float(x['snr_db']),str(x.get('preprocessing_id',''))):x for x in bsummary}
  expected_preprocess=sorted({str(x.get('preprocessing_id','')) for x in bsummary})
  if len(expected_preprocess)!=1: raise RuntimeError(f'fixed reference mixes preprocessing identities: {expected_preprocess}')
  preprocess=expected_preprocess[0]
  bv=np.array([[float(bvals[(i,s,preprocess)]['psnr_db']),float(bvals[(i,s,preprocess)]['lpips_alex']),float(bvals[(i,s,preprocess)]['dino_cosine'])] for i in range(100) for s in [1,4,7,13,19]])
  # a is source x snr; same order from sorted summary
  av=np.array([[float(x['psnr_db']),float(x['lpips_alex']),float(x['dino_cosine'])] for x in sorted([z for z in summary if z['family']==row['family'] and int(z['budget'])==row['budget'] and z['renderer']==row['renderer'] and z['policy']==row['policy']],key=lambda z:(int(z['source_index']),float(z['snr_db'])))])
  row2=dict(row)
  for j,name in enumerate(('delta_psnr','delta_lpips','delta_dino')):row2[name]=paired_interval((av[:,j]-bv[:,j]).reshape(100,5).mean(1),2026091800+j,10000)
  pairs.append(row2)
 (OUT/'paired_vs_enhancement.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2)+'\n')
 (OUT/'completion.json').write_text(json.dumps({'status':'FROZEN_DIGITAL_POLICY_DEVELOPMENT_COMPLETE','rows':len(selected),'policies':str(POLICY),'new_holdout_used':False},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(overall,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
