"""Public fixed-policy directories and finite, result-independent candidates."""
from study_common import *
from grid_codec import endpoints
from fractions import Fraction

def catkey(method,q,rate):return f'{method}_q{q}_r{rate.replace("/","-")}'
def nt(e):return OFFSETS[e[0]]+e[1]
def candidates():
 p=read(D/'protocol.json')['unified'];rows=[]
 for method in ['WHOLE','NeST','Ada']:
  for e in p['target_ceilings'][method]:
   target=[e,0]if method=='Ada'else e
   for a in p['MCS']:
    q,rate=a['q'],a['rate'];key=catkey(method,q,rate)
    rows.append(dict(method=method,target=target,q=q,rate=rate,catalogue_key=key,candidate_id=f'{key}_m{target[0]:03d}_K{target[1]:03d}'))
 assert len(rows)==396;return rows

def directories():
 out=L/'calibration_assets';out.mkdir(exist_ok=False)
 sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
 import c_predclass_frame as cp
 phy=cp.isolated_engine();phy.pack_body.__globals__['FAMILIES']=('EC_VAR_WHOLE','EC_VAR_PARTIAL','ADA_PREDICTED_CLASS_PREFIX','EXPLICIT_NULL_FIXEDFRAME')
 qual=read(OLD/'final_phy_1024_registration.json')['qualification'];rt=phy.create_runtime(R,str(checked(qual)))
 base=rt.catalogue;cats={};profiles=[]
 for c in candidates():
  key=c['catalogue_key']
  if key in cats:continue
  method,q,rate=c['method'],c['q'],c['rate'];n=956*q;k=int(Fraction(rate)*n);layout=rt.backend.plan(k,n,q);rt.backend.codecs(k,n,q)
  assert layout['code_blocks']==1
  es=[[m,0]for m in range(4,11)]if method=='WHOLE'else endpoints('token')if method=='NeST'else [[i,0]for i in range(1,257)]
  rows=[]
  for index,(m,K)in enumerate(list(es)+[[0,0]]):
   null=m==0;family='EXPLICIT_NULL_FIXEDFRAME'if null else'ADA_PREDICTED_CLASS_PREFIX'if method=='Ada'else'EC_VAR_PARTIAL'if K else'EC_VAR_WHOLE'
   p=dict(profile_id=index,family=family,mode='arithmetic',m=m,K=K,token_count=0 if null else m if method=='Ada'else nt((m,K)),q=q,nominal_rate=rate,k=k,n=n,N=1024,header_symbols=68,body_symbols=956,frame_padding_symbols=0,length_bits=13,crc_bits=16,class_bits=10 if method=='Ada'and not null else 0,source_capacity_bits=k-29,arithmetic_capacity_bits=k-(39 if method=='Ada'and not null else 29),layout=layout,layout_id=layout['layout_id'])
   p['profile_key']=digest(p);rows.append(p)
  cat=dict(protocol='N1024_PUBLIC_FIXED_POLICY_DIRECTORY_V1',method=method,q=q,rate=rate,profiles=rows,profile_bits=12,preagreed_policy_context=True,per_source_directory=False,qualification=qual)
  cat['catalogue_sha256']=digest(cat);assert max(p['profile_id']for p in rows)<4096
  path=out/(key+'.json');save(path,cat);cats[key]=pin(path)
  profiles.extend({k:p[k]for k in ['profile_id','family','m','K','q','nominal_rate','k','n','arithmetic_capacity_bits']}|{'catalogue_key':key}for p in rows)
 # Public directory choice is method/MCS policy context, never source-dependent.
 save(D/'public_directories.json',dict(directory=cats,candidates=candidates(),qualification=qual,source_code=pin(E/'calibration_plan.py'),protocol=pin(D/'protocol.json')))
 csvout(D/'profile_directory.csv',profiles)
 print('DIRECTORIES',len(cats),'CANDIDATES',len(candidates()),flush=True)

def token_sources(stage):
 if stage=='formal':return {s['source_id']:s for s in read(OL/'final_D_source/outputs.json')}
 existing=read(L/'source_plan/reuse.json')
 if stage=='full':
  closed('cal_token_source800')
  for s in read(L/'cal_token_source800/outputs.json'):assert s['source_id']not in existing;existing[s['source_id']]=s
 return existing

def plan(stage):
 assert stage in ['pilot','full','formal'];protocol=read(D/'protocol.json')['unified'];pub=read(D/'public_directories.json');cats={k:cr(v)for k,v in pub['directory'].items()};streams=token_sources(stage)
 sources=read(B/'nestcom_n512_20261028_v1/qualified_sources'/('common500.json'if stage=='formal'else'cal1000.json'))
 if stage=='pilot':
  ids=read(R/'results/nestcom_n512_20261028_v1/protocol.json')['pilot_source_ids'];sources=[s for s in sources if s['source_id']in ids];assert len(sources)==100 and all(s['source_id']in streams for s in sources)
  configs={str(snr):pub['candidates']for snr in protocol['SNR']};seeds=[4101]
 elif stage=='full':configs=read(D/'pilot_finalists.json')['candidates_by_snr'];seeds=[4101,4102,4103];assert len(sources)==1000
 else:configs={snr:list(cs.values())for snr,cs in read(D/'frozen_policies.json')['policies'].items()};seeds=[6201,6202,6203];assert len(sources)==500
 out=L/('cal_'+stage+'_plan');out.mkdir(exist_ok=False);choices={};physical={};logical=[]
 for s in sources:
  v=streams[s['source_id']];assert s['preprocessing_id']==v['source_pixels_sha256']
  with np.load(checked(v['streams']),allow_pickle=False)as z:nest={k:z[k].copy()for k in z.files}
  with np.load(checked(s['ada_streams']),allow_pickle=False)as z:ada={k:z[k].copy()for k in z.files}
  for snr in protocol['SNR']:
   for c in configs[str(snr)]:
    cid=digest([s['source_id'],c]);ck=c['catalogue_key'];cat=cats[ck];method=c['method']
    if cid not in choices:
     ceiling=c['target'];es=list(range(1,ceiling[0]+1))if method=='Ada'else [e for e in ([[m,0]for m in range(4,11)]if method=='WHOLE'else endpoints('token'))if nt(e)<=nt(ceiling)]
     es.sort(key=(lambda e:e)if method=='Ada'else nt,reverse=True);cap=cat['profiles'][0]['k']-(39 if method=='Ada'else 29)
     fits=[(e,(ada['p%03d'%e]if method=='Ada'else nest['m%d_K%d'%tuple(e)]))for e in es if len(ada['p%03d'%e]if method=='Ada'else nest['m%d_K%d'%tuple(e)])<=cap]
     if fits:
      e,b=fits[0];m,K=(e,0)if method=='Ada'else e
      p=next(p for p in cat['profiles']if p['m']==m and p['K']==K)
      payload=np.concatenate((np.array([(s['predicted_class']>>i)&1 for i in range(9,-1,-1)],np.uint8),b))if method=='Ada'else b
     else:p=cat['profiles'][-1];b=np.zeros(0,np.uint8);payload=np.zeros(2,np.uint8)
     choices[cid]=dict(catalogue_key=ck,profile_id=p['profile_id'],payload=payload.tolist(),payload_sha256=asha(payload),m=p['m'],K=p['K'],source_bits=len(b),capacity_bits=cap,utilization=len(b)/cap,NULL=not bool(fits),class_bits=10 if method=='Ada'and fits else 0,q=c['q'],rate=c['rate'],actual_length_comparisons=len(es))
    choice=choices[cid]
    for repeat,seed in enumerate(seeds):
     counter=(90000000 if stage=='formal'else 85000000)+s['source_index']*3+repeat
     identity=dict(N=1024,source_id=s['source_id'],source_index=s['source_index'],snr_db=snr,noise_seed=seed,counter=counter,catalogue_sha256=pub['directory'][ck]['sha256'],profile_id=choice['profile_id'],payload_sha256=choice['payload_sha256'])
     key=digest(identity);physical.setdefault(key,dict(key=key,choice_id=cid,catalogue_key=ck,**identity))
     logical.append(dict(source_id=s['source_id'],source_index=s['source_index'],method=method,N=1024,snr_db=snr,noise_seed=seed,channel_repeat=repeat,candidate_id=c['candidate_id'],target=c['target'],physical_key=key,choice_id=cid))
 limit=protocol['confirmation'if stage=='full'else stage]['max_logical'];assert len(logical)<=limit
 save(out/'choices.json',choices);save(out/'logical.json',logical);save(out/'sources.json',[source_only(s)for s in sources]);save(out/'physical.json',list(physical.values()));shards=[]
 for i in range(8):
  p=out/f'phy_{i:02d}.json';save(p,[j for j in physical.values()if j['source_index']%8==i]);shards.append(pin(p))
 save(out/'completion.json',dict(status='COMPLETE_PLAN',stage=stage,logical=len(logical),physical=len(physical),sources=len(sources),shards=shards,choices=pin(out/'choices.json'),logical_file=pin(out/'logical.json'),sources_file=pin(out/'sources.json'),protocol=pin(D/'protocol.json')))
 print('PLAN',stage,len(logical),len(physical),flush=True)
if __name__=='__main__':directories()if sys.argv[1]=='directories'else plan(sys.argv[1])
