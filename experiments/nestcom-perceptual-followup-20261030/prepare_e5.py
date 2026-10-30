"""Freeze 16 source-to-symbol / symbol-to-image timing groups, no inference."""
from followup_common import *
from frame_plan import policy,nt
from grid_codec import endpoints

def main():
 out=L/'E5_plan';out.mkdir(exist_ok=False)
 sources=read(L/'inventory_core/sources.json')[:32]
 assert [s['source_id']for s in sources]==read(D/'source_sets.json')['timing32']
 a=read(OLD/'protocol_resolved_p0.json');old=read(R/'results/five_method_timing_20261020_v1/registration.json')
 ada=read(R/'results/adatoken_response_20261026_v1/c_predclass_common500_source_registration.json')
 cats={str(n):pin(OLD/f'grid_catalogues/N{n}.json')for n in [512,1024]}
 ada1024=read(R/'results/adatoken_response_20261026_v1/c_predclass_common500_link_registration.json')['catalogue']
 cfg={k:old[k]for k in ['P_checkpoint','Swin_checkpoint','arpc_public','bpg_config','bpg_policies','arpc_policies','static','phy_qualification']}
 cfg.update(model_routes=a['models'],ada_native={k:ada[k]for k in ['weights','upstream_code','classifier','revision']},catalogues=cats,ada1024_catalogue=ada1024,
  n512_qualification=read(OLD/'final_phy_512_registration.json')['n512_qualification'],source_order=[s['source_id']for s in sources],
  cpu_affinity=[38,39],cpu_cores=2,torch_threads=6,torch_interop_threads=2,resource_note='Six original math-library threads confined to exactly two logical CPUs for every group; not six-core measurements.',
  GPU_UUID='GPU-f5047136-08e5-9416-3c86-87786d0b355e',batch=1,warmups=2,measurements=3,
  purpose='Fresh matched deployment timing only; never added to quality populations',boundaries=dict(TX='canonical RGB in host memory through actual emitted symbols',RX='actual noisy symbols in host memory through final canonical RGB'),
  excluded=['initial model/codec setup','disk I/O','metric networks','logs'],BPG_scratch='RAM-backed /dev/shm native CLI buffers; codec-required byte serialization and process invocation remain inside TX/RX',
  source_output_cache=False,training=False)
 groups=[]
 def add(name,method,N,snr,grid=None,target=None,matched=False):
  if method in ['WHOLE','Token_NeST']:
   c=policy(N,snr,'WHOLE'if method=='WHOLE'else'NEST',a)
   if target is not None:c=dict(c,target=target)
   es=tuple((m,0)for m in range(4,11))if grid=='whole'else endpoints(grid)
   # A WHOLE comparator never admits a partial target just because it is the ceiling.
   es=tuple(sorted((e for e in es if nt(e)<=nt(c['target'])),key=nt,reverse=True))
   g=dict(name=name,method=method,N=N,snr_db=snr,grid=grid,policy=c,endpoints=[list(e)for e in es],matched_granularity=matched,catalogue=cats[str(N)])
  elif method=='Ada':
   c=policy(N,snr,'ADA',a)if N==512 else dict(target=[256,0],q=4,rate='1/2'if snr==7 else'2/3')
   g=dict(name=name,method=method,N=N,snr_db=snr,policy=c,catalogue=cats['512']if N==512 else cats['1024']if snr==7 else ada1024)
  else:g=dict(name=name,method=method,N=N,snr_db=snr)
  groups.append(g)
 for N,snr in [(1024,7),(1024,10),(512,10)]:
  for m,grid in [('WHOLE','whole'),('Token_NeST','token'),('Ada',None)]:add(f'{m}_N{N}_S{snr}',m,N,snr,grid)
 for m in ['ARPC','SwinJSCC','Latent_JSCC','BPG_LDPC']:add(m+'_N1024_S10',m,1024,10)
 for grid in ['whole','quarter','eighth']:add('granularity_'+grid,'WHOLE'if grid=='whole'else'Token_NeST',1024,10,grid,[8,126],True)
 assert len(groups)==16
 qualified=read(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json');qs={s['source_id']:s for s in qualified}
 dsource={s['source_id']:s for s in read(OL/'final_D_source/outputs.json')}
 truths=[]
 for g in groups:
  if g['method']not in ['WHOLE','Token_NeST','Ada']:continue
  cat=cr(g['catalogue']);pp=[p for p in cat['profiles']if (p['q'],p['nominal_rate'])==(g['policy']['q'],g['policy']['rate'])];assert pp
  for s in sources:
   v=qs[s['source_id']];assert v['archive']==s['archive']and v['preprocessing_id']==s['preprocessing_id']
   sp=v['ada_streams']if g['method']=='Ada'else dsource[s['source_id']]['streams']
   with np.load(checked(sp),allow_pickle=False)as z:
    cap=pp[0]['k']-(39 if g['method']=='Ada' else 29)
    ee=range(256,0,-1)if g['method']=='Ada'else[gx for gx in g['endpoints']]
    fitted=[]
    for e in ee:
     key='p%03d'%e if g['method']=='Ada'else'm%d_K%d'%tuple(e);b=z[key]
     if len(b)<=cap:fitted.append((e,b.copy()))
    assert fitted,'All timing sources must preserve the original no-fit policy if ever reached'
    e,b=fitted[0]
   truths.append(dict(group=g['name'],source_id=s['source_id'],endpoint=[e,0]if isinstance(e,int)else e,arithmetic_bits_sha256=asha(b),arithmetic_bits=len(b),predicted_class=v['predicted_class']if g['method']=='Ada'else None))
 save(out/'sources.json',sources);save(out/'groups.json',groups);save(out/'expected_TX.json',truths);save(out/'config.json',cfg)
 save(out/'completion.json',dict(status='PREPARED',groups=16,sources=32,fresh_TX_calls=2560,fresh_RX_calls=2560,measured_calls_per_direction=1536,warmup_calls_per_direction=1024,new_models=0,new_PHY=0,protocol=pin(D/'protocol.json'),counter_noise='Original common500 repeat0 counter and method-specific noise recipes; each timing call recomputes outputs. Generation uses original received-payload seed recipe.'))
 print('E5_PREPARED',len(groups),len(truths))
if __name__=='__main__':main()
