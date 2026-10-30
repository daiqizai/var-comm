"""Bounded native reconstruction variants, exact cached receiver outputs only."""
from opt_common import *
import traceback,tempfile
from receiver_variants import completed_latent,decode_image,load_anchor

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 jobs=cr(r['jobs']);routes=cr(r['model_routes']);static=r['static'];reuse=cr(r['reuse'])if r.get('reuse')else{}
 out=Path(r['output']);out.mkdir(exist_ok=False)
 for sub in ('images','latents','tmp'):(out/sub).mkdir()
 tempfile.tempdir=str(out/'tmp');os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
 os.sched_setaffinity(0,set(r['cpu_affinity']));start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps'])
 save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop(r['route']+' time cap or requested stop')
 try:
  sys.path[:0]=[str(R/'src'),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
  from runtime_boundary import firewall
  allowed=[v['path']for v in routes['files']+routes['author_files']]+[routes['dist']['path'],static['table']['path'],static['completion']['path']]
  allowed += [v['image']['path']for v in reuse.values()]+[v['path']for v in r.get('anchors',[])]+[v['path']for v in r.get('correctors',[])]
  firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  from kn_native import load,receiver_codec
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));torch=n.torch
  codec=receiver_codec(n);from h800_ep48_link_core_v1 import bind_static;bind_static(codec,static)
  anchors={}
  if r['route']=='A':
   for i,p in enumerate(r['anchors']):anchors['A'+str(i+1)]=ledger.call('anchor_load',str(i),lambda p=p,i=i:load_anchor(n,p,2026102201+i))
  rows=[];new_decode=0;new_render=0;baseline_reuse=0
  with torch.no_grad():
   for j in jobs:
    boundary();key=j['key'];p=j['projection'];cached=reuse.get(key);lat=None;meta={};tokens=None
    if cached is not None:
     with np.load(checked(cached['image']),allow_pickle=False)as z:base=z['image'].copy();tokens=z['received_tokens'].copy()
     assert asha(base)==cached['pixel_sha256']and asha(tokens)==cached['received_tokens_sha256']
     baseline=dict(cached,variant='original',origin='EXACT_CACHED_RECEIVER');baseline_reuse+=1
    elif p['status']!='SOURCE_INPUT':
     base=np.full((3,256,256),.5,np.float32);tokens=np.empty(0,np.int64)
     baseline=dict(key=key,variant='original',image=None,pixel_sha256=asha(base),received_tokens_sha256=asha(tokens),status='FIXED_GRAY',origin='FIXED_GRAY')
    else:
     assert p['codec']=='NEST'
     v=ledger.call('source_decode',key,lambda:codec.decode(p['family'],np.array(p['bits'],np.uint8),p['m'],p['K']));new_decode+=1
     assert v['canonical']and not v['receiver_truth_used']and not v['transmitter_probability_table_used'];tokens=np.asarray(v['received_tokens'],np.int64)
     lat,meta=ledger.call('completion_forward',key+':original',lambda:completed_latent(n,tokens,p['m'],p['K']))
     base=ledger.call('render',key+':original',lambda:decode_image(n,lat));new_render+=1
     path=out/'images'/(key+'_original.npz')
     with path.open('xb')as f:np.savez(f,image=base,received_tokens=tokens)
     baseline=dict(key=key,variant='original',image=pin(path),pixel_sha256=asha(base),received_tokens_sha256=asha(tokens),status='ACCEPTED_SOURCE',origin='NEW_BASELINE',**meta)
    rows.append(baseline)
    variants= [('A1',0,None),('A2',0,None)]if r['route']=='A'else[(f'B_T{t:g}_s{s}',t,s)for t in(.3,.7)for s in(2026103001,2026103002,2026103003)]
    for name,T,seed in variants:
     bypass=p['status']!='SOURCE_INPUT'or(p['m']==10 and p['K']==0)or(r['route']=='A'and (p['m'],p['K'])not in((7,0),(7,25)))
     if bypass:rows.append(dict(baseline,variant=name,origin='EXACT_BYPASS',reused_variant='original'));continue
     if r['route']=='A':
      if lat is None:lat,meta=ledger.call('completion_forward',key+':original',lambda:completed_latent(n,tokens,p['m'],p['K']))
      image=ledger.call('anchor_render',key+':'+name,lambda:decode_image(n,lat,anchors[name]))
     else:
      lat,meta=ledger.call('completion_forward',key+':'+name,lambda:completed_latent(n,tokens,p['m'],p['K'],T,seed))
      image=ledger.call('render',key+':'+name,lambda:decode_image(n,lat))
     new_render+=1;path=out/'images'/(key+'_'+name+'.npz')
     with path.open('xb')as f:np.savez(f,image=image,received_tokens=tokens)
     rows.append(dict(key=key,variant=name,image=pin(path),pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),status='ACCEPTED_SOURCE',origin='NEW_VARIANT',temperature=T,generation_seed=seed,**meta))
    if len(rows)%100==0:print('RX',r['route'],len(rows),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE',route=r['route'],unique_inputs=len(jobs),rows=len(rows),new_source_decodes=new_decode,new_renders=new_render,baseline_reused=baseline_reuse,ledger=ledger.summary(),seconds=time.monotonic()-start,new_PHY=0,new_training=0))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),ledger=ledger.summary(),seconds=time.monotonic()-start,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
