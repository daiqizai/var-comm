"""Two bounded receiver counterfactuals. Only accepted RX tokens enter models."""
from evidence_common import *
import tempfile,traceback
SIZES=(1,2,3,4,5,6,8,10,13,16)
def ada_tail(n,tokens,received_class,seed,greedy):
 t=n.torch;n.start(received_class);a=[];calls=0
 with t.random.fork_rng(devices=[0]):
  t.manual_seed(seed);t.cuda.manual_seed_all(seed)
  for i in range(256):
   logits=n.logits(i);calls+=1
   if i<len(tokens):q=int(tokens[i])
   elif greedy:
    cond,uncond=t.split(logits,len(logits)//2,dim=0)
    q=int((uncond+(cond-uncond)*4.0)[0,0].argmax(-1).item())
   else:q=int(n.tail(logits).reshape(-1)[0].item())
   a.append(q);n.previous=q
  result=np.asarray(a,np.int64);assert np.array_equal(result[:len(tokens)],tokens)
  image=n.render(result)
 return image,result,dict(prior_forwards=calls,prior_branch_forwards=2*calls,decoder_forwards=1)
def coarse(n,tokens,m):
 # _Prior.advance is the original S=10 residual update (including C_s).
 p=n.receiver._Prior(n.loaded['vae'],n.loaded['var'],n.loaded['device']);offset=0;hashes=[]
 try:
  for s in range(m):
   size=SIZES[s]**2;part=tokens[offset:offset+size];assert len(part)==size
   p.advance(part,s);offset+=size;hashes.append(asha(p.fhat.cpu().numpy()))
  assert offset==len(tokens)and p.fhat.shape[-2:]==(16,16)
  image=n.loaded['decoder'](p.fhat)[0].clamp(0,1).cpu().numpy().astype(np.float32)
  return image,dict(S=10,m=m,latent_shape=list(p.fhat.shape),prefix_latent_sha256=hashes[-1],received_scale_latent_hashes=hashes,unreceived_scale_updates=0,codebook_zero_substitution=False,prior_forwards=0,decoder_forwards=1)
 finally:p.close()
def nest_replay(n,tokens,m):
 p=n.receiver._Prior(n.loaded['vae'],n.loaded['var'],n.loaded['device']);offset=0;prefix_hash=None
 try:
  for s,size in enumerate(SIZES):
   chosen=p.logits(s)[0].argmax(-1).cpu().numpy()
   if s<m:chosen[:]=tokens[offset:offset+size*size];offset+=size*size
   p.advance(chosen,s)
   if s==m-1:prefix_hash=asha(p.fhat.cpu().numpy())
  assert offset==len(tokens)
  image=n.loaded['decoder'](p.fhat)[0].clamp(0,1).cpu().numpy().astype(np.float32)
  return image,prefix_hash
 finally:p.close()
def main(request,expected):
 assert sha(request)==expected;r=read(request);rp=pin(request)
 for d in r['code']:checked(d)
 jobs=cr(r['inputs']);assert len(jobs)==r['count']and len({j['key']for j in jobs})==len(jobs)
 # Validate all archived receiver inputs and identities before installing access firewall.
 for j in jobs:
  old=cr(j['numeric_identity']);assert old==r['donor_identity']
  a=np.asarray(j['received_tokens'],np.int64);assert asha(a)==j['received_tokens_sha256']
  assert set(j)==({'kind','key','received_tokens','received_tokens_sha256','received_class','generation_seed','numeric_identity','donor_image','donor_pixel_sha256','prefix'}if r['kind']=='e1'else{'kind','key','m','K','received_tokens','received_tokens_sha256','numeric_identity','donor_image','donor_pixel_sha256'})
  with np.load(checked(j['donor_image']),allow_pickle=False)as z:
   assert asha(z['image'])==j['donor_pixel_sha256']and np.array_equal(z['received_tokens'],a)
  if r['kind']=='e1':assert 0<len(a)==j['prefix']<256 and np.all((a>=0)&(a<16384))and 0<=j['received_class']<1000
  else:assert j['K']==0 and 1<=j['m']<=10 and len(a)==sum(x*x for x in SIZES[:j['m']])and np.all((a>=0)&(a<4096))
 out=Path(r['output']);out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'))
 os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);start=time.monotonic()
 save(out/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),registration=rp))
 try:
  def boundary():assert time.monotonic()-start<r['max_seconds']and not(L/'STOP').exists()
  sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'));from runtime_boundary import firewall
  if r['kind']=='e1':
   firewall(out,[d['path']for d in r['weights'].values()],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
   sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'));from c_predclass_native import load_native
   n=load_native(r,out,ledger)
  else:
   routes=cr(r['model_routes'])
   firewall(out,[d['path']for d in routes['files']+routes['author_files']]+[routes['dist']['path']],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
   sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'));from kn_native import load
   n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']))
  rows=[];qualifications=[]
  with(n.torch.inference_mode()if r['kind']=='e1'else n.torch.no_grad()):
   for j in jobs:
    boundary();assert n.torch.cuda.mem_get_info()[0]>=2<<30;a=np.asarray(j['received_tokens'],np.int64);key=j['key']
    if r['kind']=='e1'and key in r['qualification_keys']:
     def replay():
      image,tokens,counts=ada_tail(n,a,j['received_class'],j['generation_seed'],False)
      assert asha(image)==j['donor_pixel_sha256'],'Original sampling replay pixel mismatch'
      return dict(key=key,original_pixel_sha256=asha(image),matched=True,**counts)
     qualifications.append(ledger.call('qualification_replay',key,replay))
    def produce():
     if r['kind']=='e1':image,completed,counts=ada_tail(n,a,j['received_class'],j['generation_seed'],True);extra=dict(completed_tokens=completed);proof=dict(counts,tail='argmax',prefix_unchanged=True,context='fresh KV; greedy path after first generated token',entropy_CDF_changed=False)
     else:image,proof=coarse(n,a,j['m']);extra={}
     assert image.shape==(3,256,256)and image.dtype==np.float32 and np.isfinite(image).all()and 0<=image.min()<=image.max()<=1
     p=out/'images'/(key+'.npz')
     with p.open('xb')as f:np.savez(f,image=image,received_tokens=a,**extra);f.flush();os.fsync(f.fileno())
     row=dict(key=key,image=pin(p),pixel_sha256=asha(image),received_tokens_sha256=asha(a),proof=proof,new_wireless_frames=0,source_truth_used=False)
     save(p.with_suffix('.json'),row);return row
    row=ledger.call('receiver_render',key,produce);rows.append(row)
    if r['kind']=='e2'and key in r['qualification_keys']:
     def replay():
      image,h=nest_replay(n,a,j['m'])
      assert asha(image)==j['donor_pixel_sha256'],'Original WHOLE replay pixel mismatch'
      assert h==row['proof']['prefix_latent_sha256'],'Received-scale latent contribution changed'
      if j['m']==10:assert asha(image)==row['pixel_sha256']
      return dict(key=key,original_pixel_sha256=asha(image),prefix_latent_sha256=h,matched=True,prior_forwards=10,decoder_forwards=1)
     qualifications.append(ledger.call('qualification_replay',key,replay))
    if len(rows)%25==0:print('EVIDENCE_RX',r['kind'],len(rows),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',rows);save(out/'qualification.json',qualifications)
  save(out/'completion.json',dict(status='COMPLETE_COUNTERFACTUAL_RX',registration=rp,outputs=pin(out/'outputs.json'),qualification=pin(out/'qualification.json'),count=len(rows),ledger=ledger.summary(),seconds=time.monotonic()-start,new_PHY=0,new_training=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
