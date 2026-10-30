"""Frozen teacher cache, 10,000 train IDs at each of m7/m8; no RX target leak."""
from opt_common import *
import traceback,tempfile
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];sources=cr(r['sources']);assert len(sources)==10000
 routes=cr(r['model_routes']);out=Path(r['output']);out.mkdir(exist_ok=False);(out/'items').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'));os.sched_setaffinity(0,set(r['cpu_affinity']))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);rows=[];actual_bytes=0
 save(out/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),registration=pin(request)))
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('C cache budget or requested stop')
 try:
  import torch
  sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'));from kn_native import load
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']))
  torch.save(n.loaded['vae'].quantize.embedding.weight.detach().cpu(),out/'codebook.pt')
  loaded={};last=None;lat=None;rgb=None
  # Sorted by shard only changes cache traversal, not frozen sample identities or training order.
  with torch.no_grad():
   for s in sorted(sources,key=lambda s:s['source_index']):
    boundary();key=s['source_id'];name=f"i{s['source_index']:05d}";ip=out/'items'/(name+'.pt')
    if last!=s['old_latent_shard']['path']:
     lat=torch.load(checked(s['old_latent_shard']),map_location='cpu',weights_only=True)
     rgb=torch.load(checked(s['rgb_shard']),map_location='cpu',weights_only=True);last=s['old_latent_shard']['path']
    i=s['within_shard'];assert lat['image_ids'][i]==rgb['image_ids'][i]==s['source_id']
    pixels=rgb['targets_u8'][i].numpy();assert hashlib.sha256(pixels.tobytes()).hexdigest()==s['source_pixels_sha256']
    tokens=lat['full_tokens'][i].numpy().astype(np.int64);assert tokens.shape==(680,)and ((tokens>=0)&(tokens<4096)).all()
    def produce():
     p=n.receiver._Prior(n.loaded['vae'],n.loaded['var'],n.loaded['device']);records=[];nxt=None
     q=n.loaded['vae'].quantize;advance=q.get_next_autoregressive_input
     def capture(*args,**kwargs):
      nonlocal nxt
      f,nxt=advance(*args,**kwargs);return f,nxt
     q.get_next_autoregressive_input=capture
     try:
      for scale in range(9):
       logits=p.logits(scale)
       if scale in(7,8):
        h=SIZES[scale];assert nxt.shape==(1,32,h,h)
        records.append(dict(m=scale,coarse=nxt[0].reshape(32,-1).T.contiguous().cpu().clone(),base_logits=logits[0].cpu().clone(),target=torch.from_numpy(tokens[OFFSETS[scale]:OFFSETS[scale+1]].copy())))
       if scale<8:p.advance(tokens[OFFSETS[scale]:OFFSETS[scale+1]],scale)
     finally:q.get_next_autoregressive_input=advance;p.close()
     assert len(records)==2
     # Labels are stored separately and consumed only by supervised loss/mask construction.
     torch.save(dict(source_id=key,pixel_sha256=s['source_pixels_sha256'],records=records),ip)
     return dict(source_id=key,source_index=s['source_index'],native_source_hash=s['source_pixels_sha256'],item=pin(ip),scales=[7,8])
    row=ledger.call('source_teacher_cache',key,produce);rows.append(row);actual_bytes+=row['item']['bytes']
    assert actual_bytes<80*(1<<30)
    save(ip.with_suffix('.json'),row)
    if len(rows)%100==0:print('C_CACHE',len(rows),10000,'seconds',round(time.monotonic()-start,1),flush=True)
  save(out/'index.json',rows);save(out/'completion.json',dict(status='COMPLETE',count=len(rows),source_scale_pairs=20000,actual_bytes=actual_bytes,index=pin(out/'index.json'),codebook=pin(out/'codebook.pt'),ledger=ledger.summary(),seconds=time.monotonic()-start))
 except BaseException as e:
  save(out/'partial_index.json',rows);save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),committed=len(rows),actual_bytes=actual_bytes,ledger=ledger.summary(),seconds=time.monotonic()-start,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
