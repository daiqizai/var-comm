"""RX revision: original NeST no_grad route; request pinned before RX firewall.
Original failed worker and all committed outputs remain unchanged.
"""
from n512_common import *
import tempfile,traceback
def main(request,expected):
 assert sha(request)==expected;r=read(request);request_identity=pin(request)
 for d in r['code']:checked(d)
 jobs=cr(r['inputs']);assert len({j['key'] for j in jobs})==len(jobs)
 out=Path(r['output']);out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir();(out/'tmp').mkdir()
 os.environ.update(TMPDIR=str(out/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');tempfile.tempdir=str(out/'tmp')
 os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);beg=time.monotonic();rows=[]
 save(out/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),request=request_identity))
 try:
  codec_name=r['codec'];assert codec_name in ('NEST','ADA')
  fields={'key','projection'}
  for j in jobs:
   assert set(j)==fields and j['key']==digest(j['projection']);p=j['projection'];assert p['codec']==codec_name and p['status']=='SOURCE_INPUT'
   assert set(p)==({'status','codec','m','K','family','bits'} if codec_name=='NEST' else {'status','codec','m','K','received_class','bits'})
  def boundary():assert time.monotonic()-beg<r['max_seconds'] and not (L/'STOP').exists()
  sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
  from runtime_boundary import firewall
  if codec_name=='NEST':
   routes=cr(r['model_routes']);static=r['static'];allowed=[v['path'] for v in routes['files']+routes['author_files']]+[routes['dist']['path'],static['completion']['path'],static['table']['path']]
   firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
   sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'))
   from kn_native import load,receiver_codec
   n,render=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));codec=receiver_codec(n)
   from ep_source_codec import InvalidSourceStream
   from h800_ep48_link_core_v1 import bind_static
   bind_static(codec,static)
  else:
   sys.path[:0]=[str(R/'experiments/adatoken-response-20261026/c_predclass_production_rx_v1'),str(R/'experiments/adatoken-response-20261026')]
   from c_arithmetic import load_codec
   from c_predclass_native import load_native
   from c_predclass_protocol import generation_seed
   from protocol_rx import realize
   firewall(out,[v['path'] for v in r['weights'].values()],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
   Encoder,Decoder,cdf,proof=load_codec();save(out/'arithmetic_proof.json',proof);n=load_native(r,out,ledger)
  with (n.torch.no_grad() if codec_name=='NEST' else n.torch.inference_mode()):
   for j in jobs:
    boundary();assert n.torch.cuda.mem_get_info()[0]>=2<<30;p=j['projection'];key=j['key'];bits=np.array(p['bits'],np.uint8)
    def receive():
     if codec_name=='NEST':
      try:v=codec.decode(p['family'],bits,p['m'],p['K'])
      except InvalidSourceStream as e:
       v=dict(status='FIXED_GRAY_SOURCE_REJECT',reason=str(e));image=np.full((3,256,256),.5,np.float32);tokens=np.empty(0,np.int64)
      else:
       tokens=v['received_tokens'];assert v['canonical'] and not v['receiver_truth_used'] and not v['transmitter_probability_table_used']
       image=render(n,tokens.copy(),p['m'],p['K']);v=dict(status='ACCEPTED_SOURCE',reason=None,canonical=True)
     else:
      seed=generation_seed(bits,p['m'],p['received_class']);v=realize(n,Encoder,Decoder,cdf,bits,p['m'],p['received_class'],seed)
      image=v.pop('image');tokens=v.pop('received_tokens');v.pop('completed_tokens');v['generation_seed']=seed
     assert image.shape==(3,256,256) and image.dtype==np.float32 and np.isfinite(image).all() and 0<=image.min()<=image.max()<=1
     ip=out/'images'/(key+'.npz')
     with ip.open('xb') as f:np.savez(f,image=image,received_tokens=tokens);f.flush();os.fsync(f.fileno())
     record=dict(key=key,codec=codec_name,image=pin(ip),pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),source_truth_used=False,TX_CDF_used=False,**v)
     save(ip.with_suffix('.json'),record);return record
    rows.append(ledger.call('receive_render',key,receive))
    if len(rows)%25==0:print('RX_COMMITTED',codec_name,len(rows),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE_RX',request=request_identity,outputs=pin(out/'outputs.json'),count=len(rows),ledger=ledger.summary(),seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
