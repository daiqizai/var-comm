"""E2 independently decodes the transferred prefix, then samples its tail."""
from followup_common import *
import tempfile,traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']]
 jobs=cr(r['jobs']);reuse=cr(r['reuse']);assert len({j['key']for j in jobs})==len(jobs)
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp');os.sched_setaffinity(0,set(r['cpu_affinity']))
 os.environ.update(TMPDIR=str(out/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps'])
 try:
  sys.path[:0]=[str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/adatoken-response-20261026/c_predclass_production_rx_v1'),str(R/'experiments/adatoken-response-20261026')]
  from runtime_boundary import firewall
  firewall(out,[p['path']for p in r['weights'].values()]+[v['image']['path']for v in reuse.values()if v['image']is not None],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  from c_predclass_native import load_native
  from c_arithmetic import load_codec
  from protocol_rx import realize
  # Original loader compares the driver's short/lower-case PCI spelling.
  load_request=dict(r,pci='0000:'+r['pci'].split(':',1)[1].lower())
  Encoder,Decoder,cdf,proof=load_codec();save(out/'arithmetic_proof.json',proof);n=load_native(load_request,out,ledger)
  output=[]
  with n.torch.inference_mode():
   for j in jobs:
    if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('E2 Ada bounded stage')
    key=j['key'];p=j['projection']
    if key in reuse:
     v=reuse[key];assert p['m']==256 and p['status']=='SOURCE_INPUT'
     with np.load(checked(v['image']),allow_pickle=False)as z:assert asha(z['image'])==v['pixel_sha256']and asha(z['received_tokens'])==v['received_tokens_sha256']and len(z['received_tokens'])==256
     output.append(dict(v,key=key,origin='EXACT_FULL_256_PREFIX_REUSE_SEED_IRRELEVANT'));continue
    if p['status']!='SOURCE_INPUT':
     a=np.full((3,256,256),.5,np.float32);output.append(dict(key=key,image=None,pixel_sha256=asha(a),received_tokens_sha256=asha(np.empty(0,np.int64)),status='FIXED_GRAY',origin='REGISTERED_NULL',reason=p['reason']));continue
    assert set(p)=={'status','codec','m','K','received_class','bits'}and p['codec']=='ADA'
    def receive():
     v=realize(n,Encoder,Decoder,cdf,np.array(p['bits'],np.uint8),p['m'],p['received_class'],j['generation_seed'])
     image=v.pop('image');tokens=v.pop('received_tokens');completed=v.pop('completed_tokens')
     assert image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()and 0<=image.min()<=image.max()<=1
     path=out/'images'/(key+'.npz')
     with path.open('xb')as f:np.savez(f,image=image,received_tokens=tokens,completed_tokens=completed);f.flush();os.fsync(f.fileno())
     row=dict(key=key,image=pin(path),pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),generation_seed=j['generation_seed'],origin='NEW_ERROR_FREE_PREFIX_DECODE_AND_SAMPLING',source_truth_used=False,TX_CDF_used=False,new_PHY=0,**v)
     save(path.with_suffix('.json'),row);return row
    output.append(ledger.call('receive_render',key,receive))
    if len(output)%25==0:print('ADA_INFORMATION_RX',len(output),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',output);save(out/'completion.json',dict(status='COMPLETE',outputs=len(output),new_receive_render=sum(v['origin']=='NEW_ERROR_FREE_PREFIX_DECODE_AND_SAMPLING'for v in output),reuse=len(reuse),ledger=ledger.summary(),seconds=time.monotonic()-start,new_PHY=0,new_training=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
