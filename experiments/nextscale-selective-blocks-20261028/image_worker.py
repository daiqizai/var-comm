"""Native reconstructions from actual RX tokens or independently decoded wire bits."""
from block_common import *
from protocol import state_identity,OFFSETS
import traceback,tempfile

def main(path,expected):
 assert sha(path)==expected;r=read(path);rp=pin(path)
 for d in r['code']:checked(d)
 jobs=cr(r['inputs']);routes=cr(r['model_routes']);mode=r['mode'];assert mode in('SINGLE','SELECTIVE')
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);beg=time.monotonic()
 save(out/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),request=rp))
 try:
  sys.path[:0]=[str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
  from runtime_boundary import firewall
  allowed=[d['path']for d in routes['files']+routes['author_files']]+[routes['dist']['path']]
  firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs',Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu')])
  from kn_native import load,receiver_codec
  from selective_receiver import render as selective_render
  def boundary():assert time.monotonic()-beg<r['max_seconds']and not(L/'STOP').exists()
  n,original_render=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));codec=receiver_codec(n);rows=[]
  with n.torch.no_grad():
   for j in jobs:
    boundary();assert n.torch.cuda.mem_get_info()[0]>=2<<30
    token_key=j['wire_key']if mode=='SINGLE'else j['key']
    def realize():
     if mode=='SINGLE':
      p=j['projection'];assert digest(p)==j['wire_key']and p['status']=='SOURCE_INPUT'and p['codec']=='NEST'
      v=codec.decode(p['family'],np.asarray(p['bits'],np.uint8),p['m'],p['K']);tokens=v['received_tokens']
      assert v['canonical']and not v['receiver_truth_used']and not v['transmitter_probability_table_used']
      state=state_identity(p['m'],tokens[:OFFSETS[p['m']]].tolist(),list(range(p['K'])),tokens[OFFSETS[p['m']]:].tolist(),j['preprocessing'],r['route'])
      image=original_render(n,tokens.copy(),p['m'],p['K']);trace=None;role='NORMAL_CONTINUOUS'
     else:
      state=j['state'];assert digest(state)==j['key']and state['route']==r['route']
      image,trace=selective_render(n,state);role=j['role']
     assert image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()and 0<=image.min()<=image.max()<=1
     key=digest(state);ip=out/'images'/(key+'.npz')
     with ip.open('xb')as f:np.savez(f,image=image);f.flush();os.fsync(f.fileno())
     record=dict(key=key,wire_key=j.get('wire_key'),state=state,image=pin(ip),pixel_sha256=asha(image),role=role,trace=trace,source_truth_used=False,TX_CDF_used=False)
     save(ip.with_suffix('.json'),record);return record
    rows.append(ledger.call('receive_render'if mode=='SINGLE'else'selective_render',token_key,realize))
    if len(rows)%20==0:print('IMAGE_SAVED',mode,len(rows),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE_IMAGES',mode=mode,request=rp,outputs=pin(out/'outputs.json'),count=len(rows),ledger=ledger.summary(),seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
