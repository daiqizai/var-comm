"""Independent received-prefix arithmetic RX, unchanged frozen VAR/Dc math."""
from shared import *
from adaptive_plan import NEST_ENDPOINTS,nest_family,SIZES
import numpy as np
def main(public):
 cfg=read(public);out=Path(cfg['run'])/'rx';out.mkdir();cache(out);os.sched_setaffinity(0,{0,1});ledger=None
 try:
  routes=read(verify(cfg['model_routes']));static=cfg['static'];jobs=read(verify(cfg['jobs']));assert 0<len(jobs)<=cfg['new_jobs_cap']
  sys.path.insert(0,str(R/'experiments/t1c-m10-test-20261016'));sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
  from tc_receiver import ReceiverCodec
  from tb_native import build
  from runtime_boundary import firewall
  from h800_ep48_link_core_v1 import bind_static
  from ep_source_codec import InvalidSourceStream
  ledger=Ledger(out/'calls.sqlite',dict(model_load=1,source_rx=len(jobs),var_render=len(jobs),prior_scale=20*len(jobs),decoder_forward=len(jobs)))
  allowed=[v['path'] for v in routes['files']+routes['author_files']]+[routes['dist']['path'],static['completion']['path'],static['table']['path']]+[j['input']['path'] for j in jobs]
  firewall(out,allowed,[BASE,RT/'assets',RT/'qualification',R/'results',R/'outputs']);start=time.time();last=[0.]
  def boundary():
   assert time.time()-start<cfg['max_seconds']
   if time.time()-last[0]>10:
    assert int(subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>=2048;last[0]=time.time()
  n,render=ledger.call('model_load','once',lambda:build(routes,ledger,boundary,out,cfg['gpu_pci']));codec=ReceiverCodec(R,native=n);bind_static(codec,static)
  (out/'images').mkdir();outputs=[]
  for j in jobs:
   boundary();data=verify(j['input']).read_bytes();bits=np.unpackbits(np.frombuffer(data,np.uint8))[:j['bit_length']].copy();key=j['key'];m,K=NEST_ENDPOINTS[j['endpoint']]
   def decode():
    try:return codec.decode(nest_family(j['endpoint']),bits,m,K)
    except InvalidSourceStream as e:return dict(parse_error=str(e))
   actual=ledger.call('source_rx',key,decode);image=np.full((3,256,256),.5,np.float32);e=dict(source_input_key=key,endpoint=j['endpoint'],m=m,K=K,source_truth_used=False,TX_CDF_used=False,status='SOURCE_PARSE_FAIL')
   if 'parse_error' in actual:e['reason']=actual['parse_error']
   else:
    tokens=actual['received_tokens'];assert len(tokens)==sum(s*s for s in SIZES[:m])+K and actual['canonical'] and not actual['receiver_truth_used'] and not actual['transmitter_probability_table_used']
    def go():
     with n.torch.no_grad():return render(n,tokens.copy(),m,K)
    image=ledger.call('var_render',key,go);e.update(status='SOURCE_DECODED',received_tokens=tokens.tolist(),canonical=True)
   assert image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and image.min()>=0 and image.max()<=1
   ip=out/'images'/(key+'.npz')
   with ip.open('xb')as f:np.savez(f,image=image)
   e.update(image=pin(ip),float_sha256=hashlib.sha256(image.tobytes()).hexdigest());put(out/'images'/(key+'.json'),e);outputs.append(e)
   if len(outputs)%20==0:print(json.dumps(dict(completed=len(outputs),total=len(jobs))),flush=True)
  put(out/'completion.json',dict(status='COMPLETE',outputs=outputs,ledger=ledger.summary(),new_TX_passes=0,truth_reads=0))
 except BaseException:put(out/'failure.json',dict(error=traceback.format_exc(),ledger=None if ledger is None else ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(sys.argv[1])
