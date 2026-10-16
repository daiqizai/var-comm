"""RX-only independent arithmetic decoding and frozen VAR on new observations."""
from tc_common import *
import traceback,numpy as np,subprocess
from ep_source_codec import PartialSourceCodec,InvalidSourceStream
from t6_source_codec import SourceCodec10
from tc_protocol import closed_wait
class ReceiverCodec(PartialSourceCodec):
 def decode(self,family,payload,m,K=0):
  if m==10:
   assert family=='EC_VAR_WHOLE' and K==0
   return SourceCodec10.decode(self,family,payload,10)
  return super().decode(family,payload,m,K)
def run(expected):
 r=registration('link',expected);out=T/'receiver';out.mkdir();runtime_cache(out);os.sched_setaffinity(0,{0,1});ledger=None;start=time.monotonic();last=[0.]
 try:
  w=read(T/'phy_control/actual_wait.json');closed_wait(w,gone)
  d=read(T/'phy/completion.json');assert d['registration_sha256']==expected;inputs=checkread(d['rx_projection']);routes=checkread(r['model_routes']);static=r['static']
  caps=dict(model_load=1,source_rx=len(inputs),var_render=len(inputs),prior_scale=20*len(inputs),decoder_forward=len(inputs));ledger=Ledger(out/'calls.sqlite',caps)
  allow=[v['path'] for v in routes['files']+routes['author_files']]+[routes['dist']['path'],static['completion']['path'],static['table']['path']]
  from runtime_boundary import firewall
  firewall(out,allow,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  def boundary():
   assert time.monotonic()-start<r['max_seconds'] and not(T/'STOP').exists()
   if time.monotonic()-last[0]>10:
    assert int(subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>=2048;last[0]=time.monotonic()
  from tb_native import build
  n,render=ledger.call('model_load','once',lambda:build(routes,ledger,boundary,out,r['gpu_pci']));codec=ReceiverCodec(R,native=n)
  from h800_ep48_link_core_v1 import bind_static
  bind_static(codec,static);(out/'images').mkdir();results={}
  for i,item in enumerate(inputs):
   boundary();key=item['source_input_key'];rx=item['actual_RX'];assert source_key(rx)==key;b=rx['body'];p=rx['rx_profile'];image=np.full((3,256,256),.5,dtype=np.float32)
   e=dict(source_decode_called=False,render_called=False,source_status=rx['status'],received_tokens=None,receiver_truth_used=False,TX_CDF_used=False)
   if rx['header']['header_ok'] and b and b['crc_accepted'] and b['parser_accepted']:
    def decode():
     try:return codec.decode(p['family'],np.asarray(b['payload'],dtype=np.uint8),p['m'],p['K'])
     except InvalidSourceStream as err:return dict(parse_error=str(err))
    actual=ledger.call('source_rx',key,decode);e['source_decode_called']=True
    if 'parse_error' in actual:e.update(source_status='SOURCE_PARSE_REJECT',parse_error=actual['parse_error'])
    else:
     tokens=actual['received_tokens'];assert actual['canonical'] and actual['zero_extension_reads']==30 and not actual['receiver_truth_used'] and not actual['transmitter_probability_table_used']
     assert len(tokens)==OFFSETS[p['m']]+p['K']
     def reconstruct():
      with n.torch.no_grad():return render(n,tokens.copy(),p['m'],p['K'])
     image=ledger.call('var_render',key,reconstruct);e.update(source_status='ARITHMETIC_SOURCE_DECODED',received_tokens=tokens.tolist(),m=p['m'],K=p['K'],canonical=True,render_called=True)
   assert image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and image.min()>=0 and image.max()<=1
   ip=out/'images'/f'{i:04d}.npz'
   with ip.open('xb') as f:np.savez(f,image=image)
   ep=out/'images'/f'{i:04d}.json';save(ep,dict(source_input_key=key,evidence=e,image_archive=pin(ip),image_sha256=asha(image)))
   results[key]=dict(image_archive=pin(ip),image_sha256=asha(image),image_key='image',image_slot=None,evidence=pin(ep),origin='ACTUAL_NEW')
   if (i+1)%20==0:progress(out,dict(completed=i+1,total=len(inputs),ledger=ledger.summary()))
  assert ledger.summary()['unresolved']==0;save(out/'images.json',results);save(out/'completion.json',dict(status='COMPLETE_T1C_RX',registration_sha256=expected,images=pin(out/'images.json'),unique_images=len(results),ledger=ledger.summary(),seconds=time.monotonic()-start,truth_reads=0,TX_CDF_reads=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=None if ledger is None else ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':run(sys.argv[1])
