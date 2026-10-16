"""Independent arithmetic RX and frozen VAR/Dc reconstruction, no TX truth."""
from tb_common import *
import traceback,subprocess,numpy as np
from ep_source_codec import PartialSourceCodec,InvalidSourceStream
from t6_source_codec import SourceCodec10
class ReceiverCodec(PartialSourceCodec):
 def decode(self,family,payload,m,K=0):
  if m==10:
   assert family=='EC_VAR_WHOLE' and K==0
   return SourceCodec10.decode(self,family,payload,10)
  return super().decode(family,payload,m,K)

def firewall(out,allowed):
 from runtime_boundary import firewall as install
 install(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])

def run(expected):
 from recovery_binding import verify_recovery
 verify_recovery(sys.argv[2])
 r,out=stage_start('receiver_recovery_v1',expected);ledger=None;start=time.monotonic();last=[0.]
 try:
  done=read(T/'phy/completion.json');assert done['registration_sha256']==expected
  inputs=checkread(done['rx_projection']);routes=checkread(r['model_routes']);static=r['static']
  caps=dict(model_load=1,source_rx=r['caps']['source_rx'],var_render=r['caps']['renders'],prior_scale=20*r['caps']['source_rx'],decoder_forward=r['caps']['renders'])
  ledger=Ledger(out/'calls.sqlite',caps)
  allow=[x['path']for x in routes['files']+routes['author_files']]+[routes['dist']['path'],static['completion']['path'],static['table']['path']]
  allow += [x['reuse']['image_archive']['path']for x in inputs if x['reuse']]
  # Projection is already fixed in memory; source/packet/score archives are denied.
  firewall(out,allow)
  def boundary():
   assert time.monotonic()-start<r['seconds']['receiver'] and not(T/'STOP').exists()
   if time.monotonic()-last[0]>10:
    free=int(subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=2048;last[0]=time.monotonic()
  from tb_native import build
  n,render=ledger.call('model_load','once',lambda:build(routes,ledger,boundary,out,r['gpu_pci']))
  codec=ReceiverCodec(R,native=n)
  from h800_ep48_link_core_v1 import bind_static
  bind_static(codec,static)
  (out/'images').mkdir();results={};new=0;reused=0;gray=0
  for ix,item in enumerate(inputs):
   boundary();key=item['source_input_key'];rx=item['actual_RX'];assert source_key(rx)==key
   if item['reuse']:
    d=item['reuse'];checked(d['image_archive'])
    with np.load(d['image_archive']['path'],allow_pickle=False)as z:image=z['image'].copy()
    assert image.dtype==np.float32 and image.shape==(3,256,256) and asha(image)==d['image_sha256']
    results[key]=dict(image_archive=d['image_archive'],image_sha256=d['image_sha256'],origin='CLOSED_EXACT_RX_INPUT',old_reconstruction_result=d['reconstruction_result']);reused+=1
   else:
    body=rx['body'];p=rx['rx_profile'];evidence=dict(source_decode_called=False,render_called=False,source_status=rx['status'],received_tokens=None,receiver_truth_used=False,TX_CDF_used=False)
    image=np.full((3,256,256),.5,dtype=np.float32)
    if rx['header']['header_ok'] and body and body['crc_accepted'] and body['parser_accepted']:
     def decode():
      try:return codec.decode(p['family'],np.asarray(body['payload'],dtype=np.uint8),p['m'],p['K'])
      except InvalidSourceStream as e:return dict(parse_error=str(e))
     actual=ledger.call('source_rx',key,decode);evidence['source_decode_called']=True
     if 'parse_error'in actual:evidence.update(source_status='SOURCE_PARSE_REJECT',parse_error=actual['parse_error'])
     else:
      tokens=actual['received_tokens'];assert actual['canonical'] and actual['zero_extension_reads']==30 and not actual['receiver_truth_used'] and not actual['transmitter_probability_table_used']
      assert len(tokens)==OFFSETS[p['m']]+p['K']
      def reconstruct():
       with n.torch.no_grad():return render(n,tokens.copy(),p['m'],p['K'])
      image=ledger.call('var_render',key,reconstruct);evidence.update(source_status='ARITHMETIC_SOURCE_DECODED',received_tokens=tokens.tolist(),m=p['m'],K=p['K'],canonical=True,render_called=True)
    assert image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and (image>=0).all()and(image<=1).all()
    ip=out/'images'/f'{new:05d}.npz'
    with ip.open('xb')as f:np.savez(f,image=image)
    ep=out/'images'/f'{new:05d}.json';save(ep,dict(source_input_key=key,evidence=evidence,image_archive=pin(ip),image_sha256=asha(image)))
    results[key]=dict(image_archive=pin(ip),image_sha256=asha(image),origin='ACTUAL_NEW',evidence=pin(ep));new+=1;gray+=int(not evidence['render_called'])
   if(ix+1)%100==0:progress(out,dict(unique_inputs=ix+1,new=new,reused=reused,ledger=ledger.summary()))
  assert ledger.summary()['unresolved']==0
  save(out/'images.json',results);save(out/'completion.json',dict(status='COMPLETE_T1B_RX_IMAGES',registration_sha256=expected,images=pin(out/'images.json'),new_images=new,reused_images=reused,fixed_gray_new=gray,ledger=ledger.summary(),seconds=time.monotonic()-start,truth_reads=0,TX_CDF_reads=0))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=None if ledger is None else ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':run(sys.argv[1])
