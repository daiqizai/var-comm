"""24 fixed Kodak sources, one TX probability traversal each, 25 finished streams."""
from kn_common import *
import numpy as np,subprocess,traceback
def main(expected):
 r,out=start('source',expected);runtime_cache(out);os.sched_setaffinity(0,{0,1});beg=time.monotonic();last=[0.]
 ledger=Ledger(out/'calls.sqlite',dict(model_load=1,tokenizer=24,source_tx=24,prior_scale=240,decoder_forward=0))
 def boundary():
  assert time.monotonic()-beg<r['max_seconds']['source']and not(L/'STOP').exists()
  if time.monotonic()-last[0]>10:
   assert int(subprocess.check_output(['nvidia-smi','-i',r['gpu']['uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>=2048;last[0]=time.monotonic()
 try:
  from PIL import Image
  from kn_native import load,source_codec,tokenizer,choose
  routes=cr(r['model_routes']);n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['gpu']['pci_bus_id']));codec=source_codec(n)
  sources=cr(r['population'])['sources'];results=[];endpoint_rows=[]
  for s in sources:
   boundary();i=s['source_index'];sd=out/f'i{i:02d}';sd.mkdir()
   pixels=np.asarray(Image.open(checked(s['png'])).convert('RGB'),dtype=np.uint8).transpose(2,0,1).copy()
   assert pixels.shape==(3,256,256)and hashlib.sha256(pixels.tobytes()).hexdigest()==s['pixels_chw_uint8_sha256']
   reuse=r['token_reuse'].get(str(i))
   if reuse:
    with np.load(checked(reuse['archive']),allow_pickle=False)as z:tokens=z[reuse['key']].copy().astype(np.int64)
    assert asha(tokens)==reuse['tokens_sha256'];origin='EXACT_PRIOR_TOKEN_BYTES'
   else:
    tokens=ledger.call('tokenizer',i,lambda:tokenizer(n,pixels));origin='MISSING_OLD_TOKEN_CACHE'
   assert tokens.shape==(680,)and tokens.dtype==np.int64 and np.all((tokens>=0)&(tokens<4096))
   with (sd/'source.npz').open('xb')as f:np.savez(f,tokens=tokens,pixels=pixels)
   encoded,trace=ledger.call('source_tx',i,lambda:codec.encode_all(tokens))
   with (sd/'streams.npz').open('xb')as f:np.savez(f,**{f'm{m}_K{k}':v['bits']for(m,k),v in encoded.items()})
   assert len(encoded)==25
   save(sd/'source_output_fixed.json',dict(source_index=i,source_id=s['source_id'],source=pin(sd/'source.npz'),streams=pin(sd/'streams.npz'),tokens_sha256=asha(tokens),cdf_trace=trace,token_origin=origin))
   streams={ep:v['bits']for ep,v in encoded.items()}
   for family in ('WHOLE','PARTIAL'):
    for snr in (4,10,19):
     ep,cap,target=choose(streams,family,snr);bits=streams[ep]
     endpoint_rows.append(dict(source_index=i,source_id=s['source_id'],family=family,snr_db=snr,target_m=target[0],target_K=target[1],actual_m=ep[0],actual_K=ep[1],terminated_arithmetic_bits=len(bits),finish_bits=encoded[ep]['finish_bits'],source_capacity_bits=cap,utilization=len(bits)/cap,q={4:2,10:4,19:6}[snr],nominal_rate='5/6'if snr==19 else'2/3',stream_file=str(sd/'streams.npz'),stream_file_sha256=sha(sd/'streams.npz'),stream_key=f'm{ep[0]}_K{ep[1]}',payload_sha256=asha(bits),tokens_sha256=asha(tokens)))
   results.append(dict(s,source_archive=pin(sd/'source.npz'),streams=pin(sd/'streams.npz'),tokens_sha256=asha(tokens),token_origin=origin,completion=pin(sd/'source_output_fixed.json')))
   print('SOURCE_COMPLETE',i+1,24,flush=True)
  assert ledger.summary()['unresolved']==0 and len(results)==24 and len(endpoint_rows)==144
  save(out/'sources.json',results);csvsave(out/'endpoint_utilization.csv',endpoint_rows)
  save(out/'completion.json',dict(status='COMPLETE_SOURCE24',registration_sha256=expected,sources=pin(out/'sources.json'),endpoint_utilization=pin(out/'endpoint_utilization.csv'),ledger=ledger.summary(),seconds=time.monotonic()-beg,source_count=24,finished_stream_count=600,source_RX=0,channel_events=0,images=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(sys.argv[1])
