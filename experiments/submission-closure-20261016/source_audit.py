"""One TX/source on original cal1000, first8 independent RX, no channel/render."""
from common import *
import sys,os,tempfile,subprocess,traceback
def run(expected):
 reg=O/'source_registration.json';assert sha(reg)==expected;r=read(reg)
 for p in r['code']+r['old_controls']:checked(p)
 routes=read(checked(r['model_routes']));out=T/'m10_source';out.mkdir(exist_ok=False)
 for sub in ['runtime_cache/tmp','runtime_cache/inductor','runtime_cache/triton','streams','rx']:(out/sub).mkdir(parents=True)
 os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TMP=str(out/'runtime_cache/tmp'),TEMP=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'))
 tempfile.tempdir=os.environ['TMPDIR'];os.sched_setaffinity(0,{0,1});start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);latest=[0]
 save(out/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,registration=pin(reg)))
 def boundary():
  assert time.monotonic()-start<7200 and not (T/'STOP').exists()
  if time.monotonic()-latest[0]>10:
   free=int(subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=2048;latest[0]=time.monotonic()
 try:
  import numpy as np
  from native import build
  n,_=ledger.call('model_load','once',lambda:build(routes,ledger,boundary,out))
  from m10_codec import AuditCodec
  codec=AuditCodec(R,native=n);lengths=[];roundtrips=[]
  for source in r['sources']:
   boundary();i=source['source_index'];checked(source['archive']);checked(source['old_streams'])
   with np.load(source['archive']['path'],allow_pickle=False)as z:tokens=z['tokens'].copy()
   assert hashlib.sha256(f'int64:{len(tokens)}\0'.encode()+tokens.astype('<i8').tobytes()).hexdigest()==source['tokens_sha256']
   encoded,trace=ledger.call('source_tx',str(i),lambda:codec.encode_all(tokens));p=out/'streams'/f'{i:04d}.npz'
   with p.open('xb')as f:np.savez(f,**{f'm{m}_K{K}':v['bits']for(m,K),v in encoded.items()})
   save(out/'streams'/f'{i:04d}_fixed_output.json',dict(archive=pin(p),source_index=i,cdf_trace=trace))
   # Audit after outputs are fixed. Never supply this truth to RX.
   with np.load(source['old_streams']['path'],allow_pickle=False)as z:
    matches={f'm{m}_K{K}':bool(np.array_equal(v['bits'],z[f'm{m}_K{K}']))for(m,K),v in encoded.items()if m<10}
   save(out/'streams'/f'{i:04d}_regression.json',dict(source_index=i,old24_exact=matches))
   assert len(matches)==24 and all(matches.values()),('BLOCKED_OLD_ENDPOINT_REGRESSION',i,matches)
   v=encoded[10,0];bits=v['bits']
   lengths.append(dict(source_index=i,source_id=source['source_id'],preprocessing_id=source['preprocessing_id'],tokens_sha256=source['tokens_sha256'],m=10,K=0,tokens=680,arithmetic_bits=len(bits),finish_bits=v['finish_bits'],fits_uint13=len(bits)<=8191,bits_sha256=asha(bits),old24_byte_exact=True,archive_sha256=sha(p)))
   if i<8:
    rx=ledger.call('source_rx',str(i),lambda:codec.decode_m10(bits.copy()));rp=out/'rx'/f'{i:04d}.npz'
    with rp.open('xb')as f:np.savez(f,received_tokens=rx['received_tokens'])
    save(out/'rx'/f'{i:04d}_fixed_output.json',dict(actual_RX=pin(rp),receiver={k:v for k,v in rx.items()if k!='received_tokens'}))
    equal=bool(np.array_equal(tokens,rx['received_tokens']));proof=dict(source_index=i,source_id=source['source_id'],token_exact=equal,canonical=rx['canonical'],receiver_truth_used=rx['receiver_truth_used'],TX_CDF_used=rx['transmitter_probability_table_used'],fixed_output=pin(rp))
    save(out/'rx'/f'{i:04d}_audit.json',proof);roundtrips.append(proof);assert equal and proof['canonical']and not proof['receiver_truth_used']and not proof['TX_CDF_used']
   save(out/'streams'/f'{i:04d}_length.json',lengths[-1]);print('SOURCE_COMPLETE',i+1,1000,flush=True)
  assert len(lengths)==1000 and len(roundtrips)==8 and ledger.summary()['unresolved']==0
  csvsave(out/'m10_lengths.csv',lengths);save(out/'source_roundtrip_report.json',dict(status='PASS_FIRST8_INDEPENDENT_RX_AND_ALL_24000_OLD_ENDPOINTS',proofs=roundtrips))
  save(out/'completion.json',dict(status='COMPLETE_M10_CAL1000_SOURCE_ONLY',registration=pin(reg),counts=ledger.summary(),new_TX=1000,new_RX=8,new_prior_scales=10080,new_channel=0,new_images=0,new_metrics=0,seconds=time.monotonic()-start,outputs=[pin(out/'m10_lengths.csv'),pin(out/'source_roundtrip_report.json')]))
 except BaseException:
  save(out/'failure.json',dict(status='BLOCKED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),counts=ledger.summary()));raise
if __name__=='__main__':run(sys.argv[1])
