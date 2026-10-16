"""One m10 source traversal per fixed test source; no RX, image or channel call."""
from tc_common import *
import numpy as np,subprocess,traceback
def main(expected):
 r=registration('source',expected);sources=checkread(r['sources']);out=T/'source';out.mkdir();runtime_cache(out);os.sched_setaffinity(0,{0,1})
 start=time.monotonic();last=[0.];ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,registration_sha256=expected))
 def boundary():
  assert time.monotonic()-start<r['max_seconds'] and not(T/'STOP').exists()
  if time.monotonic()-last[0]>10:
   free=int(subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=2048;last[0]=time.monotonic()
 try:
  from tb_native import build
  from m10_codec import AuditCodec
  routes=checkread(r['model_routes']);n,_=ledger.call('model_load','once',lambda:build(routes,ledger,boundary,out,r['gpu_pci']));codec=AuditCodec(R,native=n)
  rows=[];outputs=[];(out/'streams').mkdir()
  for s in sources:
   boundary();i=s['source_global_index'];checked(s['archive']);checked(s['old_streams'])
   with np.load(s['archive']['path'],allow_pickle=False) as z:tokens=z['tokens'].copy()
   assert hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==s['tokens_sha256']
   encoded,trace=ledger.call('source_tx',i,lambda:codec.encode_all(tokens));p=out/'streams'/f'{i:04d}.npz'
   with p.open('xb') as f:np.savez(f,**{f'm{m}_K{k}':v['bits'] for (m,k),v in encoded.items()})
   save(out/'streams'/f'{i:04d}_fixed_output.json',dict(archive=pin(p),source_global_index=i,cdf_trace=trace))
   with np.load(s['old_streams']['path'],allow_pickle=False) as z:matches={f'm{m}_K{k}':bool(np.array_equal(v['bits'],z[f'm{m}_K{k}'])) for (m,k),v in encoded.items() if m<10}
   save(out/'streams'/f'{i:04d}_regression.json',dict(source_global_index=i,old24_exact=matches));assert len(matches)==24 and all(matches.values()),('BLOCKED_OLD_ENDPOINT_REGRESSION',i,matches)
   v=encoded[10,0];row=dict(population=s['population'],source_index=s['source_index'],source_id=s['source_id'],tokens_sha256=s['tokens_sha256'],arithmetic_bits=len(v['bits']),finish_bits=v['finish_bits'],fits_uint13=len(v['bits'])<=8191,fits_4751=len(v['bits'])<=4751,payload_sha256=asha(v['bits']),old24_byte_exact=True)
   save(out/'streams'/f'{i:04d}_length.json',row);rows.append(row);outputs.append(dict(s,current_streams=pin(p),m10=row))
   progress(out,dict(completed=i+1,total=600));print('SOURCE_COMPLETE',i+1,600,flush=True)
  assert ledger.summary()['unresolved']==0;csvsave(out/'m10_lengths.csv',rows);save(out/'sources.json',outputs)
  save(out/'completion.json',dict(status='COMPLETE_T1C_SOURCE600',registration_sha256=expected,ledger=ledger.summary(),sources=pin(out/'sources.json'),lengths=pin(out/'m10_lengths.csv'),counts_by_population={pop:dict(sources=sum(x['population']==pop for x in rows),fits_4751=sum(x['population']==pop and x['fits_4751'] for x in rows)) for pop in ('common500','new100')},source_RX=0,channel=0,images=0,quality_scores=0,automatic_successor=False,seconds=time.monotonic()-start))
 except BaseException:save(out/'failure.json',dict(status='BLOCKED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),ledger=ledger.summary()));raise
if __name__=='__main__':main(sys.argv[1])
