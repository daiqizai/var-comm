"""One TX probability traversal per source, independently finished nested grids."""
from common import *
import argparse,traceback,subprocess
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 out=L/role;out.mkdir(parents=True,exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,{0,1});beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps'])
 save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 try:
  import numpy as np
  sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'));from kn_native import load
  routes=cr(r['model_routes']);sources=cr(r['sources']);rows=[]
  def boundary():
   assert time.monotonic()-beg<r['max_seconds']and not(L/'STOP').exists()
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']))
  from ep_source_codec import PartialSourceCodec,WHOLE_FAMILY
  from t1_entropy_core import source_tokens
  from t6_source_codec import SourceCodec10
  ns=dict(R=R,OFFSETS=OFFSETS,ENDS=grids()['8'],asha=asha,np=np,PartialSourceCodec=PartialSourceCodec,WHOLE_FAMILY=WHOLE_FAMILY,source_tokens=source_tokens,SourceCodec10=SourceCodec10)
  definitions(checked(r['audit_codec']),{'AuditCodec'},ns);codec=ns['AuditCodec'](R,native=n)
  for s in sources:
   boundary();dest=out/('source_%04d'%s['source_index']);dest.mkdir()
   with np.load(checked(s['archive']),allow_pickle=False)as z:tokens=z['tokens'].copy()
   assert tokens.dtype==np.int64 and tokens.shape==(680,) and hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==s['tokens_sha256']
   def encode_commit():
    streams,trace=codec.encode_all(tokens)
    p=dest/'streams.npz'
    with p.open('xb')as f:np.savez(f,**{'m%d_K%d'%k:v['bits']for k,v in streams.items()});f.flush();os.fsync(f.fileno())
    save(dest/'trace.json',trace)
    info=dict(source_id=s['source_id'],source_index=s['source_index'],streams=pin(p),trace=pin(dest/'trace.json'),lengths=[dict(m=m,K=K,bits=len(v['bits']),finish_bits=v['finish_bits'])for(m,K),v in streams.items()],probability_passes=1,tokenizer_calls=0)
    save(dest/'encoded_commit.json',info);return info
   info=ledger.call('source_probability_pass',s['source_id'],encode_commit)
   with np.load(checked(s['current_streams']),allow_pickle=False)as old,np.load(info['streams']['path'],allow_pickle=False)as new:
    comparisons=[dict(endpoint=k,equal=bool(np.array_equal(old[k],new[k])),old_bits=len(old[k]),new_bits=len(new[k]))for k in old.files]
   assert len(comparisons)==25
   save(dest/'old25_byte_regression.json',comparisons)
   if not all(v['equal']for v in comparisons):raise RuntimeError('BLOCKED_OLD25_BYTE_REGRESSION: source commit retained; no repeat allowed')
   rows.append(info);print('SOURCE_COMPLETE',s['source_index'],len(rows),flush=True)
  save(out/'sources.json',rows);save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,sources=pin(out/'sources.json'),count=len(rows),ledger=ledger.summary(),new_tokenizer=0,elapsed_seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
