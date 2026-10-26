"""Decode already-committed new49 streams independently; no TX or truth reads."""
from common import *
import argparse,traceback,numpy as np
from a_codec import receiver_class
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 inputs=cr(r['inputs']);routes=cr(r['model_routes']);out=L/role;out.mkdir(exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,{0,1});beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 try:
  sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'));sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
  from kn_native import load
  from runtime_boundary import firewall
  firewall(out,[d['path']for d in routes['files']+routes['author_files']]+[routes['dist']['path']],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  def boundary():assert time.monotonic()-beg<r['max_seconds']and not(L/'STOP').exists()
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));C,proof=receiver_class();save(out/'codec_proof.json',proof);codec=C(R,native=n);rows=[]
  for job in inputs:
   boundary();bits=np.array(job['bits'],np.uint8);assert asha(bits)==job['payload_sha256']
   def decode():
    value=codec.decode('EC_VAR_PARTIAL'if job['K']else'EC_VAR_WHOLE',bits,job['m'],job['K']);tokens=value.pop('received_tokens')
    p=out/(job['key']+'.npz')
    with p.open('xb')as f:np.savez(f,received_tokens=tokens);f.flush();os.fsync(f.fileno())
    v=dict(key=job['key'],m=job['m'],K=job['K'],payload_sha256=asha(bits),received=pin(p),token_sha256=asha(tokens),cdf_trace=codec.cdf_trace,evidence=value)
    save(p.with_suffix('.json'),v);return v
   rows.append(ledger.call('source_rx',job['key'],decode));print('A_RX_COMMITTED',len(rows),len(inputs),flush=True)
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='INDEPENDENT_RX_COMPLETE_AWAITING_POST_OUTPUT_AUDIT',outputs=pin(out/'outputs.json'),ledger=ledger.summary(),TX_calls=0,render_calls=0,source_reads=0,elapsed_seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
