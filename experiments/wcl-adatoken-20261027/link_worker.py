"""Exactly two fixed development PHY events with received-profile counter derivation."""
from pathlib import Path
import sys,argparse,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adatoken-response-20261026'))
from common import *
from c_predclass_frame import runtime,choose,source_projection
from public_counter_rx import receive
import numpy as np
L=B/'wcl_adatoken_20261027_v1'

def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 scope=cr(r['development_scope']);assert scope['source_indices']==[95,21]and scope['snr_db']==10
 w=cr(r['source_actual_wait']);assert w['returncode']==0 and gone(w['process'])and gone(w['parent_process'])
 sources=cr(r['sources']);assert [s['source_index']for s in sources]==[95,21]
 assert r['caps']==dict(physical_events=2,packet_decodes=4)
 out=L/role;out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(range(6)));beg=time.monotonic();ledger=None
 save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 try:
  phy,rt=runtime(r)
  from t2_ledger import Ledger as PacketLedger
  ledger=PacketLedger(out/'packets.sqlite',expected,4);frames=[];inputs=[]
  profiles=cr(r['catalogue'])['profiles']
  for s in sources:
   assert time.monotonic()-beg<r['max_seconds']
   with np.load(checked(s['streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
   p,bits,attempts=choose(streams,profiles,4,'2/3',s['predicted_class'])
   public_base=12000000+4096*s['source_index'];tx_counter=public_base+p['profile_id']
   event=digest([scope['noise_namespace'],s['source_id'],10,2001])
   wave,tx=rt.transmit(p['profile_id'],bits,tx_counter)
   noise=phy.standard_noise(s['source_id'],2001);observed=wave+noise*10**(-10/20)
   op=out/(event+'.npz')
   with op.open('xb')as f:np.savez(f,observed=observed);f.flush();os.fsync(f.fileno())
   rx=receive(phy,rt,observed,10,public_base,ledger,event)
   projection=source_projection(rx)
   if projection['status']=='SOURCE_INPUT':
    payload=np.array(rx['body']['payload'],np.uint8);container_sha=asha(payload)
    inp=dict(key=digest(dict(prefix=projection['prefix'],container_sha256=container_sha)),prefix=projection['prefix'],bits=projection['bits'],payload_sha256=projection['payload_sha256'],received_class_bits=payload[:10].tolist(),container_sha256=container_sha)
    inputs.append(inp);rxkey=inp['key']
   else:rxkey=None
   v=dict(source_id=s['source_id'],source_index=s['source_index'],snr_db=10,noise_seed=2001,noise_namespace=scope['noise_namespace'],standard_noise_sha256=asha(noise),event_id=event,public_base=public_base,transmission=tx,actual_RX=rx,source_projection=projection,rx_key=rxkey,observation=pin(op),length_checks=attempts,source_truth_supplied_to_RX=False)
   save(out/(event+'.json'),v);frames.append(v)
  assert len(frames)==2 and ledger.snapshot()['unresolved']==0
  save(out/'frames.json',frames);save(out/'received_inputs.json',inputs)
  save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,physical_events=2,logical_events=2,frames=pin(out/'frames.json'),received_inputs=pin(out/'received_inputs.json'),packet_ledger=ledger.snapshot(),elapsed_seconds=time.monotonic()-beg))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
 finally:
  if ledger is not None:ledger.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
