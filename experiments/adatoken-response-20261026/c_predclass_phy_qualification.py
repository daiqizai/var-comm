"""Exactly24 real qualification frames; frozen outputs precede any TX audit."""
from common import *
from c_predclass_frame import runtime,choose,CANDIDATES,FAMILY,NULL,source_projection
import argparse,traceback,numpy as np
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 out=L/role;out.mkdir(exist_ok=False);beg=time.monotonic();save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected));ledger=None
 try:
  phy,rt=runtime(r);from t2_ledger import Ledger as PacketLedger
  ledger=PacketLedger(out/'packets.sqlite',expected,48);sources=cr(r['sources']);jobs=cr(r['jobs']);assert len(jobs)==24;rows=[]
  for job in jobs:
   assert time.monotonic()-beg<r['max_seconds'];s=sources[job['source_position']]
   with np.load(checked(s['streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
   p,bits,attempts=choose(streams,rt.catalogue['profiles'],job['q'],job['rate'],s['predicted_class']);wave,tx=rt.transmit(p['profile_id'],bits,job['counter']);standard=phy.standard_noise(s['source_id'],job['noise_seed']);observed=wave+standard*10**(-job['snr_db']/20)
   op=out/('observed_%02d.npz'%len(rows))
   with op.open('xb')as f:np.savez(f,observed=observed);f.flush();os.fsync(f.fileno())
   rx=rt.receive(observed,job['snr_db'],job['counter'],rt.profiles,ledger,'C_QUAL/%02d'%len(rows),phase='fixed24_qualification')
   v=dict(job=job,source_id=s['source_id'],transmission=tx,paid_class_bits=0 if p['family']==NULL else 10,container_bits=len(bits),actual_arithmetic_bits=0 if p['family']==NULL else len(bits)-10,source_projection=source_projection(rx),TX_payload_sha256=asha(bits),actual_RX=rx,observation=pin(op),standard_noise_sha256=asha(standard),attempts=attempts,source_truth_supplied_to_RX=False)
   jp=out/('frame_%02d.json'%len(rows));save(jp,v);rows.append(pin(jp));print('C_PHY_QUAL_COMMITTED',len(rows),24,flush=True)
  assert len(rows)==24 and ledger.snapshot()['unresolved']==0
  # Audit after every received result has been fixed. No correction/rejection is fed back.
  audit=[]
  for d in rows:
   v=cr(d);rx=v['actual_RX'];body=rx['body'];accepted=bool(rx['header']['header_ok']and body and body['crc_accepted']and body['parser_accepted'])
   exact=accepted and rx['rx_profile']['profile_id']==v['transmission']['profile_id']and asha(np.array(body['payload'],np.uint8))==v['TX_payload_sha256']
   audit.append(dict(frame=d,accepted=accepted,post_output_payload_and_profile_exact=bool(exact),received_status=rx['status']))
  save(out/'frames.json',rows);save(out/'post_output_audit.json',audit);save(out/'completion.json',dict(status='QUALIFICATION24_PHY_CLOSED_AWAITING_REVIEW',logical_events=24,physical_events=24,packets=ledger.snapshot(),frames=pin(out/'frames.json'),audit=pin(out/'post_output_audit.json'),CUDA_initialized=rt.torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),packets=None if ledger is None else ledger.snapshot(),automatic_retry=False));raise
 finally:
  if ledger is not None:ledger.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
