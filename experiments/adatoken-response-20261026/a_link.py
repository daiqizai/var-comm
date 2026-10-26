"""Four fixed-MCS nested endpoint grids; actual common N1024 frame engine."""
from common import *
from fixed_frame import runtime
import argparse,traceback,numpy as np
def main(role,path,expected):
 assert sha(path)==expected;r=read(path);assert r['branch']=='A'
 for d in r['code']:checked(d)
 out=L/role;out.mkdir(exist_ok=False);beg=time.monotonic();save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected));ledger=None
 try:
  phy,rt=runtime(r);from t2_ledger import Ledger as PacketLedger
  ledger=PacketLedger(out/'packets.sqlite',expected,r['caps']['packets']);sources=cr(r['sources']);jobs=cr(r['physical_jobs']);logical=cr(r['logical_jobs']);assert len(logical)==3600
  (out/'physical').mkdir();(out/'observations').mkdir();rows=[];projection={};current=None;streams=None
  sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'));from tb_common import source_key
  for job in jobs:
   assert time.monotonic()-beg<r['max_seconds']and not(L/'STOP').exists();s=sources[job['source_position']]
   if current!=job['source_position']:
    with np.load(checked(s['streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
    current=job['source_position']
   bits=np.array([0,0],np.uint8)if job['null']else streams[job['stream_key']];assert asha(bits)==job['payload_sha256']and len(bits)==job['arithmetic_bits']
   wave,tx=rt.transmit(job['profile_id'],bits,job['public_frame_counter']);standard=phy.standard_noise(s['source_id'],job['noise_seed']);noise=standard*10**(-job['snr_db']/20);observed=wave+noise
   op=out/'observations'/('%04d.npz'%len(rows))
   with op.open('xb')as f:np.savez(f,observed=observed);f.flush();os.fsync(f.fileno())
   rx=rt.receive(observed,job['snr_db'],job['public_frame_counter'],rt.profiles,ledger,'A_GRID/%04d'%len(rows),phase='A_CONTROLLED_GRID')
   sk=source_key(rx);projection.setdefault(sk,dict(source_input_key=sk,actual_RX=rx));jp=out/'physical'/('%04d.json'%len(rows))
   save(jp,dict(event=job,transmission=tx,payload_sha256=asha(bits),waveform_sha256=asha(wave),standard_noise_sha256=asha(standard),noise_sha256=asha(noise),observation_sha256=asha(observed),observation=pin(op),actual_RX=rx,source_input_key=sk,catalogue=r['catalogue'],source_truth_supplied_to_RX=False))
   rows.append(dict(event_key=job['event_key'],physical_frame=pin(jp),source_input_key=sk))
   if len(rows)%50==0:print('A_PHYSICAL_COMMITTED',len(rows),len(jobs),flush=True)
  assert ledger.snapshot()['unresolved']==0 and len(rows)==len(jobs)and len(projection)<=3551 and not rt.torch.cuda.is_initialized()
  index={v['event_key']:v for v in rows};alllogical=[dict(j,**{k:v for k,v in index[j['event_key']].items()if k!='event_key'})for j in logical]
  save(out/'physical_frames.json',rows);save(out/'logical_frames.json',alllogical);save(out/'rx_projection.json',list(projection.values()))
  save(out/'completion.json',dict(status='A_PHY_COMPLETE_AWAITING_RX',registration_sha256=expected,logical_frames=len(alllogical),physical_events=len(rows),within_grid_exact_reuse=len(alllogical)-len(rows),packets=ledger.snapshot(),unique_RX_inputs=len(projection),logical=pin(out/'logical_frames.json'),rx_projection=pin(out/'rx_projection.json'),physical=pin(out/'physical_frames.json'),elapsed_seconds=time.monotonic()-beg,CUDA_initialized=False))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),packets=None if ledger is None else ledger.snapshot(),automatic_retry=False));raise
 finally:
  if ledger is not None:ledger.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
