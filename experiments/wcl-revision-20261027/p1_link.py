"""Actual independent receive at true SNR; reuse only exact frozen 10 dB events."""
from p1_common import *
import traceback
def main(expected):
 r=registration(expected);out=L/'link';out.mkdir(exist_ok=False);save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv));start=time.monotonic();ledger=None
 try:
  imports();import ep_phy as phy,tb_plan
  from tb_common import source_key
  from t2_ledger import Ledger
  rt=phy.create_runtime(R,r['phy_qualification']['path']);cat=cr(r['catalogue']);assert tb_plan.extend_catalogue(rt.catalogue)==cat and rt.backend.identity==r['backend_identity']
  rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']}
  ledger=Ledger(out/'packets.sqlite',expected,3600);noise=np.load(checked(r['noise']),allow_pickle=False);assert ash(noise)==r['noise_array_sha256']
  sources=cr(r['sources']);jobs=cr(r['jobs']);(out/'physical').mkdir();(out/'observations').mkdir();rows=[];waves={};streams={};proofs=[];projection={};n=0
  for i,j in enumerate(jobs):
   assert time.monotonic()-start<r['max_seconds']and not(L/'STOP').exists()
   s=sources[j['rank']];wk=(j['source_id'],j['family'],j['noise_seed']);old=cr(j['old10']['physical_frame']);standard=noise[j['noise_index']]
   assert np.array_equal(standard,phy.standard_noise(j['source_id'],j['noise_seed']))
   if wk not in waves:
    if j['rank']not in streams:
     with np.load(checked(s['current_streams']),allow_pickle=False)as z:streams[j['rank']]={k:z[k].copy()for k in z.files}
    m,k,bits=choose(streams[j['rank']],j['family']);assert(m,k,ash(bits))==(j['m'],j['K'],j['payload_sha256'])
    waves[wk]=rt.transmit(j['profile_id'],bits,j['public_frame_counter'],j['session'])
   wave,tx=waves[wk];scaled=standard*10**(-j['snr_db']/20);observed=wave+scaled
   oldid=old.get('observation_identity',old)
   reuse=(j['snr_db']==10 and tx['transmitted_frame_sha256']==old['transmission']['transmitted_frame_sha256'] and ash(observed)==oldid['observation_sha256'] and tb_plan.directory_unchanged_on_old_event(old['actual_RX']['header']))
   if reuse:
    if 'noise_sha256'in oldid:assert ash(scaled)==oldid['noise_sha256']
    if 'standard_noise_sha256'in oldid:assert ash(standard)==oldid['standard_noise_sha256']
    rx=old['actual_RX'];pd=j['old10']['physical_frame'];origin='EXACT_OLD10_WAVE_NOISE_OBSERVATION_BACKEND_RX_DOMAIN';proofs.append(dict(logical=i,old=pd,wave=ash(wave),noise=ash(scaled),standard=ash(standard),observation=ash(observed),backend_identical=True,header_domain_equivalent=True))
   else:
    op=out/'observations'/f'{i:04d}.npz'
    with op.open('xb')as f:np.savez(f,observed=observed);f.flush();os.fsync(f.fileno())
    rx=rt.receive(observed,j['snr_db'],j['public_frame_counter'],rt.profiles,ledger,f'MISMATCH/{i:04d}',phase='TX_POLICY_MISMATCH',session=j['session'])
    p=out/'physical'/f'{i:04d}.json';save(p,dict(event={k:v for k,v in j.items()if k!='old10'},transmission=tx,observation=pin(op),standard_noise_sha256=ash(standard),noise_sha256=ash(scaled),observation_sha256=ash(observed),actual_RX=rx,truth_supplied_to_RX=False));pd=pin(p);origin='NEW_ACTUAL_RECEPTION';n+=1
   sk=source_key(rx);oldsk=source_key(old['actual_RX']);recon=None
   if sk==oldsk:recon=j['old10']['reconstruction'];state='EXACT_ACTUAL_RX_KEY_CACHED_FINAL_SOURCE_RESULT'
   elif not rx['header']['header_ok']or rx['body']is None or not rx['body']['parser_accepted']:state='FIXED_GRAY_REJECT'
   else:state='NEW_ACCEPTED_SOURCE_REQUIRES_INDEPENDENT_PARSE';projection.setdefault(sk,dict(source_input_key=sk,actual_RX=rx))
   rows.append(dict(event={k:v for k,v in j.items()if k!='old10'},physical_frame=pd,physical_origin=origin,actual_RX_input_key=sk,wire_status=rx['status'],RX_resolution=state,reconstruction=recon,old_metrics=j['old10']['metrics']if recon else None))
   if (i+1)%60==0:print('COMMITTED',i+1,'new',n,'packets',ledger.snapshot(),flush=True)
  assert len(rows)==1800 and ledger.snapshot()['unresolved']==0 and not rt.torch.cuda.is_initialized()
  save(out/'logical_frames.json',rows);save(out/'reuse_proofs.json',proofs);save(out/'new_source_inputs.json',list(projection.values()))
  save(out/'completion.json',dict(status='PHY_COMPLETE',registration_sha256=expected,logical=1800,new_physical=n,old10_reused=len(proofs),packets=ledger.snapshot(),unique_new_accepted_inputs=len(projection),frames=pin(out/'logical_frames.json'),reuse_proofs=pin(out/'reuse_proofs.json'),new_source_inputs=pin(out/'new_source_inputs.json'),backend_identity=rt.backend.identity,seconds=time.monotonic()-start))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),packets=None if ledger is None else ledger.snapshot(),automatic_retry=False));raise
 finally:
  if ledger is not None:ledger.close()
if __name__=='__main__':main(sys.argv[1])
