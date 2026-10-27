"""Metadata-only closure after the post-loop CUDA-initialization assertion.

No encode/decode, model call, channel simulation, or old-file mutation. All 1200
new received frames and 2400 packet results were committed before the assertion.
The frozen CPU backend constructor initializes CUDA with zero allocated bytes;
that process flag does not imply a CUDA decoder. Original exit1 is preserved.
"""
from p1_common import *
import sqlite3,collections
def main(expected):
 r=registration(expected);w=read(L/'link_control/actual_wait.json');assert w['returncode']==1 and w['actual_wait']and gone(w['process'])and gone(w['parent_process'])
 failure=read(L/'link/failure.json');assert failure['traceback'].endswith('AssertionError\n')and "not rt.torch.cuda.is_initialized()"in failure['traceback']and failure['packets']==dict(total=2400,unresolved=0,cap=3600)
 log=(L/'link_control/stdout.txt').read_text();assert log.splitlines()[-1]=="COMMITTED 1800 new 1200 packets {'total': 2400, 'unresolved': 0, 'cap': 3600}"
 out=L/'link_closed_v1';out.mkdir(exist_ok=False);beg=time.monotonic()
 db=sqlite3.connect('file:'+str(L/'link/packets.sqlite')+'?mode=ro',uri=True);assert db.execute('select hash,cap from config').fetchall()==[(expected,3600)]
 events=db.execute('select event_id,event,request,result,status from events').fetchall();assert len(events)==2400 and all(e[-1]=='COMPLETE'for e in events)
 packets={e[0]:(json.loads(e[1]),json.loads(e[2]),json.loads(e[3]))for e in events};db.close()
 imports();from tb_common import source_key
 import tb_plan
 jobs=cr(r['jobs']);noise=np.load(checked(r['noise']),allow_pickle=False);assert ash(noise)==r['noise_array_sha256']
 rows=[];proofs=[];projection={};counts=collections.Counter();pfiles=list((L/'link/physical').glob('*.json'));assert len(pfiles)==1200
 for i,j in enumerate(jobs):
  standard=noise[j['noise_index']];scaled=standard*10**(-j['snr_db']/20);old=cr(j['old10']['physical_frame'])
  if j['snr_db']!=10:
   p=L/'link/physical'/f'{i:04d}.json';d=read(p);assert d['event']=={k:v for k,v in j.items()if k!='old10'};pd=pin(p)
   with np.load(checked(d['observation']),allow_pickle=False)as z:obs=z['observed'].copy()
   assert ash(obs)==d['observation_sha256'] and ash(standard)==d['standard_noise_sha256'] and ash(scaled)==d['noise_sha256']
   rx=d['actual_RX'];he,hr,hv=packets[f'MISMATCH/{i:04d}:header'];be,br,bv=packets[f'MISMATCH/{i:04d}:body']
   assert hv==rx['header']and bv==rx['body']and hr['received_sha256']==ash(obs[:68])and br['received_sha256']==ash(obs[68:].astype(np.float32))
   assert hr['snr_db']==br['snr_db']==j['snr_db']and br['public_frame_counter']==j['public_frame_counter']
   origin='NEW_ACTUAL_RECEPTION'
  else:
   # The waveform contains exact float32 constellation values promoted to float64.
   # Remove saved 8 dB noise, round back to float32, and require its original hash.
   a=read(L/'link/physical'/f'{i-1:04d}.json');assert a['event']['snr_db']==8
   with np.load(checked(a['observation']),allow_pickle=False)as z:o8=z['observed'].copy()
   wave=(o8-standard*10**(-8/20)).astype(np.float32).astype(np.float64)
   assert ash(wave)==a['transmission']['transmitted_frame_sha256']==old['transmission']['transmitted_frame_sha256']
   obs=wave+scaled;oldid=old.get('observation_identity',old);assert ash(obs)==oldid['observation_sha256']and tb_plan.directory_unchanged_on_old_event(old['actual_RX']['header'])
   if 'standard_noise_sha256'in oldid:assert ash(standard)==oldid['standard_noise_sha256']
   if 'noise_sha256'in oldid:assert ash(scaled)==oldid['noise_sha256']
   rx=old['actual_RX'];pd=j['old10']['physical_frame'];origin='EXACT_OLD10_WAVE_NOISE_OBSERVATION_BACKEND_RX_DOMAIN'
   proofs.append(dict(logical=i,old=pd,saved8=pin(L/'link/physical'/f'{i-1:04d}.json'),wave=ash(wave),standard=ash(standard),noise=ash(scaled),observation=ash(obs),numeric_backend=r['backend_identity'],decoder_threads=2,header_directory_equivalent=True))
  sk=source_key(rx);oldsk=source_key(old['actual_RX']);recon=None
  if sk==oldsk:recon=j['old10']['reconstruction'];state='EXACT_ACTUAL_RX_KEY_CACHED_FINAL_SOURCE_RESULT'
  elif not rx['header']['header_ok']or rx['body']is None or not rx['body']['parser_accepted']:state='FIXED_GRAY_REJECT'
  else:state='NEW_ACCEPTED_SOURCE_REQUIRES_INDEPENDENT_PARSE';projection.setdefault(sk,dict(source_input_key=sk,actual_RX=rx))
  counts[j['family'],j['snr_db'],state]+=1
  rows.append(dict(event={k:v for k,v in j.items()if k!='old10'},physical_frame=pd,physical_origin=origin,actual_RX_input_key=sk,wire_status=rx['status'],RX_resolution=state,reconstruction=recon,old_metrics=j['old10']['metrics']if recon else None))
 assert len(rows)==1800 and len(proofs)==600
 save(out/'logical_frames.json',rows);save(out/'reuse_proofs.json',proofs);save(out/'new_source_inputs.json',list(projection.values()))
 save(out/'completion.json',dict(status='METADATA_CLOSED_AFTER_POSTLOOP_ASSERTION',original_actual_exit=1,original_failure=pin(L/'link/failure.json'),original_wait=pin(L/'link_control/actual_wait.json'),registration_sha256=expected,logical=1800,new_physical=1200,old10_reused=600,packets=2400,unresolved_packets=0,new_scientific_calls=0,frames=pin(out/'logical_frames.json'),reuse_proofs=pin(out/'reuse_proofs.json'),unique_new_accepted_inputs=len(projection),new_source_inputs=pin(out/'new_source_inputs.json'),counts=[dict(family=k[0],snr_db=k[1],resolution=k[2],count=n)for k,n in counts.items()],elapsed_seconds=time.monotonic()-beg))
 print(json.dumps(read(out/'completion.json')))
if __name__=='__main__':main(sys.argv[1])
