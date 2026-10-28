"""Freeze real TX inputs and P3 counters from the nominated strategies once."""
from block_common import *
from protocol import choose
from nominate_p2 import lengths
from phy_runtime import counter

def main():
 out=L/'p3_preparation';out.mkdir(exist_ok=False)
 scope=read(O/'scope.json');nom=read(O/'nomination.json');cat=cr(nom['catalog']);sources=cr(scope['source_manifest']);selected={r['candidate_id']for r in nom['selected']}
 assert nom['status']=='FROZEN_FOR_P3'
 tx={s['source_id']:s for s in read(O/'p1_source_delivery.json')['records']};jobs=[]
 for ordinal,s in enumerate(sources):
  t=tx[s['source_id']]
  with np.load(checked(s['nest_streams']),allow_pickle=False)as z:single={k:z[k].copy()for k in z.files}
  with np.load(checked(t['streams']),allow_pickle=False)as z:triple={k:z[k].copy()for k in z.files}
  for c in cat['candidates']:
   if c['candidate_id']not in selected:continue
   p=choose(c,cat['profiles'],lengths(s,t));payloads=[]
   if p['null']:payloads=[[]for _ in p['layouts']]
   elif p['kind']=='SINGLE':payloads=[single[f'm{p["m"]}_K{p["K"]}'].tolist()]
   else:payloads=[triple['C'+str(p['m'])].tolist()]+[triple[f'm{p["m"]}_K{p["K"]}_{g}'].tolist()if p['K']else []for g in ('A','B')]
   for snr in scope['actual_snr_db']:
    for rep in scope['replicates']:
     identity=dict(candidate_id=c['candidate_id'],source_index=s['source_index'],snr_db=snr,replicate=rep)
     jobs.append(dict(event_id=digest(identity),**identity,source_id=s['source_id'],ordinal=ordinal,profile_id=p['profile_id'],layouts=p['layouts'],null=p['null'],fallback=p['profile_id']!=c['profiles'][0],payloads=payloads,counter=counter('P3_FRAME',snr,s['source_index'],rep,c['candidate_id'])))
 assert len(jobs)==nom['physical_frames']<=4608 and sum(len(j['layouts'])for j in jobs)==nom['tx_body_blocks']<=11520
 assert len({j['event_id']for j in jobs})==len(jobs)
 save(out/'all_tx_frames.json',jobs)
 for i in range(4):save(out/f'shard{i:02d}.json',[j for j in jobs if j['ordinal']%4==i])
 save(O/'p3_plan.json',dict(status='FROZEN_NOT_LAUNCHED',scope=pin(O/'scope.json'),nomination=pin(O/'nomination.json'),tx_plan=pin(out/'all_tx_frames.json'),shards=[pin(out/f'shard{i:02d}.json')for i in range(4)],frames=len(jobs),tx_bodies=sum(len(j['layouts'])for j in jobs),header_decodes=len(jobs),maximum_body_decode_attempts=11520,random_rule='PCG64 SHA256 first128 bits; P3_AWGN,SNR,original source_index,replicate,candidate_id,FULL_FRAME; independent1024x2 AWGN; counter P3_FRAME with same public fields',public_RX_inputs='received full waveform,actual SNR,stable public counter,versioned catalogue only',null_rule='All original geometry blocks contain paid empty L13/CRC16 codewords; Es2,all1024symbols',replays='ALL/SEQ/SEL share same received waveform/header/block results',no_new_model_calls=True))
 print(json.dumps(dict(frames=len(jobs),tx_bodies=sum(len(j['layouts'])for j in jobs))))
if __name__=='__main__':main()
