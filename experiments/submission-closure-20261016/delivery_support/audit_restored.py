"""Audit exact restored old source/receiver evidence; no learned calls."""
from pathlib import Path
import json,hashlib,numpy as np
H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def tsh(a):return hashlib.sha256(f'int64:{len(a)}\0'.encode()+a.astype('<i8').tobytes()).hexdigest()
def psh(a):return hashlib.sha256(b'uint8_bits\0'+a.astype(np.uint8).tobytes()).hexdigest()
def save(p,d):
 with Path(p).open('x',encoding='utf8')as f:json.dump(d,f,indent=2,allow_nan=False)
b=H/'restored_m10/outputs/WCL-EVIDENCE-CLOSURE-20261009'
roots=[b/'T6_calibration_source100_v1',b/'T6_calibration_source1000_v1']
closures=[]
for p in roots:
 c=read(p/'completion.json');w=read(p/'parent_observer/exit.json');r=read(p/'request.json')
 assert w['actual_child_waited']is True and type(w['exit_code'])is int and w['exit_code']==0
 assert c['request_sha256']==sha(p/'request.json') and c['new_packet_decodes']==c['new_image_renders']==c['new_metric_calls']==0
 for old,digest in c['outputs'].items():
  q=b/Path(old).relative_to('/home/liulu/projects/VAR_COMM/outputs/WCL-EVIDENCE-CLOSURE-20261009');assert sha(q)==digest,(q,'closure SHA')
 closures.append({'completion_sha256':sha(p/'completion.json'),'wait_sha256':sha(p/'parent_observer/exit.json'),'request_sha256':sha(p/'request.json'),'counts':c['counts'],'seconds':c['seconds']})
rows=[]
for i in range(1000):
 p=roots[i>=100];cpfile=p/'source_checkpoints'/f'{i:04d}.json';cp=read(cpfile);arc=p/'sources'/f'{i:04d}.npz'
 assert sha(arc)==cp['archive']['sha256'] and cp['source_index']==i and cp['source_role']=='calibration'
 with np.load(arc,allow_pickle=False)as z:
  tokens=z['tokens'];bits=z['var_m10_bits'];rx=z['var_m10_received_tokens'];s=cp['streams']['EC_VAR_WHOLE']['10']
  assert tokens.shape==rx.shape==(680,) and np.array_equal(tokens,rx) and cp['m10_independent_roundtrip']is True
  assert len(bits)==s['payload_bits'] and psh(bits)==s['payload_sha256'] and np.isin(bits,[0,1]).all()
  assert tsh(tokens)==cp['source_tokens_sha256']
  proof=s['receiver_cache_proof'];assert proof['canonical'] and proof['independent_roundtrip']
  for role in ['m10_TX','m10_RX']:
   pin=proof['upstream'][role];f=p/Path(pin['path']).relative_to(read(p/'request.json')['out']);assert sha(f)==pin['sha256']
  old={str(m):{'bits':len(z[f'var_m{m}_bits']),'sha256':ash(z[f'var_m{m}_bits'])}for m in range(4,10)if f'var_m{m}_bits'in z}
 rows.append(dict(source_index=i,source_id=cp['source_id'],preprocessing_id=cp['preprocessing_id'],tokens_sha256=cp['source_tokens_sha256'],length=len(bits),m10_bits_sha256=s['payload_sha256'],m10_rx_tokens_sha256=tsh(rx),old_whole=old,checkpoint_sha256=sha(cpfile),archive_sha256=sha(arc),source_model_id=proof['source_model_id'],historical_independent_roundtrip=True))
save(H/'historical_m10_audit.json',dict(schema='RESTORED_M10_CAL1000_AUDIT_V1',closures=closures,rows=rows,new_model_calls=0))
print('M10_AUDIT',len(rows))
