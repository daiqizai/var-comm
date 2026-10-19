"""Read original token/CDF witnesses only after actual independent RX exits."""
from shared import *
import numpy as np

def main(expected):
 r=registration(expected)
 for hostfile in ['nest_process_closure.json','arpc_process_closure.json']:assert read(O/'formal'/hostfile)['status']=='GONE'
 nc=read(L/'formal/nest/rx/completion.json');ac=read(L/'formal/arpc_resource_v1/rx/completion.json')
 nd={x['source_input_key']:x for x in nc['outputs']};ad={x['key']:x for x in ac['outputs']}
 jobs={x['key']:x for x in r['formal_jobs']};rows=[];seen=set()
 for f in read(O/'formal/logical_frames.json'):
  if not f['accepted']:continue
  key=(f['codec'],f['source_input_key'])
  if key in seen:continue
  seen.add(key);j=jobs[f['key']];a=j['asset'];source=verify(a['source_archive'])
  if key[0]==0:
   x=nd[key[1]];assert x['status']=='SOURCE_DECODED' and not x['source_truth_used'] and not x['TX_CDF_used']
   with np.load(source,allow_pickle=False)as z:truth=z['tokens'].copy()
   same=np.array_equal(truth,np.asarray(x['received_tokens']));assert same,'NeST token/CDF mismatch; do not retry'
   row=dict(codec=0,source_id=j['source_id'],key=key[1],tokens_exact=True,image=x['image'],source_archive=pin(source))
  else:
   x=ad[key[1]];assert x.get('status')!='SOURCE_PARSE_FAIL'
   prefix=Path(a['wire']['path']).stem.removesuffix('_e7');root=Path(a['wire']['path']).parent
   witness=root/(prefix+'_source.json');truthfile=root/(prefix+'_truth.npz');w=read(witness)
   assert w['source_id']==j['source_id'] and w['caption_bytes_hex']==x['caption_bytes_hex'] and w['cdf_hashes']==x['cdf_hashes'],'ARPC CDF/caption mismatch; do not retry'
   assert next(q['payload_sha256']for q in w['lengths']if q['endpoint']==7)==a['wire']['sha256']
   decoded=Path(x['path']).with_name(key[1]+'_decoded.npz')
   with np.load(truthfile,allow_pickle=False)as t,np.load(decoded,allow_pickle=False)as got:
    assert t.files==got.files and all(np.array_equal(t[k],got[k])for k in t.files),'ARPC decoded tokens mismatch; do not retry'
   row=dict(codec=1,source_id=j['source_id'],key=key[1],tokens_exact=True,cdfs_exact=True,image=pin(x['path']),witness=pin(witness),source_truth=pin(truthfile),received_tokens=pin(decoded))
  rows.append(row)
 assert len(rows)==len(nd)+len(ad)==200
 put(O/'formal/native_audit.json',dict(status='PASS_200_ACTUAL_FULL_REPRESENTATIONS',outputs=rows,truth_read_only_after_actual_RX_closed=True,source_truth_used_by_RX=False,scientific_calls_in_audit=0,model_source_calls=200))
 print(json.dumps(dict(status='PASS_200_ACTUAL_FULL_REPRESENTATIONS',count=200)))
if __name__=='__main__':main(sys.argv[1])
