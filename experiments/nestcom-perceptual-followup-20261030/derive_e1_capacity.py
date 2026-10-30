"""Descriptive frame accounting from immutable E1 records; no new scores/CI."""
from followup_common import *
from collections import Counter

def main():
 closed('E1_statistics');frames=read(L/'E1_join/frames.json');out=[];endpoints=[]
 core=['WHOLE_final_receiver','original_NeST','final_NeST','Ada_frozen']
 for method,N,snr in sorted(set((v['method'],v['N'],v['snr_db'])for v in frames if v['method']in core)):
  vs=[v for v in frames if (v['method'],v['N'],v['snr_db'])==(method,N,snr)];assert len(vs)==1500 and len(set(v['source_id']for v in vs))==500
  entry=dict(method=method,N=N,snr_db=snr,n_sources=500,n_frames=1500)
  for field in ['source_bits','capacity_bits','utilization','length_checks']:
   vals=[float(v[field])for v in vs if v.get(field)not in [None,'']]
   entry[field+'_n']=len(vals);entry[field+'_mean']=float(np.mean(vals))if vals else None
  entry['target']=json.dumps(vs[0].get('target'));entry['q']=vs[0].get('q');entry['rate']=vs[0].get('rate')
  entry['receiver_status_counts']=json.dumps(Counter(str(v['receiver_status'])for v in vs),sort_keys=True)
  out.append(entry)
  for endpoint,count in sorted(Counter(json.dumps(v.get('endpoint'),sort_keys=True)for v in vs).items()):endpoints.append(dict(method=method,N=N,snr_db=snr,endpoint=endpoint,logical=count))
 csvout(D/'e1/capacity_summary.csv',out);csvout(D/'e1/endpoint_counts.csv',endpoints)
 save(D/'e1/capacity_identity.json',dict(input=pin(L/'E1_join/frames.json'),new_scientific_calls=0,meaning='Descriptive means over original 1500 logical frames, equal repeats per source; source-bit utilization, not air-interface efficiency. No new bootstrap or rescoring. Missing fields remain null.'))
 print(json.dumps([v for v in out if v['method']=='final_NeST'],indent=2))
if __name__=='__main__':main()
