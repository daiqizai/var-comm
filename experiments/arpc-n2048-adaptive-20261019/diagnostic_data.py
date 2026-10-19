"""Tabulate completed cal100 source/quality results; no new inference/statistics."""
from shared import *
from adaptive_plan import NEST_ENDPOINTS
import csv,collections

def table(path,rows):
 with path.open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 for stage in ('cal100_clean_native','cal100_dino','cal100_prescreen'):
  c=L/'controls'/stage;w=read(c/'actual_wait.json');launch=read(c/'launch.json')
  assert w['returncode']==0 and gone(w['process']) and gone(launch['parent_process'])
 out=O/'calibration_diagnostics';out.mkdir()
 new={(x['source_id'],x['image_float_sha256']):x['dino']for x in read(L/'cal100_scores_v1/worker/scores.json')}
 objects={(x['source_id'],x['codec'],x['endpoint']):x for x in read(L/'b0/source_catalogue.json') if x['cal100']}
 rows=[]
 for x in read(L/'cal100_scores_v1/all_states.json'):
  s=objects[x['source_id'],x['codec'],x['endpoint']];m,k=NEST_ENDPOINTS[x['endpoint']]if x['codec']==0 else (x['endpoint'],0)
  rows.append(dict(source_id=x['source_id'],codec=x['codec'],endpoint=x['endpoint'],depth_m=m,partial_K=k,source_bits=s['bit_length'],charged_source_bits=8*((s['bit_length']+7)//8),dino=x['dino']if x['dino']is not None else new[x['source_id'],x['image_float_sha256']],image_sha256=x['image']['sha256'],image_float_sha256=x['image_float_sha256'],origin=x['origin']))
 assert len(rows)==3200;table(out/'source_quality.csv',rows)
 groups=collections.defaultdict(list)
 for x in rows:groups[x['codec'],x['endpoint']].append(x)
 means=[]
 for (codec,endpoint),rr in sorted(groups.items()):
  assert len(rr)==100
  means.append(dict(codec=codec,endpoint=endpoint,depth_m=rr[0]['depth_m'],partial_K=rr[0]['partial_K'],sources=100,source_bits_mean=sum(x['source_bits']for x in rr)/100,charged_source_bits_mean=sum(x['charged_source_bits']for x in rr)/100,source_bits_min=min(x['source_bits']for x in rr),source_bits_max=max(x['source_bits']for x in rr),dino_mean=sum(x['dino']for x in rr)/100))
 table(out/'endpoint_means.csv',means)
 with (O/'cal100_prescreen_v1/actual_fallback_quality.csv').open()as f:actual=list(csv.DictReader(f))
 groups=collections.defaultdict(list)
 for x in actual:groups[int(x['codec']),x['policy_id']].append(x)
 totals=[];distributions=[]
 for (codec,policy),rr in sorted(groups.items()):
  assert len(rr)==100
  endings=collections.Counter(x['actual_endpoint']or'NULL'for x in rr)
  for end,n in sorted(endings.items()):distributions.append(dict(codec=codec,policy_id=policy,actual_endpoint=end,sources=n))
  totals.append(dict(codec=codec,policy_id=policy,source_fit_count=sum(x['source_no_fit']=='False'for x in rr),sources=100,fit_rate=sum(x['source_no_fit']=='False'for x in rr)/100,capacity_bits=int(rr[0]['capacity_bits']),mean_charged_source_bits=sum(8*((int(x['object_bits'])+7)//8)for x in rr)/100,mean_occupancy=sum(8*((int(x['object_bits'])+7)//8)/int(x['capacity_bits'])for x in rr)/100))
 table(out/'policy_capacity_utilization.csv',totals);table(out/'actual_endpoint_distributions.csv',distributions)
 put(out/'manifest.json',dict(status='COMPLETE_CAL100_DIAGNOSTIC_TABLES',source=pin(__file__),inputs=[pin(L/'cal100_scores_v1/all_states.json'),pin(L/'cal100_scores_v1/worker/scores.json'),pin(L/'b0/source_catalogue.json'),pin(O/'cal100_prescreen_v1/actual_fallback_quality.csv')],outputs=[pin(out/n)for n in ('source_quality.csv','endpoint_means.csv','policy_capacity_utilization.csv','actual_endpoint_distributions.csv')],new_scientific_calls=0,new_bootstrap=0,not_wireless_test=True))
 print(json.dumps(dict(rows=len(rows),endpoint_means=len(means),policies=len(totals))))

if __name__=='__main__':main()
