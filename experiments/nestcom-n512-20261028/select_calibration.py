"""Prespecified DINO selection; pilot one realization, full three per source."""
from n512_common import *
from pipeline_prepare import closed
import csv,collections
def csvsave(path,rows):
 with Path(path).open('x',newline='',encoding='utf8')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def aggregate(logical,physical,scores,expected_sources,seeds):
 grouped=collections.defaultdict(lambda:collections.defaultdict(dict))
 for f in logical:
  key=(f['method'],int(f['snr_db']),f['candidate_id']);s=f['source_id'];noise=int(f['noise_seed']);assert noise not in grouped[key][s]
  p=physical[f['physical_key']];v=scores[p['pair_key']]['value'];assert np.isfinite(v);grouped[key][s][noise]=float(v)
 rows=[]
 for (method,snr,candidate),values in sorted(grouped.items()):
  assert set(values)==set(expected_sources)
  assert all(set(x)==set(seeds)for x in values.values())
  means=[np.mean([values[s][n]for n in seeds],dtype=np.float64)for s in expected_sources]
  rows.append(dict(method=method,snr_db=snr,candidate_id=candidate,source_count=len(means),logical_count=len(means)*len(seeds),mean_dino_cosine=float(np.mean(means,dtype=np.float64))))
 return rows
def main(stage):
 assert stage in ('pilot','full');closed(stage+'_dino');protocol=read(O/'protocol.json');pid=sha(O/'protocol.json');plan=read(L/(stage+'_plan')/'completion.json');pairs=read(L/(stage+'_metric_plan')/'completion.json');scores=read(L/(stage+'_dino')/'scores.json')
 pop=cr(protocol['sources']['cal1000']);pop=[s for s in pop if stage=='full'or s['source_id']in set(protocol['pilot_source_ids'])];ids=[s['source_id']for s in sorted(pop,key=lambda x:x['source_index'])]
 logical=cr(plan['logical']);rows=aggregate(logical,cr(pairs['physical']),scores,ids,protocol['pilot_noise'if stage=='pilot'else'full_noise']);lookup={c['candidate_id']:c for c in protocol['candidates']};out=O/(stage+'_calibration');out.mkdir(exist_ok=False);csvsave(out/'candidate_summary.csv',rows)
 selected={};details=[]
 for snr in (4,10,19):
  selected[str(snr)]=[]if stage=='pilot'else{}
  for method in ('NEST','WHOLE','ADA'):
   pool=sorted((x for x in rows if x['method']==method and x['snr_db']==snr),key=lambda x:(-x['mean_dino_cosine'],x['candidate_id']));assert pool
   if stage=='pilot':
    top=[x['candidate_id']for x in pool[:3]];target=[256,0]if method=='ADA'else[10,0]
    anchor=next(c['candidate_id']for c in protocol['candidates']if c['method']==method and c['q']==2 and c['rate']=='1/2'and c['target']==target)
    names=list(dict.fromkeys(top+[anchor]));selected[str(snr)].extend(lookup[n]for n in names);details.append(dict(method=method,snr_db=snr,top3=top,robust_anchor=anchor,advanced=names))
   else:selected[str(snr)][method]=lookup[pool[0]['candidate_id']];details.append(dict(method=method,snr_db=snr,selected=pool[0]))
 result=dict(status='PILOT_FINALISTS_FROZEN'if stage=='pilot'else'N512_POLICIES_FROZEN_BEFORE_COMMON500',protocol_sha256=pid,candidate_summary=pin(out/'candidate_summary.csv'),selection=details,source_count=len(ids),noise_seeds=protocol['pilot_noise'if stage=='pilot'else'full_noise'],metric='DINOv2-L cosine',common500_used=False,new_bootstrap=False,actual_wait=closed(stage+'_dino'))
 result['candidates_by_snr'if stage=='pilot'else'policies']=selected;save(O/('pilot_finalists.json'if stage=='pilot'else'frozen_policies.json'),result);print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
