"""Freeze every predeclared MCS result including gray receptions, once."""
from g import *
import csv

def main():
 wait=read(L/'cal_dino_control/actual_wait.json');assert wait['returncode']==0 and gone(wait['process'])and gone(read(L/'cal_dino_control/started.json')['process'])
 done=read(L/'cal_dino/completion.json');assert done['ledger']['unresolved']==0
 scores=cr(done['scores']);frames=read(D/'cal_frame_pair_map.json');assert len(frames)==2000
 rows=[];methods={}
 for method in ['NEST','WHOLE','ADA','ARPC']:
  candidates=[]
  for q,rate in MCS:
   cid=f'q{q}_r{rate.replace("/","-")}';a=[x for x in frames if x['method']==method and x['candidate_id']==cid];assert len(a)==len({v['source_id']for v in a})==100
   vals=[scores[x['pair_key']]['value']for x in a]
   row=dict(method=method,q=q,rate=rate,candidate_id=cid,sources=100,channel_repetitions=1,dinov2_cosine=float(np.mean(vals)),physical_rejections=sum(not x['received']for x in a),mean_source_utilization=float(np.mean([x['source_bits']/x['capacity_bits']for x in a])))
   rows.append(row);candidates.append(row)
  best=max(candidates,key=lambda x:x['dinov2_cosine']);methods[method]=best
 methods['BPG']=read(D/'protocol.json')['BPG']
 with(D/'calibration_summary.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(D/'frozen_policies.json',dict(status='FROZEN_BEFORE_DEVELOPMENT',protocol=pin(D/'protocol.json'),score_identity=pin(L/'cal_dino/identity.json'),scores=done['scores'],summary=pin(D/'calibration_summary.csv'),methods=methods,selection='maximum equal-source DINOv2 cosine; ties declaration order; BPG reuses historical PSNR-calibrated profile',new_test_results_seen=0,frozen_unix=time.time()))
 print(json.dumps(methods))
if __name__=='__main__':main()
