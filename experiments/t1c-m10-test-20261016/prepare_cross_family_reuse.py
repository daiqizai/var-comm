"""Resolve exact already-completed physical events shared across frozen families."""
from tc_common import *
from tc_protocol import validate_grid
import collections
def main():
 assert not(O/'link_registration.json').exists()
 prior=read(T/'link_preparation_v2/completion.json');rows=checkread(prior['frames']);available={}
 for row in rows:
  if row['reuse']:
   k=row['event_key'];old=row['legacy']
   if k in available:
    assert available[k]['image']['image_sha256']==old['image']['image_sha256'] and available[k]['metrics']==old['metrics']
   available[k]=old
 counts=collections.Counter()
 for row in rows:
  row['original_logical_pair_unchanged']=row['reuse'];other=available.get(row['event_key']);row['execution_reference']=other
  if other is not None:
   row['reuse']=True
   if not row['original_logical_pair_unchanged']:row['reuse_reason']='EXACT_OTHER_FAMILY_CLOSED_EVENT_AND_FIXED_OUTPUT'
  counts[row['population']+'_'+row['strategy_family']+'_'+row['reuse_reason']]+=1
 grid=validate_grid(rows);out=T/'link_preparation_v3';out.mkdir();save(out/'frames.json',rows)
 summary=dict(prior,status='T1C_CROSS_FAMILY_EXACT_REUSE_PREPARED_NO_NEW_PHY',prior_preparation=pin(T/'link_preparation_v2/completion.json'),frames=pin(out/'frames.json'),logical_reused=grid['reused'],logical_changed=grid['changed'],unique_new_physical=grid['new_physical'],packet_cap=grid['packet_cap'],counts=dict(counts),original_logical_pairs_unchanged=sum(x['original_logical_pair_unchanged'] for x in rows),cross_family_complete_reuse=sum(x['reuse'] and not x['original_logical_pair_unchanged'] for x in rows))
 save(out/'completion.json',summary);print(json.dumps(dict(grid=grid,counts=dict(counts),cross_family_reuse=summary['cross_family_complete_reuse']),indent=2))
if __name__=='__main__':main()
