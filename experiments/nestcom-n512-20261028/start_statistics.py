"""One statistics registration, after both frozen metric roles actually close."""
from n512_common import *
from dispatch import launch
from pipeline_prepare import closed
from stage_operations_v4 import budget

def main():
 for role in ('formal_three_metrics','formal_clip_metrics'):closed(role)
 protocol=read(O/'protocol.json');bank=protocol['bootstrap']['indices'];assert isinstance(bank,dict)
 a=np.load(checked(bank),allow_pickle=False);assert a.shape==(10000,500)and asha(a)==protocol['bootstrap']['array_sha256']
 old=R/'results/adatoken_response_20261026_v1/c_predclass_formal_delivery_v1/common500/paired_source_values.csv'
 inputs=[pin(O/p)for p in ['protocol.json','frozen_policies.json','formal_post_output_audit.json']]+[pin(L/p)for p in ['formal_plan/completion.json','formal_metric_plan/completion.json','formal_three_metrics/completion.json','formal_three_metrics/scores.json','formal_clip_metrics/completion.json','formal_clip_metrics/scores.json']]+[pin(old)]
 seconds,spent=budget(False,1);role='formal_statistics'
 r=dict(bootstrap=bank,bootstrap_array_sha256=asha(a),N1024_paired_sources=pin(old),inputs=inputs,code=[pin(E/p)for p in ['n512_common.py','n512_phy.py','pipeline_prepare.py','select_calibration.py','formal_statistics.py','start_statistics.py']],max_seconds=min(seconds,1800),CPU_process_seconds_already_spent=spent,mean_CI=36,paired_CI=24,budget_interaction_CI=12,new_random_draws=0)
 rp=O/'statistics_registration.json';save(rp,r);print(launch(role,rp,E/'formal_statistics.py'),flush=True)

if __name__=='__main__':main()
