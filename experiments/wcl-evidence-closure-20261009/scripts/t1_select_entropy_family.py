"""Freeze one entropy family for both T5 and T6 using calibration only."""
import argparse,math
from t1_entropy_core import read,require,write
from t1_holdout_metadata import FAMILIES,SNRS,pin

def run(freeze,out):
    policy=read(freeze)
    require(policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and not policy['holdout_used_for_selection']
        and policy['families']==FAMILIES and policy['snrs']==SNRS and policy['source_count']==1000,
        'Actual full calibration policy freeze required')
    means={f:math.fsum(policy['policies'][f][str(s)]['calibration_DINOv2_L'] for s in SNRS)/3 for f in FAMILIES}
    require(all(math.isfinite(v) for v in means.values()),'Finite calibration criteria required')
    family=sorted(FAMILIES,key=lambda f:(-means[f],f))[0]
    result=dict(status='T1_SINGLE_ENTROPY_FAMILY_FROZEN_CALIBRATION_ONLY',family=family,N=1024,snrs=SNRS,
        original_policy_freeze=pin(freeze),selection_metric='equal_weight_mean_of_three_fullcal_frozen_champion_DINOv2_L_means',
        criterion_by_family=means,tie_break='family_lexical',policies=policy['policies'][family],
        uses=['T5_fixed16_examples','T6_additional_budget'],holdout_used_for_selection=False,
        source_count=1000,noise_count=3,holdout_files_read=False,new_model_calls=0,new_packet_decodes=0)
    write(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(run(a.freeze,a.out)['family'])
