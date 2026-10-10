"""Freeze all nine N2048 policies from complete original-calibration evidence."""
import argparse,json
from pathlib import Path
import t2_pilot as h


def freeze(a):
    rd=h.read(a.raw_completion);ed=h.read(a.entropy_completion);rf=h.checked(rd['request']);ef=h.checked(ed['request']);family=h.read(a.entropy_family_selection)
    h.require(rd['status']=='T6_RAW_FULL1000_CALIBRATION_COMPLETE'and ed['status']=='T6_ENTROPY_FULL1000_CALIBRATION_COMPLETE','Both finite full1000 calibration stages must actually complete')
    h.require(rd['source_count']==ed['source_count']==1000 and rd['noise_count']==ed['noise_count']==3
        and rd['noise_seeds']==ed['noise_seeds']==[4101,4102,4103]and rd['source_ids']==ed['source_ids'],'Same original1000 sources and3 noises')
    selected=h.desc(a.entropy_family_selection)
    h.require(rf['entropy_family_selection']==ef['entropy_family_selection']==selected and ef['family']==family['family']
        and family['status']=='T1_SINGLE_ENTROPY_FAMILY_FROZEN_CALIBRATION_ONLY'and not family['holdout_used_for_selection']and not family['holdout_files_read'],'One common N1024 calibration-frozen entropy family')
    h.require(not any(x['holdout_used_for_selection']or x['confirmation_images_opened']for x in(rf,ef)),'No confirmation-conditioned policy selection')
    rp=h.checked(rd['policy']);ep=h.checked(ed['policy'])
    h.require(rd['outputs'][rd['policy']['path']]==rd['policy']['sha256']and ed['outputs'][ed['policy']['path']]==ed['policy']['sha256'],'Policies sealed by actual completion')
    policies={k:{}for k in('RAW_WHOLE','RAW_PARTIAL','ENTROPY_WHOLE')};means={k:{}for k in policies}
    for winner in rp['winners']:
        snr=winner['snr_db'];fam=winner['family'];h.require(fam in('RAW_WHOLE','RAW_PARTIAL')and snr in[4,10,19],'Expected raw winner family/SNR')
        p=next(x['profile']for x in rf['schedule']if x['snr_db']==snr and x['candidate_id']==winner['candidate_id'])
        h.require(fam!='RAW_WHOLE'or p['K']==0,'WHOLE winner must have K0');h.require(str(snr)not in policies[fam],'Duplicate frozen policy')
        policies[fam][str(snr)]=p;means[fam][str(snr)]=winner['dinov2_vitl14_cosine']
    for winner in ep['winners']:
        snr=winner['snr_db'];c=next(x for x in ef['candidates']if x['candidate_id']==winner['candidate_id'])
        h.require(str(snr)not in policies['ENTROPY_WHOLE'],'Duplicate entropy policy');policies['ENTROPY_WHOLE'][str(snr)]=dict(c,family=ef['family'])
        means['ENTROPY_WHOLE'][str(snr)]=winner['dinov2_vitl14_cosine']
    h.require(all(set(p)=={'4','10','19'}for p in policies.values()),'All nine actual frozen winners required')
    value=dict(status='T6_N2048_POLICIES_FROZEN_CALIBRATION_ONLY_V1',N=2048,source_count=1000,noise_count=3,
        calibration_source_ids=rd['source_ids'],calibration_noise_seeds=[4101,4102,4103],policies=policies,full_calibration_DINO_L_means=means,
        entropy_family_selection=selected,raw_calibration_completion=h.desc(a.raw_completion),entropy_calibration_completion=h.desc(a.entropy_completion),
        raw_phy_qualification=rf['phy_qualification'],entropy_phy_qualification=ef['phy_qualification'],
        criterion='source-mean DINOv2-L within registered finite full-calibration unions, lexical candidate tie break',
        selection_used_confirmation=False,holdout_used_for_selection=False,confirmation_files_read=False,
        interpretation='Finite calibrated candidates only; no claim of globally optimal N2048 policy')
    h.save(a.out,value);return dict(status=value['status'],freeze=h.desc(a.out),policy_count=9,new_scientific_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for x in('raw-completion','entropy-completion','entropy-family-selection','out'):p.add_argument('--'+x,required=True)
    print(json.dumps(freeze(p.parse_args())))
