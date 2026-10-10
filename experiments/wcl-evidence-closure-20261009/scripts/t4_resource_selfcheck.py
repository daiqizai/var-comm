"""Engineering arithmetic/absence checks only; no scientific execution."""
import math
import json
import t4_summarize as s

def row(i,seed,state,E):
    return dict(method='ENGINEERING_ONLY',snr_db=10,source_id='fixture'+str(i),source_index=i,noise_seed=seed,N=1024,
        failure_state=state,E_frame=E,rho=None if E is None else E/2048,psnr_db=float(i+seed),lpips_alex=.2,
        dinov2_vitl14_cosine=.4,convnext_top1_source_prediction=i==0)
def main():
    checks=[]
    rows=[row(0,1,'DECODED',2000),row(0,2,'DECODED',2100),row(1,1,'SOURCE_UNFIT',None),row(1,2,'GRAY',2050)]
    e=s.energy_summary(rows)[0]
    assert e['actual_energy_frame_count']==3 and e['missing_energy_frame_count']==e['source_unfit_frame_count']==1
    assert e['E_mean']==2050 and math.isclose(e['E_std'],math.sqrt(5000/3)) and e['E_p50']==2050 and e['E_min']==2000 and e['E_max']==2100
    assert e['rho_mean']==2050/2048
    bad=[dict(rows[2],E_frame=0,rho=0)]
    try:s.energy_summary(bad)
    except RuntimeError:pass
    else:raise AssertionError('SOURCE_UNFIT fabricated energy accepted')
    checks.append('SOURCE_UNFIT energy remains absent/countable; ddof0 and E/(2N) from observed transmitted frames')
    quality,counts=s.state_quality(rows,['DECODED','SOURCE_UNFIT','GRAY','EMPTY'])
    empty=[r for r in quality if r['failure_state']=='EMPTY'];assert all(r['metric_frame_count']==0 and r['frame_weighted_mean'] is None and r['status']=='EMPTY_CONDITION' for r in empty)
    assert sum(r['frames'] for r in counts if r['failure_state']!='ALL_FRAMES')==4
    assert all(r['frame_count']==4 for r in quality if r['failure_state']=='ALL_FRAMES')
    checks.append('No failed source/noise omitted; explicit zero-count states never become zero quality')
    timing=[]
    for i in range(16):
        for j in range(3):
            timing.append(dict(method='ENGINEERING_ONLY',snr_db=4,phase='measured',source_id=str(i),repetition=j,
                gray=False,header_accepted=True,body_crc_accepted=j!=0,TX_seconds=.1,RX_seconds=.2,software_e2e_excluding_channel_seconds=.3))
    table=s.timing_summary(timing)
    assert all(r['sample_count']==16 for r in table if r['condition']=='BODY_CRC_REJECT')
    assert all(r['sample_count']==48 for r in table if r['condition']=='NON_GRAY_OUTPUT')
    assert all(r['sample_count']==0 and r['mean'] is None for r in table if r['condition']=='GRAY_OUTPUT')
    checks.append('CRC rejection does not imply gray; measured-only16x3 coverage and conditional counts are explicit')
    print(json.dumps(dict(status='PASS_ENGINEERING_ONLY',checks=checks,new_PHY=0,new_model_calls=0,new_bootstrap=0)))
if __name__=='__main__':main()
