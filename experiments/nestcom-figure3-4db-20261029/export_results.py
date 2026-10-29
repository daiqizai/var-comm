"""CPU-only publication export from completed, hash-checked native outputs."""
from pathlib import Path
import sys,csv,json,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adatoken-response-20261026'))
from common import *
import numpy as np
from PIL import Image
L=B/'nestcom_fig3_4db_20261029_v1';D=R/'results/nestcom_fig3_4db_20261029_v1'

def main():
    out=D/'delivery';out.mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'receipts').mkdir(exist_ok=True)
    waits=[]
    for role in ['whole_source_v2','whole_phy_v2','whole_rx_v2','ada_phy','ada_rx','scores']:
        p=L/(role+'_control/actual_wait.json');w=read(p)
        assert w['returncode']==0 and gone(w['process'])and gone(w['parent_process'])
        waits.append(dict(role=role,wait=pin(p),worker_gone=True,owner_gone=True))
        shutil.copyfile(p,out/'receipts'/(role+'_actual_wait.json'))
    old=read(L/'recovery_supervisor/actual_wait.json');assert old['returncode']==0 and gone(old['child'])
    assert read(L/'recovery_owner/completion.json')['exact_historical_hash_matches']==10
    jobs=read(D/'metric_jobs.json');scores=list(csv.DictReader((L/'scores/scores.csv').open()));assert len(scores)==14
    cfg=read(D/'timing_config.json');bpg_policy=cfg['bpg_policies']
    profiles=[];mapping=[]
    for j in jobs:
        i=j['source_index'];method=j['method'];row=next(v for v in scores if int(v['source_index'])==i and v['method']==method)
        image=np.load(checked(j['image']),allow_pickle=False)['image'];assert asha(image)==row['pixel_sha256']
        display=np.rint(np.clip(image,0,1)*255).astype(np.uint8).transpose(1,2,0)
        png=out/'images'/('%04d_%s.png'%(i,method));Image.fromarray(display).save(png)
        profile={};provenance=j['provenance']
        if method in ['WHOLE','ADA_PREDCLASS_FIXEDFRAME']:
            phy='whole_import_revision_v2/whole_phy' if method=='WHOLE'else'ada_phy'
            frame=next(v for v in read(L/phy/'frames.json')if v['event_id']==j['frame_event_id'])
            profile=frame['actual_RX'].get('rx_profile',frame['transmission'])
            profiles.append(frame)
        elif method in ['NEST','ARPC']:
            profile=cr(j['received_input'])['profile']
        elif method=='BPG':profile=dict(frozen_policy=bpg_policy)
        else:profile=dict(channel='continuous JSCC',endpoint='not applicable',MCS='not applicable')
        row.update(timing_case=('s4_i%04d_r3'%i)if method not in ['WHOLE','ADA_PREDCLASS_FIXEDFRAME']else'',repeat='first registered visualization event'if method in ['WHOLE','ADA_PREDCLASS_FIXEDFRAME']else'0 (timing r3)',noise_namespace=j.get('noise_namespace','historical method-specific five_method_timing_20261020_v1'),display_image='images/'+png.name,display_png_sha256=sha(png),preprocessing_id=j['preprocessing_id'])
        mapping.append(dict(source_index=i,source_id=j['source_id'],method=method,N=1024,snr_db=4,source=j['source'],native_image=j['image'],native_array_sha256=j['pixel_sha256'],display=pin(png),profile=profile,receive_status=j['receive_status'],provenance=provenance,received_input=j.get('received_input'),new_frame=method in ['WHOLE','ADA_PREDCLASS_FIXEDFRAME']))
    for s in read(D/'verified_sources.json'):
        pixels=np.load(checked(s['archive']),allow_pickle=False)['pixels'];assert pixels.dtype==np.uint8
        Image.fromarray(pixels.transpose(1,2,0)).save(out/'images'/('%04d_original.png'%s['source_index']))
    csvsave(out/'figure3_metrics.csv',scores);save(out/'source_receive_reconstruction_map.json',mapping)
    save(out/'new_frame_events.json',profiles)
    completion=read(L/'scores/completion.json');assert completion['ledger']['unresolved']==0
    save(out/'actual_compute_and_reuse.json',dict(status='COMPLETE',historical_reconstructions_replayed=10,exact_historical_array_hash_matches=10,new_WHOLE_source_probability_passes=2,new_WHOLE_prior_scale_calls=14,new_tokenizer_calls=0,new_classifier_calls=0,new_PHY_frames=4,new_packet_decodes=8,new_WHOLE_RX=2,new_Ada_RX=2,total_native_reconstructions=14,new_lpips_pairs=14,new_dreamsim_pairs=14,new_training=0,new_calibration=0,new_bootstrap=0,failed_pre_model_WHOLE_import_calls=0,failed_pre_registration_metric_manifest_model_calls=0,all_denominators_retained=True,noise_vectors_not_claimed_identical=True,stage_waits=waits,metric_ledger=completion['ledger']))
    for name,p in [('historical_recovery',L/'recovery_owner/completion.json'),('metric_identity',L/'scores/metric_identity.json'),('scores',L/'scores/completion.json'),('original_import_failure',L/'whole_source/failure.json'),('original_import_failure_wait',L/'whole_source_control/actual_wait.json')]:
        shutil.copyfile(p,out/'receipts'/(name+'.json'))
    save(out/'export_identity.json',dict(native_arrays_only=True,display_transform='clip [0,1]; round to nearest uint8; RGB CHW to HWC',source_indices=[95,21],method_count=7,rows=14,all_actual_processes_gone=True,metric_scores=pin(L/'scores/scores.csv'),export_script=pin(__file__)))
    print(json.dumps(dict(status='EXPORTED',rows=14,images=16,metrics=pin(out/'figure3_metrics.csv'))))

if __name__=='__main__':main()
