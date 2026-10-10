"""Read-only metadata audit of historical N2048 reuse and finite clean states."""
import argparse,collections,json
from pathlib import Path
import t2_pilot as h
from t6_plan import write_csv


def audit(a):
    bundle=Path(a.historical_bundle);cat=h.read(a.catalogue);env=h.read(a.environment_request)
    m1=h.read(bundle/'m1_calibrate_0000.json');m1reg=h.read(bundle/'m1_calibrate_registration.json')
    m9=h.read(bundle/'m9_calibrate_source_checkpoints_0000.json');m9done=h.read(bundle/'m9_calibrate_completion.json')
    hd=h.read(bundle/'h_full1000_render_completion.json');hr=h.read(bundle/'h_full1000_rows_0000.json')
    hcfg=h.read(bundle/'h_render_config.json');hreg=h.read(bundle/'h_execution_registration.json')
    groups=collections.defaultdict(list)
    for p in cat['profiles']:groups[(p['m'],p['K'])].append(p)
    h.require(len(cat['profiles'])==521 and len(groups)==48,'Actual finite N2048 wire/state inventory changed')
    rows=[dict(m=m,K=K,wire_actions=len(v),whole=K==0,whole_prefix_existing_T1_T2_protocol_compatible=K==0 and m<=9,
        pilot_clean_memberships=3*100*len(v),pilot_clean_unique_states=100,
        interpretation='Correct-token scenario only; actual token equality and renderer/evaluator bindings still required')for(m,K),v in sorted(groups.items())]
    overlaps=sorted(set(groups)&{(x['received_m'],x['received_K'])for x in hr})
    h.require(overlaps==[(7,0),(8,0),(9,0)],'H sample/cache geometry intersection changed')
    h.require(hd['status']=='H_FULL1000_RX_COMPLETE'and hd['source_count']==1000 and hd['frozen_visual_identity']==env['old_visual_identity']
        and hd['numerical_runtime']==env['old_numerical_runtime'],'H actual complete renderer identity differs')
    module=hcfg['source_driver_module'];h.require(hreg['source_bindings'][module]==env['source_bindings'][module],'H renderer source binding differs')
    h.require(all('dinov2_vitl14_cosine'not in x and'dino_cosine'in x for x in m1['rows']+m9['rows']),'Historical M1 sample evaluator scope changed')
    checkpoint=m1reg['frozen_identity']['quality']['config']['dino_checkpoint']
    h.require(checkpoint.endswith('dinov2_vits14_pretrain.pth'),'Explicit historical DINO-S14 identity expected')
    output=Path(a.out).resolve();output.mkdir(parents=True,exist_ok=True);write_csv(output/'clean_state_shapes.csv',rows)
    count=156300;clean=len(groups)*100;whole=sum(len(v)for(m,K),v in groups.items()if K==0 and m<=9)*300
    result=dict(status='T6_METADATA_CACHE_REUSE_AUDIT_NOT_SCIENTIFIC_EXECUTION',N=2048,raw_wire_actions=521,distinct_correct_token_mK_states=48,
        pilot_logical_conditions=count,pilot_no_error_unique_source_states=clean,
        conditional_no_error_state_deduplication_fraction=1-clean/count,
        existing_whole_shape_compatible_actions=128,existing_whole_shape_compatible_pilot_memberships=whole,
        conditional_whole_membership_cache_opportunity_fraction=whole/count,
        actual_new_GPU_renders='UNKNOWN_BEFORE_NEW_ACTUAL_RX; exact post-CPU received-state plan required',
        actual_cache_hit_fraction='NOT_MEASURED_FOR_NEW_N2048_RX',
        M1_sample=dict(source_index=0,rows=len(m1['rows']),DINO_checkpoint=checkpoint,
            DINO_L_available=False,actual_token_state_available=False,calibration_RGB_archive_available=False,
            completion='No complete M1 calibration closure supplied; no completed-stage claim'),
        M9=dict(completion_status=m9done['status'],sources=m9done['sources'],sample_rows=len(m9['rows']),
            old_physical_frames=m9done['physical_frames'],DINO_L_available=False,actual_token_state_available=False,
            calibration_RGB_archive_available=False,old_scientific_frames_relabelled_as_new=0),
        H=dict(completion_status=hd['status'],sources=hd['source_count'],old_frames=hd['frame_count'],sample_rows=len(hr),
            sample_unique_RGBs=len({x['image_sha256']for x in hr}),sample_unique_actual_token_hashes=len({x['rx_summary']['actual_received_tokens_sha256']for x in hr}),
            identical_frozen_visual_identity=True,identical_numerical_runtime=True,identical_render_source=True,
            sample_mK_intersection_with_T6=overlaps,DINO_L_available=False,LPIPS_available=False,
            reuse_scope='Exact same-source received-token hash, m/K and renderer may reuse RGB only. Current metric evaluation still needed.'),
        preferred_cache_order=['original RAW90/source exact state+RGB+three metrics','completed T2 full1000 actual state+RGB+three metrics',
            'completed T1 full1000 actual independent source-decode state+RGB+three metrics','this T6 pilot when registering its full1000 recheck'],
        planned_GPU_workers=2,actual_packet_decodes=0,actual_model_calls=0,source_pixels_opened=0,
        outputs={str(output/'clean_state_shapes.csv'):h.sha(output/'clean_state_shapes.csv')})
    inputs=[Path(a.catalogue),Path(a.environment_request)]+[bundle/n for n in('m1_calibrate_0000.json','m1_calibrate_registration.json','m9_calibrate_source_checkpoints_0000.json',
        'm9_calibrate_completion.json','h_full1000_render_completion.json','h_full1000_rows_0000.json','h_render_config.json','h_execution_registration.json')]
    result['inputs']=[h.desc(p)for p in inputs];h.save(output/'completion.json',result)
    return{k:v for k,v in result.items()if k not in('inputs','outputs')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for x in('historical-bundle','catalogue','environment-request','out'):p.add_argument('--'+x,required=True)
    print(json.dumps(audit(p.parse_args()),indent=2))
