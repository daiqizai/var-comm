"""Bind restored old common500 mismatch inputs; metadata preparation only.

No NumPy, model, channel, decoder, metric, bootstrap, SSH or process execution.
Restored receipts are evidence candidates, never automatic historical admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import mismatch_plan as frozen

SCHEMA='H800_RAW100_MISMATCH_INPUT_PREPARATION_V1'
RESTORE_SHA='6cea0f60a55af9d6129db3ed9e3cf662ffa56f4e61b8f445f28113d59ecdd896'
CPU_COMPLETE_SHA='b424925cfb4d6ea2295f2e3658cc870196e1a20419914a5b132d0181a32ba682'
CPU_CONFIG_SHA='01de61630f65d78a304f3ec50f5cee8e27eff9e0a8a851a146d6048ce8d26902'
POLICY_SHA='7871ac7f9c619bd21c63ba31157738cd56c0dda9344c993e99b935f60ce70c0c'
OLD='/home/liulu/projects/VAR_COMM/outputs/MAIN-RAW64-20261007/'
require=frozen.require


def requirements(config):
    names=('adapter_module','receiver_module','original_receiver_module','aliases','profiles','final_freeze',
           'selected','original_ledger_module','science_registration')
    paths={config[k] for k in names}|set(config['adapter_config'].values())
    result=[]
    for path in sorted(paths):
        digest=config['source_bindings'].get(path,config['input_bindings'].get(path))
        require(isinstance(digest,str) and len(digest)==64,'Missing original code/input binding: '+path)
        result.append(dict(original_path=path,sha256=digest))
    return result


def future_caps():
    return dict(logical_frames=5400,physical_frames=4500,actual_new_packet_decodes=9000,
        visual=dict(model_load=1,actual_model_objects=3,encoder=0,source_tx_VAR=0,source_rx_VAR=0,
                    var_render=4500,prior_scale=45000,decoder_forward=4500),
        metrics=dict(model_constructions=3,reference_preparations=100,image_scores=4500,
            dinov2_vitl14_reference=100,dinov2_vitl14_reconstruction=4500,
            convnext_reference=100,convnext_reconstruction=4500,lpips_pair=4500,lpips_alexnet_backbone_forward=9000),
        historical_PHY_reuse_admitted=0,historical_image_reuse_admitted=0,historical_metric_reuse_admitted=0,
        qualification_calls=0,qualification_requires_separate_bound_registration=True)


def build(original,manifest,config,cpu,policies,restore,inventory):
    sources,schedule=frozen.validate_inputs(original,manifest);base=frozen.make_plan(original,manifest)
    require(cpu['status']=='MAIN_RAW64_UNIFIED500_RAW_NORMALLY_COMPLETE_V1' and cpu['all_waited'] is True and
        cpu['source_count']==500 and cpu['config_sha256']==CPU_CONFIG_SHA,'Original normal CPU closure differs')
    require(policies['status']=='POLICIES_FROZEN_ON_CALIBRATION1000' and policies['holdout_used'] is False and
        policies['objective']=='dinov2_vitl14_cosine','Original calibration-frozen raw policies required')
    winners={(r['snr_db'],r['family']):r['candidate_id'] for r in policies['winners']}
    require(winners=={k:r['candidate_id'] for k,r in schedule.items()},'Any raw policy re-selection is forbidden')
    require(restore['schema']=='MISMATCH_RAW100_ORIGINAL_ASSETS_RESTORED_V1' and
        restore['status']=='ORIGINAL_BYTES_RESTORED_NOT_REUSE_ADMITTED' and restore['source_indices']==list(range(100)) and
        restore['source_manifest_sha256']==frozen.SOURCE_MANIFEST_SHA and restore['original_plan_sha256']==frozen.ORIGINAL_PLAN_SHA and
        restore['file_count']==746 and restore['matched_point_files']==500 and restore['matched_physical_frames']==1500 and
        restore['historical_reuse_admitted']==0,'Exact original100 restored-byte receipt required')
    for field in ('source_array_loads','model_calls','packet_calls','metric_calls','new100_population_reads'):
        require(restore[field]==0,'Restore must not perform scientific computation')
    mapping={r['original_path']:r for r in restore['mapping']}
    require(len(mapping)==746,'Repeated restoration entries')
    inputs=[]
    for row in sources:
        item=dict(row)
        for key in ('archive','checkpoint'):
            pin=mapping[row[key]]
            require(pin['sha256']==row[key+'_sha256'],'Original fixed source bytes differ')
            item['actual_'+key]=dict(path=pin['actual_path'],sha256=pin['sha256'],bytes=pin['bytes'])
        inputs.append(item)
    points=[]
    for actual in frozen.ACTUAL_SNRS:
        for i in range(100):
            for candidate in sorted({schedule[actual,f]['candidate_id'] for f in frozen.FAMILIES}):
                old=OLD+f'unified500_raw_parallel_r2/points/{i:04d}/snr{actual}/{candidate}.json';row=mapping[old]
                require(cpu['outputs'][old]==row['sha256'],'Matched candidate point differs from original completion seal')
                points.append(dict(source_index=i,actual_snr_db=actual,candidate_id=candidate,
                    path=row['actual_path'],sha256=row['sha256'],old_noise_seeds=list(frozen.SEEDS),
                    scientific_reuse_admitted=False))
    require(len(points)==500,'Exactly500 candidate points covering1500 matched physical frames required')
    needed=requirements(config);observed={r['original_path']:r for r in inventory}
    require(set(observed)=={r['original_path'] for r in needed},'Inventory must cover exactly the declared dependencies')
    dependencies=[]
    for pin in needed:
        fact=observed[pin['original_path']]
        require(fact['sha256']==pin['sha256'],'Runtime inventory changed original SHA')
        matches=[r for r in fact['actual'] if r['exact'] is True and r['sha256']==pin['sha256']]
        dependencies.append(dict(**pin,available_copies=matches,missing=not matches,checked_paths=fact['checked_paths']))
    return dict(schema=SCHEMA,status='INPUTS_RESTORED_RUNNER_NOT_EXECUTED',scientific_execution_ready=False,
        population_role=base['population_role'],N=1024,source_count=100,noise_count=3,source_records=inputs,
        frozen_cells=base['cells'],noise_spec=frozen.NOISE_SPEC,budget_caps=future_caps(),
        scheduled_accounting=base['accounting'],restored_matched_point_candidates=points,dependencies=dependencies,
        missing_exact_dependencies=[r for r in dependencies if r['missing']],
        implementation_plan=[
            dict(stage='CPU_actual_RX',implementation='Original full433 MixedPublicCatalogue, transmit_frame and Receiver.receive with a new finite charge ledger',
                true_SNR_for_noise_and_both_demappers=True,lookup_SNR_for_TX_policy_only=True,
                counter_rule=frozen.NOISE_SPEC['counter_rule'],noise_namespace_unchanged=True,
                exact_physical_key='Actual source/payload/waveform/noise/received bytes, counter, full433 catalogue, aliases and admitted original PHY runtime',
                scheduled_keys_alone_admit_reuse=False,accepted_wrong_legal_RX_profile_preserved=True,
                body_CRC_failure='KEEP actual hard raw12 tokens; do not drop or correct from TX/source truth'),
            dict(stage='GPU_actual_state',implementation='Frozen original render_received; VAR completion only; original VAE/VAR/Dc and FP32 settings',
                inputs='Only independently parsed actual receiver_state; no source tokens or class labels',
                header_failure='constant0.5 RGB without render',historical_host_image_reuse=False,
                reuse_key='Exact canonical actual state, original renderer/model hashes, actual GPU/native/numerical identity'),
            dict(stage='Four_metrics',implementation='Frozen four computation definitions with explicit mismatch population identity',
                pair_cache='Source pixels SHA plus actual image SHA plus metric identity; no cross-reference pair reuse',
                inference_pairs_maximum=4500,logical_outputs=5400,selection=False),
            dict(stage='Closure_and_comparison',implementation='Actual owner waits and counters; source-paired differences from each true-SNR matched arm',
                retain_zero_and_negative_results=True,body_CRC_denominator='body_attempted only',
                no_new_policy_or_threshold=True,statistics_not_executed=True)],
        historical_admission_pending=['Authenticate original external-owner and all worker waits against original registration',
            'Authenticate paid event request/result evidence and reparse actual hard bits with frozen full433 catalogue',
            'Prove exact intended waveform/noise/actual observation and old PHY numerical/source identity per frame',
            'Current H800 reconstruction/metric identity is distinct; old-host image/value reuse is not admitted'],
        proposed_execution_limits=dict(GPU_workers=1,CPU_cores_per_worker=2,GPU_allocator_cap_GiB=16,
            independent_stage_wall_seconds_max=21600,automatic_successor=False,automatic_retry=False),
        exact_original_bytes_restored=746,new_model_calls=0,new_packet_calls=0,new_metric_calls=0,
        new100_confirmation_content_read=False,old_results_modified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original-plan','source-manifest','cpu-config','cpu-completion','policies','restoration','runtime-inventory','out'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();read=frozen.pinned_json
    result=build(read(a.original_plan,frozen.ORIGINAL_PLAN_SHA),read(a.source_manifest,frozen.SOURCE_MANIFEST_SHA),
        read(a.cpu_config,CPU_CONFIG_SHA),read(a.cpu_completion,CPU_COMPLETE_SHA),read(a.policies,POLICY_SHA),
        read(a.restoration,RESTORE_SHA),json.loads(Path(a.runtime_inventory).read_text(encoding='utf-8')))
    result['input_inventory_sha256']=hashlib.sha256(Path(a.runtime_inventory).read_bytes()).hexdigest()
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status=result['status'],output=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        missing_dependencies=len(result['missing_exact_dependencies']),historical_reuse_admitted=0,new_scientific_calls=0)))


if __name__=='__main__':main()
