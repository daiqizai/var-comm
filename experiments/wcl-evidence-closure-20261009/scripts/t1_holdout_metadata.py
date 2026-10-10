"""Calibration-only entropy policy freeze and post-hoc common500 metadata binding.

No source archive, model, codec, metric, or channel is opened by these commands.
The common500 population was already reported; this is a post-hoc supplement.
"""
import argparse
import csv
import math
from pathlib import Path
from t1_entropy_core import candidates,read,require,sha,verify,write

FAMILIES=['EC_STATIC_WHOLE','EC_VAR_WHOLE'];SNRS=[4,10,19];SEEDS=[6201,6202,6203]
RAW_REFS=['RAW64_WHOLE_VAR_COMPLETION','RAW64_PARTIAL_VAR_COMPLETION']
RAW_DROP=['RAW64_WHOLE_CRC_DROP_VAR_COMPLETION','RAW64_PARTIAL_CRC_DROP_VAR_COMPLETION']
MANIFEST_SHA='b27128fb8eedee7f2cc74bed25c63e39e8add8c938c76d449f85c4b0d5a51e09'

def paired_prefixes():
    return [[f,r] for f in FAMILIES for r in RAW_REFS]+[['EC_VAR_WHOLE','EC_STATIC_WHOLE']]+list(map(list,zip(RAW_DROP,RAW_REFS)))+[[f,r] for f in FAMILIES for r in RAW_DROP]

def pin(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def pinned(d):verify(d['path'],d['sha256']);return read(d['path'])

def freeze(args):
    byid={c['candidate_id']:c for c in candidates()};policies={};inputs={};cal_ids=None;protocol=None
    for name in args.calibration_completion:
        done=read(name)
        require(done['status']=='T1_FULL_CALIBRATION_COMPLETE' and done['source_count']==1000 and done['noise_count']==3
            and done['holdout_used'] is False and done['policy_selection_metric']=='dinov2_vitl14_cosine_only'
            and done['ledger']['unresolved']==0,'Actual full calibration closure required')
        verify(done['request_path'],done['request_sha256']);request=read(done['request_path'])
        require(request['phase']=='full' and request['snrs']==SNRS and request['seeds']==[4101,4102,4103]
            and request['holdout_used'] is False,'Frozen calibration population/noises required')
        ids=[r['source_id'] for r in request['records']]
        require(len(ids)==len(set(ids))==1000,'Complete originalcal1000 IDs required')
        if cal_ids is None:cal_ids=ids
        else:require(cal_ids==ids,'Source populations differ across family calibrations')
        current={k:request[k] for k in ('static_completion','phy_qualification','source_model_id','registered_profile_catalogue','noise_rule','counter_rule')}
        if protocol is None:protocol=current
        else:require(protocol==current,'Source or physical protocol differs across calibrations')
        rankings=done['rankings_path'];verify(rankings,done['outputs'][rankings])
        with Path(rankings).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        require(len(rows)==9*len(done['families']),'Three full candidates per family/SNR required')
        for family in done['families']:
            require(family in FAMILIES and family not in policies,'Duplicate/unknown full family')
            policies[family]={}
            for snr in SNRS:
                group=sorted([r for r in rows if r['family']==family and int(r['snr_db'])==snr],key=lambda r:int(r['rank']))
                require([int(r['rank']) for r in group]==[1,2,3] and len({r['candidate_id'] for r in group})==3,
                    'Actual complete top3 ranking required')
                require(all(int(r['source_count'])==1000 and int(r['noise_count'])==3 and math.isfinite(float(r['dinov2_vitl14_cosine'])) for r in group),
                    'Ranking population or metric invalid')
                require(group==sorted(group,key=lambda r:(-float(r['dinov2_vitl14_cosine']),r['candidate_id'])),
                    'Ranking violates registered DINOv2-L/lexical tie break')
                choice=group[0];c=byid[choice['candidate_id']]
                require(int(choice['target_m'])==c['target_m'] and int(choice['q'])==c['q'] and choice['nominal_rate']==c['nominal_rate'],
                    'Selected candidate fields differ')
                policies[family][str(snr)]=dict(c,calibration_DINOv2_L=float(choice['dinov2_vitl14_cosine']))
        inputs.update({str(Path(name).resolve()):sha(name),done['request_path']:done['request_sha256'],rankings:sha(rankings)})
    require(set(policies)==set(FAMILIES),'Both complete family policies required before common500 supplement')
    result=dict(status='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1',N=1024,policies=policies,
        families=FAMILIES,snrs=SNRS,source_count=1000,noise_count=3,calibration_source_ids=cal_ids,
        selection_metric='dinov2_vitl14_cosine',tie_break='candidate_id_lexical',holdout_used_for_selection=False,
        inference_scope='post_hoc_supplement_on_previously_reported_common500',
        new_paired_prefixes=paired_prefixes(),
        raw_crc_drop_diagnostics=dict(methods=RAW_DROP,original_methods=RAW_REFS,
            rule='Same original actual header/body-CRC reception; only body CRC rejection changes output to constant0.5 RGB; all other original outputs unchanged',
            new_physical_receives=0,TX_truth_used_for_decision=False),
        fallback_rule='target_m descending to4 using only actual arithmetic length; no raw substitution',
        input_bindings=inputs,protocol=protocol,training_updates=0)
    write(args.out,result);return result

def prepare(args):
    policy=read(args.freeze)
    require(policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and policy['holdout_used_for_selection'] is False
        and policy['families']==FAMILIES and policy['snrs']==SNRS,'Both calibration-only family policies must already be frozen')
    for path,digest in policy['input_bindings'].items():verify(path,digest)
    verify(args.original_source_manifest,MANIFEST_SHA);manifest=read(args.original_source_manifest)
    require(manifest['population_role']=='holdout' and manifest['source_count']==500 and len(manifest['records'])==500,
        'Exact published common500 source metadata required')
    old=read(args.original_statistics_completion)
    require(old['scientific_statistics_completed'] and old['source_count']==500,'Completed original published statistics required')
    ids=manifest['source_ids'];require(len(ids)==len(set(ids))==500 and ids==old['source_ids'], 'Exact same500 ordered IDs required')
    require(not set(ids).intersection(policy['calibration_source_ids']),'Calibration overlaps common500')
    original_freeze=pinned(manifest['final_freeze'])
    require(original_freeze['population']['source_ids']==ids,'Original final freeze population differs')
    records=[];bindings={str(Path(args.freeze).resolve()):sha(args.freeze),str(Path(args.original_source_manifest).resolve()):MANIFEST_SHA,
        str(Path(args.original_statistics_completion).resolve()):sha(args.original_statistics_completion),
        manifest['final_freeze']['path']:manifest['final_freeze']['sha256']}
    for index,record in enumerate(manifest['records']):
        require(record['source_index']==index and record['source_id']==ids[index],'Original source index/order differs')
        verify(record['checkpoint'],record['checkpoint_sha256']);cp=read(record['checkpoint'])
        require(cp['status']=='COMMON500_FROZEN_SOURCE_ASSET_READY_V1' and cp['population_role']=='holdout'
            and cp['source_index']==index and cp['source_id']==ids[index] and cp['tokens_sha256']==record['tokens_sha256']
            and cp['preprocessing_id']==record['preprocessing_id'] and cp['archive']==record['archive']
            and cp['outputs'][record['archive']]==record['archive_sha256'] and cp['encoder_tokens_verified']
            and cp['asset_readback_exact'],'Original token/pixel checkpoint identity differs')
        bindings[record['checkpoint']]=record['checkpoint_sha256'];records.append(record)
    result=dict(schema='T1_POSTHOC_COMMON500_SOURCE_REQUEST_V1',status='METADATA_BOUND_NO_SOURCE_ARCHIVE_OPENED',
        root=str(Path(args.root).resolve()),freeze=pin(args.freeze),original_source_manifest=pin(args.original_source_manifest),
        original_statistics_completion=pin(args.original_statistics_completion),source_count=500,source_ids=ids,records=records,
        families=FAMILIES,snrs=SNRS,noise_seeds=SEEDS,N=1024,frame_count=9000,physical_packet_cap=18000,
        post_hoc_supplement=True,holdout_used_for_selection=False,source_archive_opened=False,new_model_calls=0,
        new_packet_decodes=0,training_updates=0,input_bindings=bindings,
        source_preparation=dict(original_encoder_calls=0,VAR_TX_upper_bound=500,VAR_independent_RX_upper_bound=1500,
            rule='One TX traversal through maximum frozen target; independently decode only unique prefixes selected by the three frozen SNR policies'),
        source_bootstrap=dict(replicates=10000,seed=2026100701,unit='source after3-noise mean'),
        cross_method_pairing='source_paired_only; no same-observation claim across methods')
    write(args.out,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    f=s.add_parser('freeze');f.add_argument('--calibration-completion',action='append',required=True);f.add_argument('--out',required=True)
    q=s.add_parser('prepare')
    for n in ('root','freeze','original-source-manifest','original-statistics-completion','out'):q.add_argument('--'+n,required=True)
    args=p.parse_args();value=freeze(args) if args.command=='freeze' else prepare(args);print(value['status'])

if __name__=='__main__':main()
