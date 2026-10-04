"""Frozen-policy real bit-channel execution; images are restored in another env.

This module admits calibration receipts BEFORE opening development TX tokens.
Receiver events contain decoded information only. TX tokens, entropy positions,
labels and offline correctness audits are never renderer inputs.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from uep_common import SIZES,require,identity,read,write,seal,sha,csv_write

VERSION='UEP-N1024-ACTUAL-EVENTS-V1'
SNRS=(4,7,10,13)
SEEDS=(2001,2002,2003)
HERE=Path(__file__).resolve().parent


class StopRequested(Exception):pass


def selected_schedule(policies):
    require(policies.get('status')=='CALIBRATION_POLICIES_SELECTED' and policies.get('synthetic') is False
        and policies.get('stage')=='final' and policies.get('source_count')==1000
        and policies.get('development_read') is False,'Full1000 calibration-selected policies required')
    require(len(policies['source_ids'])==len(set(policies['source_ids']))==1000,'Calibration identity is incomplete')
    chosen={};profiles={};cells={}
    for cell in policies['cells']:
        if cell['N']!=1024 or cell['snr_db'] not in SNRS:continue
        snr=cell['snr_db'];require(snr not in cells,'Duplicate actual-SNR policy cell');cells[snr]=cell
        require(cell.get('ready_for_actual_link') is True and cell.get('probability_refinement_complete') is True,
            'Optimizer has not qualified this actual-link point')
        if 'ready_for_real_link' in cell:require(cell['ready_for_real_link'] is True,'Conflicting actual-link readiness alias')
        labels=[]
        def add(family,choice):
            profile=choice['profile'];sid=str(profile['stable_id'])
            require(choice['stable_id']==sid and profile['N']==1024 and profile.get('encoder_qualified') is True,'Unqualified selected wire profile')
            require(sid not in profiles or profiles[sid]==profile,'One selected profile ID changed')
            profiles[sid]=profile;labels.append(dict(family=family,status='FROZEN',stable_id=sid,wire_key=profile['wire_key']))
        for family in ('B0','B1','B2','B3','B4'):
            value=cell['independent_optima'][family]
            if value['status']=='NOT_FEASIBLE':labels.append(dict(family=family,status='NOT_FEASIBLE'));continue
            require(value['status']=='CALIBRATION_SELECTED','Independent policy is not frozen');add(family,value['selected'])
        require(all(any(r['family']==f and r['status']=='FROZEN' for r in labels) for f in ('B0','B3','B4')),'Required actual policies missing')
        for family in ('B1','B2','B3'):
            value=cell['matched_to_B3'][family];label='matched_'+family
            if value['status'] in ('NOT_APPLICABLE','NOT_FEASIBLE'):
                labels.append(dict(family=label,status=value['status'],reason=value.get('reason','')))
            else:
                require(value['status']=='CALIBRATION_SELECTED','Matched policy is not frozen');add(label,value['selected'])
        add('strong_baseline',cell['strong_baseline'])
        candidate=cell['frozen_distinct_validation_candidate']
        require(candidate['status']=='FROZEN','Calibrated distinct-resource model-validation candidate required')
        add('validation_candidate',candidate['selected']);chosen[str(snr)]=labels
    require(set(cells)==set(SNRS),'All four registered actual SNR policies required')
    return chosen,list(profiles.values())


def admit_frozen(policies_path,quality_bundle_path,quality_completion_path,bler_path,codebook_path):
    """Reusable calibration-only admission; safe for the TX exporter to call."""
    paths=[Path(p).resolve() for p in (policies_path,quality_bundle_path,quality_completion_path,bler_path,codebook_path)]
    policies,quality,completion,bler,codebook=[read(p) for p in paths]
    bindings={str(path):sha(path) for path in paths}
    labels,profiles=selected_schedule(policies)
    require(identity(quality)==policies['quality_input_sha256'] and identity(bler)==policies['bler_input_sha256'],
        'Optimizer quality/probability input changed after selection')
    require(quality['population']=='calibration' and quality['source_ids']==policies['source_ids']
        and quality.get('development_read',False) is False,'Calibration source population differs')
    require(completion['status']=='SOURCE_QUALITY_COMPLETE' and completion['population']=='calibration'
        and completion['stage']=='final1000' and completion['sources']==1000
        and completion['source_ids']==policies['source_ids'] and completion['development_read'] is False
        and completion['metric']=='dinov2_vitl14_cosine','Full1000 actual calibration-Q completion required')
    regpath=paths[2].parent/'registration.json';require(sha(regpath)==completion['registration_sha256'],'Source-Q registration changed')
    qreg=read(regpath);bindings[str(regpath)]=sha(regpath)
    require(qreg['source_count']==1000 and qreg['source_ids']==policies['source_ids'] and qreg['development_read'] is False,'Source-Q registration scope differs')
    # Bind the exact quality values consumed by optimizer to the completed CSV.
    table=paths[2].parent/'quality_per_source.csv';require(completion['outputs'].get(str(table))==sha(table),'Completed quality table changed')
    bindings[str(table)]=sha(table)
    qvalues={}
    for row in quality['rows']:
        key=(row['source_id'],row['state_id'],row['receiver']);require(key not in qvalues,'Duplicate optimizer Q row')
        qvalues[key]=float(row['dinov2_vitl14_cosine'])
    seen=set()
    with table.open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            key=(row['source_id'],row['state_id'],row['receiver']);require(key not in seen and key in qvalues
                and float(row['dinov2_vitl14_cosine'])==qvalues[key],'Optimizer Q differs from completed actual source table');seen.add(key)
    require(seen==set(qvalues),'Optimizer quality population/state coverage differs')
    require(bler['status']=='REFINEMENT_COMPLETE' and bler.get('synthetic') is False,'Actual refined BLER lookup required')
    prob={(r['phy_key'],int(r['snr_db']),r.get('source_id','*')):r for r in bler['rows']}
    require(len(prob)==len(bler['rows']),'Duplicate physical probability row')
    for point in policies['refinement_requirements']:
        if point['snr_db'] not in SNRS:continue
        rows=[r for (key,snr,_),r in prob.items() if key==point['phy_key'] and snr==point['snr_db']]
        require(rows and all(r.get('n_blocks',0)>=256 and
            (r['n_blocks']-r['n_correct']>=100 or r['n_blocks']>=20000) for r in rows),'A selected/refinement physical point is incomplete')
    from profiles import freeze_codebook
    require(codebook==freeze_codebook(profiles),'Final paid codebook does not match all selected actual policies/controls')
    for values in labels.values():
        for label in values:
            if label['status']=='FROZEN':label['profile_id']=codebook['candidate_to_profile'][label['stable_id']]
    required_ms=sorted({p['m'] for p in profiles if p['K']>0})
    return dict(policies=policies,codebook=codebook,labels_by_snr=labels,required_ms=required_ms,
        visual_identity=qreg['renderer_identity']['frozen_identity'],input_bindings=bindings,
        selected_policies_sha256=sha(paths[0]),codebook_sha256=identity(codebook['entries']))


def validate_tx_manifest(manifest,frozen):
    require(manifest['schema']=='uep_tx_tokens_v1' and manifest['population']=='development'
        and manifest['source_count']==100 and manifest['labels_given_to_receiver'] is False,'Exact development100 TX-only manifest required')
    ids=manifest['source_ids'];sources=manifest['sources']
    require(len(ids)==len(set(ids))==len(sources)==100 and ids==[s['source_id'] for s in sources]
        and [s['source_index'] for s in sources]==list(range(100)),'Original development source identity/order differs')
    require(set(ids).isdisjoint(frozen['policies']['source_ids']),'Calibration and development sources overlap')
    require(manifest['selected_policies_sha256']==frozen['selected_policies_sha256']
        and manifest['codebook_sha256']==frozen['codebook_sha256']
        and manifest['visual_identity']==frozen['visual_identity'],'TX source/model/policy frozen identity differs')
    require(manifest.get('required_ms')==frozen['required_ms'],'TX exported a different entropy-order set')
    for source in sources:
        require(source['preprocessing_id'] and source['archive'] and source['archive_sha256'],'TX source artifact missing')
    return sources


def load_tx_source(record,required_ms):
    path=Path(record['archive']);require(sha(path)==record['archive_sha256'],'TX token archive changed')
    expected={'tokens',*['entropy_order_m'+str(m) for m in required_ms]}
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)==expected,'Unexpected TX archive members (labels/extra source data prohibited)')
        tokens=archive['tokens'].copy();require(tokens.shape==(sum(s*s for s in SIZES),) and np.issubdtype(tokens.dtype,np.integer)
            and ((tokens>=0)&(tokens<4096)).all(),'Original true VQ token shape/range differs')
        orders={}
        for m in required_ms:
            values=archive['entropy_order_m'+str(m)].copy();length=SIZES[m]**2
            require(np.issubdtype(values.dtype,np.integer) and values.shape==(length,) and np.array_equal(np.sort(values),np.arange(length)),
                'TX entropy order is not a complete original next-scale position permutation');orders[m]=values
    return np.split(tokens,np.cumsum([s*s for s in SIZES])[:-1]),orders


def clean_receiver_event(event,truth):
    event=copy.deepcopy(event);offline=copy.deepcopy(truth);offline['groups']=[]
    for group in event.get('groups',[]):
        audit=dict(index=group['index'])
        for name in ('truth_correct_after_receiver','false_accept_after_receiver'):
            if name in group:audit[name]=group.pop(name)
        offline['groups'].append(audit)
    # Truth flags cannot silently enter a decoder/cache through a future field.
    def check(value):
        if isinstance(value,dict):
            for name,item in value.items():
                require(not any(term in name.lower() for term in ('truth','class','label','tx_position','source_id','source_index')),'Receiver event contains forbidden TX/offline information')
                check(item)
        elif isinstance(value,list):
            for item in value:check(item)
    check(event);return event,offline


def check_receiver_state(event,codebook):
    from actual_phy import accepted_state
    if not event['header_ok']:
        expected=dict(kind='gray',prefix=[],partial_values=[],m=0,K=0,accepted_groups=0)
    else:
        mapping={int(row['profile_id']):row for row in codebook['entries']}
        require(event['profile_id'] in mapping,'Actual RX profile is absent from paid codebook')
        profile=mapping[event['profile_id']];groups=event['groups']
        require(profile['N']==1024 and len(groups)==profile['G'],'Actual received format differs')
        payloads=[];accepts=[]
        for index,(group,spec) in enumerate(zip(groups,profile['groups'])):
            require(group['index']==index and group['phy_key']==spec['phy_key'] and group['source_bits']==spec['source_bits'],'Actual received group format differs')
            accepted=group['crc_accept'];require(type(accepted)is bool,'Explicit CRC decision required')
            require(group['accepted_payload'] is not None if accepted else group['accepted_payload'] is None,'Rejected hard bits leaked into receiver output')
            value=np.asarray(group['accepted_payload'] if accepted else [])
            if accepted:require(value.shape==(spec['source_bits'],) and np.isin(value,(0,1)).all(),'Invalid accepted group payload')
            value=value.astype(np.uint8)
            payloads.append(value);accepts.append(accepted)
        expected=accepted_state(profile,payloads,accepts)
    require(event['state']==expected,'Rendered state differs from consecutive actually accepted packets')


def schedule_for_source(record,labels_by_snr):
    schedule=[]
    for snr in SNRS:
        methods={}
        for label in labels_by_snr[str(snr)]:
            if label['status']=='FROZEN':methods.setdefault(label['profile_id'],[]).append(label)
        for seed in SEEDS:
            for ordinal,(pid,labels) in enumerate(sorted(methods.items())):
                public=dict(version=VERSION,population='development',source_index=record['source_index'],snr_db=snr,noise_seed=seed,trial_ordinal=ordinal)
                counter=int(identity(public)[:15],16)
                spec=dict(N=1024,snr_db=snr,noise_seed=seed,source_index=record['source_index'],source_id=record['source_id'],
                    profile_id=pid,wire_key=labels[0]['wire_key'],frame_counter=counter,trial_ordinal=ordinal)
                schedule.append(dict(physical_frame_id=identity(public),spec=spec,
                    methods=[dict(family=row['family'],stable_id=row['stable_id']) for row in labels]))
    return schedule


def validate_checkpoint(cp,binding,record,schedule,codebook,complete_required=False,synthetic=False):
    require(cp['binding']==binding and cp['source_index']==record['source_index'] and cp['source_id']==record['source_id']
        and cp['payload_sha256']==identity({k:v for k,v in cp.items() if k!='payload_sha256'}),'Actual source checkpoint changed')
    frames=cp['frames'];require(len(frames)<=len(schedule),'Unexpected extra actual frames')
    for frame,wanted in zip(frames,schedule):
        require(all(frame[k]==wanted[k] for k in ('physical_frame_id','spec','methods')),'Actual frame schedule/policy aliases changed')
        require(frame['actual_bit_chain_executed'] is (not synthetic) and frame['synthetic'] is synthetic,'Nonphysical frame in actual event checkpoint')
        require(frame['ledger']['N']==1024 and frame['receiver_event']['N']==1024,'Actual paid budget changed')
        require(frame['receiver_event']['frame_counter']==frame['spec']['frame_counter'],'Receiver used different public frame counter')
        require(frame['receiver_event']['codebook_sha256']==identity(codebook['entries']),'Receiver used different codebook')
        clean_receiver_event(frame['receiver_event'],{})
        check_receiver_state(frame['receiver_event'],codebook)
    require(cp['complete']==(len(frames)==len(schedule)),'Actual source completion flag differs')
    if complete_required:require(cp['complete'],'Incomplete actual source')
    return cp


def run_source(record,frozen,link,out,binding,stop=lambda:False,synthetic=False):
    """Commit every real physical frame; repeated wire aliases share one event."""
    out=Path(out);path=out/'source_checkpoints'/f'{record["source_index"]:04d}.json'
    from actual_phy import FrameLink
    require(synthetic or isinstance(link,FrameLink),'Only the actual FrameLink may execute scientific physical frames')
    schedule=schedule_for_source(record,frozen['labels_by_snr'])
    cp=(validate_checkpoint(read(path),binding,record,schedule,frozen['codebook'],synthetic=synthetic) if path.exists() else
        dict(binding=binding,source_index=record['source_index'],source_id=record['source_id'],frames=[],complete=False,synthetic=synthetic))
    if cp['complete']:return cp
    if stop():raise StopRequested('Stop requested before TX source access')
    scales,orders=load_tx_source(record,frozen['required_ms']);profiles={row['profile_id']:row for row in frozen['codebook']['entries']}
    from actual_phy import serialize_groups
    payload_cache={}
    for wanted in schedule[len(cp['frames']):]:
        if stop() or (out/'STOP').exists():raise StopRequested('Safe stop requested between actual frames')
        spec=wanted['spec'];pid=spec['profile_id'];profile=profiles[pid]
        if pid not in payload_cache:
            positions=orders[profile['m']][:profile['K']] if profile['K'] else None
            payload_cache[pid]=serialize_groups(scales,profile,positions)
        began=time.monotonic()
        event,ledger,truth=link.execute(pid,payload_cache[pid],1024,spec['snr_db'],spec['frame_counter'],spec['noise_seed'],spec['source_id'])
        receiver,offline=clean_receiver_event(event,truth);check_receiver_state(receiver,frozen['codebook'])
        frame=dict(wanted,receiver_event=receiver,ledger=ledger,offline_truth=offline,
            actual_bit_chain_executed=not synthetic,synthetic=synthetic,audit_used_to_control_receiver=False,
            final_image_cache_used=False,images_restored=False,frame_wall_seconds=time.monotonic()-began,
            independent_receiver_timing=False)
        cp['frames'].append(frame);cp['complete']=len(cp['frames'])==len(schedule)
        cp['payload_sha256']=identity({k:v for k,v in cp.items() if k!='payload_sha256'});write(path,cp)
        write(out/'status.json',dict(status='ACTUAL_BIT_CHAIN_RUNNING',pid=os.getpid(),source_index=record['source_index'],
            source_frames_complete=len(cp['frames']),source_frames_total=len(schedule),updated=time.time()))
    return validate_checkpoint(cp,binding,record,schedule,frozen['codebook'],complete_required=True,synthetic=synthetic)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    for option in ('policies','quality-bundle','quality-completion','bler','codebook','tx-manifest','qualification'):parser.add_argument('--'+option,required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    require(not (out/'failure.json').exists(),'Prior actual-link failure requires review')
    import fcntl
    lock=(out/'evaluate.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    frozen=admit_frozen(args.policies,args.quality_bundle,args.quality_completion,args.bler,args.codebook)
    manifest=read(args.tx_manifest);sources=validate_tx_manifest(manifest,frozen)
    inputs=dict(frozen['input_bindings']);inputs[str(Path(args.tx_manifest).resolve())]=sha(args.tx_manifest)
    for path,digest in manifest['source_bindings'].items():require(sha(path)==digest,'TX exporter identity changed');inputs[path]=digest
    for record in sources:require(sha(record['archive'])==record['archive_sha256'],'TX input archive changed');inputs[record['archive']]=record['archive_sha256']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    from ldpc_backend import SionnaBackend
    from actual_phy import FrameLink
    backend=SionnaBackend(args.device,args.qualification);require(backend.qualified,'Actual LDPC backend unqualified')
    qualification=read(args.qualification);inputs[str(Path(args.qualification).resolve())]=sha(args.qualification)
    for path,digest in qualification['source_bindings'].items():require(sha(path)==digest,'Qualified real PHY source changed');inputs[path]=digest
    for row in frozen['codebook']['entries']:require(row['backend_id']==identity(backend.identity),'Selected LDPC backend identity differs')
    own={str(HERE/name):sha(HERE/name) for name in ('evaluate.py','test_evaluate.py','actual_phy.py','uep_phy.py','uep_common.py','ldpc_backend.py','profiles.py','optimize.py')}
    inputs.update(own);link=FrameLink(args.root,backend,frozen['codebook'])
    registration=dict(status='UEP_ACTUAL_EVENTS_REGISTERED',version=VERSION,population='development',N=1024,
        snrs=list(SNRS),noise_seeds=list(SEEDS),sources=100,source_ids=manifest['source_ids'],
        source_identity=[{k:record[k] for k in ('source_index','source_id','preprocessing_id')} for record in sources],
        selected_policies_sha256=frozen['selected_policies_sha256'],codebook_sha256=frozen['codebook_sha256'],
        labels_by_snr=frozen['labels_by_snr'],visual_identity=frozen['visual_identity'],input_bindings=inputs,
        backend_identity=backend.identity,device=backend.device,received_format='actual_crc_accepted_prefix_v1',
        actual_bit_chain_executed=True,synthetic=False,training_updates=0,policy_selection_updates=0,
        labels_given_to_receiver=False,TX_positions_given_to_receiver=False,renderer_uses='receiver_event.state only; entropy order recomputed from received prefix',
        waveform_noise_pairing='same N/source/noise seed base AWGN across all compared actual wire configurations',
        independent_receiver_timing=False,final_image_cache_used=False)
    seal(out/'registration.json',registration);binding=identity(registration);outputs={};flat=[];physical=0
    try:
        for record in sources:
            cp=run_source(record,frozen,link,out,binding,stop=lambda:stop[0]);physical+=len(cp['frames'])
            path=out/'source_checkpoints'/f'{record["source_index"]:04d}.json';outputs[str(path)]=sha(path)
            for frame in cp['frames']:
                for method in frame['methods']:
                    flat.append(dict(frame['spec'],**method,physical_frame_id=frame['physical_frame_id'],
                        source_checkpoint=str(path),source_checkpoint_sha256=outputs[str(path)],actual_bit_chain_executed=True,synthetic=False,
                        E=frame['ledger']['E'],waveform_sha256=frame['ledger']['waveform_sha256'],
                        received_sha256=frame['receiver_event']['received_sha256'],noise_sha256=frame['offline_truth']['noise_sha256'],
                        header_ok=frame['receiver_event']['header_ok'],header_correct=frame['offline_truth']['header_correct'],
                        header_false_accept=frame['offline_truth']['header_false_accept'],
                        undetected_body_errors=frame['offline_truth']['undetected_body_errors'],state_sha256=identity(frame['receiver_event']['state']),
                        accepted_groups=frame['receiver_event']['state']['accepted_groups'],m=frame['receiver_event']['state']['m'],K=frame['receiver_event']['state']['K']))
        for path,digest in inputs.items():require(sha(path)==digest,'Input changed during actual-link evaluation')
        table=out/'actual_events_per_frame.csv';csv_write(table,flat);outputs[str(table)]=sha(table)
        completion=dict(status='UEP_ACTUAL_EVENTS_COMPLETE',sources=100,physical_frames=physical,method_rows=len(flat),
            population='development',N=1024,snrs=list(SNRS),noise_seeds=list(SEEDS),registration_sha256=sha(out/'registration.json'),
            input_bindings=inputs,outputs=outputs,synthetic=False,actual_bit_chain_executed=True,
            images_restored=False,image_metrics_pending=True,training_updates=0,policy_selection_updates=0)
        seal(out/'completion.json',completion);write(out/'status.json',completion);print(json.dumps(completion))
    except StopRequested as error:write(out/'status.json',dict(status='PAUSED',reason=str(error),pid=os.getpid()));raise SystemExit(75)
    except Exception as error:write(out/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error)));raise


if __name__=='__main__':main()
