"""Read-only CPU closure and delivery of the already selected R1 calibration.

No model/PHY operation, candidate selection, bootstrap, retry or R2 entry exists
here. Before any output is created, both process layers and all scientific stages
must have actually closed. Heavy derived tables remain in personal temporary
storage; publication files obey the unchanged 10,000,000-byte ceiling.
"""
from __future__ import annotations
import argparse
import collections
import copy
import csv
import hashlib
import importlib.abc
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
TMP = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013')
MASTER_SHA = '70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47'
STAGES = ('phy', 'projection', 'visual', 'score', 'selection')
STAGE_STATUS = dict(phy='COMPLETE_R1_PHY_ONLY',projection='COMPLETE_R1_RX_PROJECTION',
    visual='COMPLETE_R1_VISUAL',score='COMPLETE_R1_SCORE',selection='COMPLETE_R1_CALIBRATION_SELECTION')
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine')
METHODS = ('U_two', 'S_robust', 'S_nom', 'S_back3', 'E_two')
LIMIT = 10_000_000
PUBLICATION_MODE = 'EXCLUSIVE_DIRECTORY_FILE_COPY_AND_HARDLINK_COMPLETION_V1'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def pin(path):
    path = Path(path).absolute()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked(value):
    path = Path(value['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == value['sha256'],
            'Bound file changed: ' + str(path))
    if 'bytes' in value: require(type(value['bytes']) is int and path.stat().st_size == value['bytes'], 'Bound bytes differ')
    return path


def readpin(value): return read(checked(value))
def same_pin(left, right): return left['path'] == right['path'] and left['sha256'] == right['sha256']


def process_identity(pid):
    try:
        value = (Path('/proc')/str(pid)/'stat').read_text()
        tail = value[value.rfind(')')+2:].split()
        return dict(pid=int(pid), start_ticks=int(tail[19]), state=tail[0])
    except (OSError, ValueError): return None


def require_gone(identity, probe=process_identity):
    require(type(identity.get('pid')) is int and type(identity.get('start_ticks')) is int, 'Missing process identity')
    now = probe(identity['pid'])
    require(now is None or now['start_ticks'] != identity['start_ticks'], 'Original process still exists (including zombie)')


def same_process(left,right):
    return all(type(value.get(key)) is int for value in (left,right) for key in ('pid','start_ticks')) and all(
        left[key]==right[key] for key in ('pid','start_ticks'))


def actual_zero(wait, identity):
    require(wait.get('actual_wait') is True and type(wait.get('returncode')) is int and wait['returncode'] == 0
            and type(wait.get('pid')) is int and wait['pid'] == identity['pid'],
            'Actual wait is not successful for the dispatched process')
    require(not any(wait.get(k, False) for k in ('timeout','timed_out','terminated','killed','stopped')),
            'Interrupted child cannot be delivered as complete')
    if 'start_ticks' in wait: require(type(wait['start_ticks']) is int and wait['start_ticks'] == identity['start_ticks'], 'Wait start ticks differ')


def routed(reg, original, args):
    root = Path(reg['pins']['common_ledger']['path']).parents[2]
    code = root/'experiments/layered-mismatch-raw-20261013/r1_source_identity_revision_v1'
    runner, adapter = code/'runner.py', code/'pin_adapter.py'
    require(reg['code_bindings'][str(runner)] == sha(runner) and reg['code_bindings'][str(adapter)] == sha(adapter),
            'Actual metadata routing code changed')
    return [str(runner),'--runner-sha256',sha(runner),'--adapter-sha256',sha(adapter),'--original',str(original),*args]


def unheld_locks(paths):
    require(sys.platform.startswith('linux'),'Actual delivery is Linux-only')
    import fcntl
    for path in paths:
        require(path.is_file() and not path.is_symlink(),'Original owner lock missing')
        with path.open('rb') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def preflight(root, temporary, registration, probe=process_identity):
    """Only small immutable metadata and process identities; no quality reads."""
    reg = readpin(registration)
    require(registration['sha256'] == MASTER_SHA and reg['expected_physical_frames'] == 32400
            and reg['expected_logical_var_images'] == 43200, 'Different R1 run')
    run = temporary/'r1_execution_v1'; result = root/'results/layered_mismatch_raw_20261013/r1_execution_v1'
    require(Path(reg['runtime']['root']) == run, 'Authoritative runtime differs')
    for stage in STAGES: require(Path(reg['runtime'][stage+'_directory']) == run/stage, 'Stage location differs')
    dispatch, control = temporary/'r1_resource_dispatch_v2', temporary/'r1_initial_dispatch_control_v1'
    final = read(result/'completion.json')
    require(final['status'] == 'COMPLETE_R1_CHAIN' and final['actual_exit_zero_all'] is True
            and final['automatic_R2'] is False and final['R2'] == 'NOT_REGISTERED_NOT_RUN'
            and same_pin(final['registration'], registration), 'R1 chain is not actually complete')
    require([x['stage'] for x in final['stages']] == list(STAGES), 'Missing or reordered stage closure')
    for base in (run, dispatch):
        for path in base.rglob('*.json'):
            require('failure' not in path.name and not path.name.startswith(('partial_', 'interrupted_')),
                    'Preserved CPU execution failure or partial result prevents delivery: ' + str(path))
    owner = read(run/'owner_launch.json'); parent = read(dispatch/'launch.json'); outer = read(control/'launch.json')
    owner_id = read(dispatch/'owner_identity.json')
    require(same_process(parent['process'],outer['process']) and same_process(owner_id['process'],owner['process']), 'Dispatcher process chain differs')
    for receipt in (owner,parent,outer,owner_id):
        require(same_pin(receipt['registration'], registration), 'Dispatch registration differs')
        require_gone(receipt['process'],probe)
    owner_wait = read(dispatch/'actual_wait.json'); actual_zero(owner_wait,owner['process'])
    require(same_process(owner_wait['identity_at_launch'],owner['process']) and same_pin(owner_wait['registration'],registration)
            and owner_wait['scientific_owner_launch_exists'] is True, 'Owner actual wait provenance differs')
    resource = root/'experiments/layered-mismatch-raw-20261013/r1_execution_resource_v2'
    cpu, visual = reg['runtime']['python_cpu'], reg['runtime']['python_visual']
    require(outer['argv'] == [cpu,'-B','-u',*routed(reg,resource/'dispatch_once.py',['--expected-registration',registration['sha256']])],
            'Initial dispatcher argv differs')
    require(parent['argv'] == [cpu,'-B','-u',*routed(reg,resource/'owner.py',['--registration',registration['path'],'--registration-sha256',registration['sha256']])],
            'Owner argv differs')
    for stage, closure in zip(STAGES,final['stages']):
        launch=read(run/(stage+'_launch.json')); wait=read(run/stage/'actual_wait.json')
        require(launch['stage']==stage and same_pin(launch['registration'],registration), 'Stage launch binding differs')
        actual_zero(wait,launch['process']);require_gone(launch['process'],probe)
        require(wait == read(run/(stage+'_actual_wait.json')) and wait.get('stage')==stage, 'Owner/child wait differs')
        require(same_pin(closure['actual_wait'],pin(run/stage/'actual_wait.json'))
                and same_pin(closure['completion'],pin(run/stage/'completion.json')), 'Chain stage pins differ')
        # Small completion metadata is authenticated before any quality file is
        # opened. Exact mapping matches the original owner.closed_stage.
        complete=read(run/stage/'completion.json')
        expected=reg['phy_registration'] if stage=='phy' else (reg['visual_registration'] if stage=='visual' else registration)
        require(complete['status']==STAGE_STATUS[stage] and same_pin(complete['registration'],expected),
                'Stage status or child registration is not closed: '+stage)
        if stage=='phy':
            ledger=complete['ledger']
            require(ledger['physical_complete']==ledger['physical_reserved']==32400
                    and ledger['packet_decodes_complete']==ledger['packet_decodes_reserved']<=108000
                    and ledger['unresolved_packets']==0,'Physical ledger is not closed')
        if stage=='phy':
            script=resource.parent/'r1_phy.py';sub=reg['phy_registration'];py=cpu
            args=['run','--registration',sub['path'],'--registration-sha256',sub['sha256']]
        elif stage in ('projection','visual'):
            script=resource/'receiver_visual.py';sub=reg['visual_registration'] if stage=='visual' else registration
            py=visual if stage=='visual' else cpu
            args=['visual' if stage=='visual' else 'project','--registration',sub['path'],'--expected-registration',sub['sha256'],
                  '--parent-pid',str(owner['process']['pid']),'--parent-ticks',str(owner['process']['start_ticks'])]
        else:
            script=resource/'score_selection.py';py=visual if stage=='score' else cpu
            args=['score' if stage=='score' else 'select','--registration',registration['path'],
                  '--registration-sha256',registration['sha256'],'--parent',str(owner['process']['pid'])]
        require(launch['argv']==[py,'-B','-u',*routed(reg,script,args)],'Actual stage argv changed: '+stage)
    require(same_pin(final['frozen_policy_completion'],pin(run/'selection/completion.json')), 'Final selected result differs')
    unheld_locks((run/'owner.lock',dispatch/'dispatch.lock'))
    return reg, dict(chain_completion=pin(result/'completion.json'),owner_actual_wait=pin(dispatch/'actual_wait.json'),
        initial_control_launch=pin(control/'launch.json'),dispatch_launch=pin(dispatch/'launch.json'),
        initial_supervisor_observation='Original PID/start no longer exists; initial detached launcher has no child-wait receipt',
        all_stage_actual_waits=[pin(run/stage/'actual_wait.json') for stage in STAGES],all_original_processes_gone=True)


class CPUReadOnly(importlib.abc.MetaPathFinder):
    """Fail closed against model imports, subprocesses and writes while auditing."""
    active = True
    def find_spec(self, fullname, path=None, target=None):
        require(fullname.split('.')[0] not in ('torch','tensorflow','sionna','lpips'), 'Scientific library import forbidden')
        return None
    def audit(self,event,args):
        if not self.active:return
        require(event not in ('subprocess.Popen','os.system','socket.connect','os.mkdir','os.remove','os.rmdir','os.rename'),
                'Audit attempted a side effect: '+event)
        if event=='open':
            mode=args[1];flags=args[2]
            require(not (isinstance(mode,str) and any(x in mode for x in 'wax+'))
                    and not (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC)),
                    'Audit attempted file writing')


def load_bound(path, name, reg):
    path=Path(path);require(reg['code_bindings'][str(path)]==sha(path),'Unbound CPU audit code')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def compare_csv(path, expected):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    require(len(rows)==len(expected) and rows,'CSV row count differs')
    for actual, wanted in zip(rows,expected):
        require(set(actual)==set(wanted),'CSV fields differ')
        for key,value in wanted.items():
            require(actual[key]==str(value),'Saved CSV differs from authenticated value: '+key)


def verify_existing_selection(selected, means, source_scores, expected, nominations, sources, policy):
    """Validate saved identities/ranks/J/gate; never construct or choose a winner."""
    require(selected['authenticated_actual_closure'] is True and selected['no_bootstrap'] is True
            and selected['bootstrap_calls']==0 and selected['source_count']==200 and selected['noise_count']==3
            and selected['logical_rows']==43200 and selected['R2']=='NOT_REGISTERED_NOT_RUN'
            and selected['stress_true_snr_plus3_used_for_selection'] is False,'Selection scope differs')
    winners={(x['method'],x['snr_hat']):x for x in selected['winners']}
    require(set(winners)=={(m,h) for m in METHODS for h in (7,10)} and len(selected['winners'])==10,'Frozen winners incomplete')
    ranked=collections.defaultdict(list)
    for row in selected['ranking']:ranked[(row['method'],row['snr_hat'])].append(row)
    require(set(ranked)==set(winners),'Frozen ranking missing methods')
    reverse={display:arm for arm,display in policy.ARM_NAMES.items()}
    for (method,hat), winner in winners.items():
        arm=reverse[method];eligible={'S','S_nom','S_back3'} if arm=='S' else {arm}
        allowed={(n['candidate_id'],rule) for n in nominations if n['snr_hat']==hat and n['arm'] in eligible for rule in n['receiver_variants']}
        rows=ranked[(method,hat)]
        require({(r['candidate_id'],r['receiver_rule']) for r in rows}==allowed and len(rows)==len(allowed), 'Saved candidate competition differs')
        require([r['rank'] for r in rows]==list(range(1,len(rows)+1)),'Saved rank ordering differs')
        previous=None
        for row in rows:
            cid,rule=row['candidate_id'],row['receiver_rule'];score=math.fsum(source_scores[(hat,cid,rule)])/200
            require(row['DINO_J']==score,'Saved candidate J differs from source means')
            arms=expected[(hat,cid)]['arms'];anchor=0 if 'S_nom' in arms else (1 if 'S_back3' in arms else 2)
            order=(-score,(anchor,cid,0 if rule=='KEEP' else 1))
            require(previous is None or previous<=order,'Saved ranking violates frozen score/tie order');previous=order
        first=rows[0]
        require((winner['candidate_id'],winner['receiver_rule'],winner['cal200_DINO_J'])==
                (first['candidate_id'],first['receiver_rule'],first['DINO_J']),'Saved winner is not saved rank one')
        require(winner['physical_key']==expected[(hat,winner['candidate_id'])]['physical_key']
                and winner['selection_metric']=='dinov2_vitl14_cosine' and winner['selection_p']==.25
                and winner['pressure_e_minus3_used_for_selection'] is False and winner['convnext_used_for_selection'] is False,
                'Winner protocol identity differs')
    j={m:{h:winners[(m,h)]['cal200_DINO_J'] for h in (7,10)} for m in ('U_two','S_robust','S_back3')}
    # Arithmetic check of the already stored decision, not a new decision output.
    import contract
    require(selected['gate']==json.loads(json.dumps(contract.calibration_gate(j))),'Stored GO is inconsistent with frozen winners')
    return winners


def deep_audit(root, temporary, registration, reg):
    experiment=root/'experiments/layered-mismatch-raw-20261013';resource=experiment/'r1_execution_resource_v2'
    for directory in (experiment,resource,experiment/'r1_source_identity_revision_v1'):
        sys.path.insert(0,str(directory))
    adapter=load_bound(experiment/'r1_source_identity_revision_v1/pin_adapter.py','pin_adapter',reg);adapter.install_phy()
    shared=load_bound(resource/'shared.py','shared',reg);require(shared.validate_registration(registration)==reg,'Master changed')
    receiver=load_bound(resource/'receiver_visual.py','receiver_visual',reg)
    score=load_bound(resource/'score_selection.py','score_selection',reg)
    policy=load_bound(experiment/'policy_selection.py','policy_selection',reg)
    run=temporary/'r1_execution_v1';selection=read(run/'selection/completion.json')
    require(selection['status']=='COMPLETE_R1_CALIBRATION_SELECTION' and same_pin(selection['registration'],registration)
            and selection['scientific_calls']==selection['bootstrap_calls']==0 and selection['R2']=='NOT_REGISTERED_NOT_RUN',
            'Selection completion differs')
    phy_audit,logical,frames=receiver.audit_phy(reg)
    projection=read(run/'projection/completion.json');require(readpin(projection['audit'])==phy_audit,'Saved PHY audit differs')
    joins=readpin(projection['logical_join']);model=readpin(projection['model_inputs'])
    require(len(joins)==len(model)==43200,'Projection grid incomplete')
    for index,row in enumerate(logical):
        expected_model,expected_join=receiver.project_one(row,frames[row['actual_RX']['sha256']],reg['phy_registration']['sha256'])
        require(model[index]==expected_model and joins[index]==expected_join,'Projection differs from actual RX')
    visual_rows,sources,nominations,visual=score.closed_visual(reg,registration)
    require(visual==selection['visual_closure'],'Selected visual closure changed')
    score_wait=score.successful_wait(reg,registration,'score');done=read(run/'score/completion.json')
    require(same_pin(selection['score_completion'],pin(run/'score/completion.json')) and selection['score_actual_wait']==score_wait
            and done['status']=='COMPLETE_R1_SCORE' and same_pin(done['registration'],registration)
            and done['visual_closure']==visual and done['metrics']==list(METRICS)
            and done['source_count']==200 and done['noise_count']==3
            and done['bootstrap_calls']==done['ConvNeXt_calls']==done['training_updates']==0,'Scoring closure differs')
    rows=readpin(done['scored_rows']);metadata=readpin(done['metric_identity']);references=readpin(done['source_preparations'])
    require(metadata['requested_metrics']==list(METRICS) and metadata['numeric_flags']==score.FLAGS
            and metadata['original_sources']==reg['metric_config']['sources'] and metadata['ConvNeXt_constructed'] is False
            and metadata['source_and_reconstruction_preprocessing_identical'] is True
            and metadata['received_truth_supplied_to_reconstruction'] is False,'Actual metric identity differs')
    identity=score.digest(metadata);unique=score.authenticate_scored_rows(rows,visual_rows,sources,identity)
    require(len(rows)==done['logical_rows']==43200 and unique==done['unique_actual_pairs'],'Actual scoring counts differ')
    require([r['source_id'] for r in references]==[s['source_id'] for s in sources],'Reference source order differs')
    for reference,source in zip(references,sources):
        pixels=score.source_pixels(source)
        require(reference==dict(source_id=source['source_id'],source_archive=source['archive'],
                source_preprocessing_id=source['preprocessing_id'],target_array_sha256=score.array_sha(pixels),metric_identity=identity),
                'Original source preprocessing changed')
    score.check_score_counts(done['counts'],len(rows),unique);score.audit_score_ledger(done['ledger'],done['counts'],rows,references)
    for index,row in enumerate(rows):require(read(run/'score/rows'/f'{index:05d}.json')==row,'Per-frame score receipt changed')
    means,source_scores,expected=policy.validate_and_reduce(rows,[s['source_id'] for s in sources],nominations)
    source_rows=[dict(source_id=k[0],snr_hat=k[1],snr_true=k[2],candidate_id=k[3],receiver_rule=k[4],metric=k[5],mean=v,noise_count=3)
                 for k,v in sorted(means.items())]
    require(readpin(selection['source_means'])==source_rows,'Stored source means differ')
    compare_csv(checked(selection['source_means_csv']),source_rows)
    selected=readpin(selection['selection']);require(same_pin(selected['registration'],registration),'Selected master differs')
    winners=verify_existing_selection(selected,means,source_scores,expected,nominations,sources,policy)
    summary,winner_summary,engineering=score.quality_summaries(means,[s['source_id'] for s in sources],selected)
    compare_csv(checked(selection['candidate_summary_csv']),summary);compare_csv(checked(selection['frozen_winner_summary_csv']),winner_summary)
    require(selected['separate_engineering_targets']==engineering,'Separate engineering targets changed')
    require(len(frames)==32400 and len(logical)==43200 and len(source_rows)==43200 and len(summary)==216 and len(winner_summary)==90,
            'Final fixed grid/table counts differ')
    catalogue=readpin(reg['pins']['grouped_catalogue'])['profiles']
    return dict(rows=rows,joins=joins,frames=frames,sources=sources,nominations=nominations,selected=selected,winners=winners,
                catalogue=catalogue,selection=selection,source_rows=source_rows,summary=summary,winner_summary=winner_summary,
                audit=dict(phy=phy_audit,visual=visual,score_completion=pin(run/'score/completion.json'),
                    selection_completion=pin(run/'selection/completion.json'),metric_identity=done['metric_identity'],
                    physical_frames=32400,logical_frames=43200,source_count=200,noise_count=3,
                    no_reselection=True,no_inference=True,no_bootstrap=True,all_signal_failures_retained=True))


def flatten(row,join,frame):
    require(row['view_id']==join['view_id'],'Flatten join differs')
    event=frame['actual_RX'];tx=frame['input_identity'];state=frame['receiver_views'][row['receiver_rule']]
    grouped=tx['G']==2;received_id=event['profile_id'] if grouped else event['rx_profile_id']
    groups=event.get('groups',[]) if grouped else []
    # Both the original raw present_actual and grouped wire.receive expose these
    # exact top-level flags. Missing evidence is an error, never a blank default.
    require(all(type(event[k]) is bool for k in ('header_ok','header_crc_ok','header_fields_legal')),
            'Missing actual header acceptance flags')
    header_ok=event['header_ok'];require(header_ok==(event['header_crc_ok'] and event['header_fields_legal']),
                                      'Actual header acceptance contradicts CRC/legality')
    status=(event['receive_status'] if grouped else ('HEAD_REJECT' if not header_ok else
            ('SINGLE_CRC_ACCEPT' if event['body_crc_accept'] else 'SINGLE_CRC_REJECT_'+row['receiver_rule'])))
    # Wrong legal public IDs are a distinct state, never counted as true-format full/base success.
    state_class=('WRONG_LEGAL_ID' if header_ok and received_id!=tx['profile_id'] else status)
    truth=frame['post_receiver_truth']
    out={k:row[k] for k in ('source_id','source_index','selection_slot','snr_hat','snr_true','noise_seed','candidate_id','physical_key','receiver_rule')}
    out.update(tx_profile_id=tx['profile_id'],rx_profile_id=received_id,tx_group_count=tx['G'],
        header_crc_ok=event['header_crc_ok'],header_fields_legal=event['header_fields_legal'],
        header_ok=header_ok,g1_crc_accept=(groups[0]['crc_accept'] if groups else (event.get('body_crc_accept') if header_ok and not grouped else None)),
        g2_crc_accept=(groups[1]['crc_accept'] if len(groups)>1 else None),received_group_count=event.get('received_group_count',1 if header_ok and not grouped else None),
        receive_status=status,state_class=state_class,actual_received_kind=join['actual_received_kind'],
        actual_received_m=join['actual_received_m'],actual_received_K=join['actual_received_K'],
        retained_token_count=sum(len(x) for x in state['prefix'])+len(state['partial_values']),
        fixed_gray_output=join['actual_failure_state'],software_execution_status=row['execution_status'],
        actual_frame_energy=frame['actual_frame_energy'],actual_frame_mean_Es=frame['actual_frame_mean_Es'],
        channel_noise_variance_per_component=frame['channel_noise_variance_per_component'],
        post_receiver_header_false_accept=truth['header_false_accept'],post_receiver_body_comparison_available=truth['body_comparison_available'],
        post_receiver_truth_json=json.dumps(truth,sort_keys=True,separators=(',',':')))
    out.update({k:row[k] for k in METRICS})
    out.update({k:row[k] for k in ('view_id','observation_key','actual_rx_key','actual_input_key','receiver_state_sha256',
                                 'model_identity','metric_identity','metric_pair_key','metric_reuse','first_scored_view_id','reconstruction_sha256')})
    out.update(source_truth_supplied_to_RX=False,failed_logical_observation_retained=True)
    return out


def state_summary(rows, keys):
    groups=collections.defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    output=[]
    for identity,group in sorted(groups.items()):
        require(len(group)==600,'Each candidate/winner condition must retain 200 sources x 3 noises')
        require(len({r['source_id'] for r in group})==200,'State denominator source mismatch')
        states=collections.defaultdict(list)
        if group[0]['tx_group_count']==2:
            labels=('HEAD_REJECT','G1_REJECT','BASE_ONLY','FULL_PREFIX','WRONG_LEGAL_ID')
        else:
            labels=('HEAD_REJECT','SINGLE_CRC_ACCEPT','SINGLE_CRC_REJECT_'+group[0]['receiver_rule'],'WRONG_LEGAL_ID')
        for label in labels:states[label]=[]
        for row in group:states[row['state_class']].append(row)
        for state,values in sorted(states.items()):
            result=dict(zip(keys,identity));result.update(state_class=state,frames=len(values),all_frames=600,
                fraction=len(values)/600,source_count=200,noise_count=3,conditioning='Frame-conditioned descriptive mean; not a source-paired effect')
            for metric in METRICS:result[metric+'_conditional_mean']=(math.fsum(r[metric] for r in values)/len(values) if values else None)
            result['header_false_accept_count']=sum(r['post_receiver_header_false_accept'] for r in values)
            for g in ('g1','g2'):
                attempts=[r[g+'_crc_accept'] for r in values if r[g+'_crc_accept'] is not None]
                result[g+'_attempted']=len(attempts);result[g+'_crc_rejected']=sum(x is False for x in attempts)
                result[g+'_rejection_rate_if_attempted']=(sum(x is False for x in attempts)/len(attempts) if attempts else None)
            output.append(result)
    return output


def physical_rows(flat):
    fields=('source_id','source_index','selection_slot','snr_hat','snr_true','noise_seed','physical_key',
            'tx_profile_id','rx_profile_id','tx_group_count','header_crc_ok','header_fields_legal','header_ok',
            'g1_crc_accept','g2_crc_accept','received_group_count','actual_frame_energy','actual_frame_mean_Es',
            'channel_noise_variance_per_component','post_receiver_header_false_accept',
            'post_receiver_body_comparison_available','post_receiver_truth_json','observation_key','actual_rx_key')
    unique={}
    for row in flat:
        key=tuple(row[k] for k in ('source_id','snr_hat','snr_true','noise_seed','physical_key'))
        value={k:row[k] for k in fields}
        require(key not in unique or unique[key]==value,'KEEP/DROP or aliases differ on physical audit values')
        unique[key]=value
    return list(unique.values())


def method_parameters(data,flat):
    profiles={p['profile_id']:p for p in data['catalogue']};lookup={}
    for n in data['nominations']:lookup[(n['snr_hat'],n['candidate_id'])]=n
    result=[]
    for (method,hat),winner in sorted(data['winners'].items()):
        n=lookup[(hat,winner['candidate_id'])];p=profiles[n['profile_id']]
        values=[r for r in flat if r['snr_hat']==hat and r['candidate_id']==winner['candidate_id'] and r['receiver_rule']==winner['receiver_rule']]
        require(len(values)==1800,'Frozen winner not backed by all six source/noise cells')
        row=dict(method=method,snr_hat=hat,candidate_id=winner['candidate_id'],profile_id=n['profile_id'],receiver_rule=winner['receiver_rule'],
            G=p['G'],j=p.get('j'),m=p['m'],K=p['K'],token_count=p['token_count'],header_symbols=68,
            body_symbols=sum(g['symbols'] for g in p['groups']),idle_symbols=p['idle_symbols'],
            cal200_DINO_J=winner['cal200_DINO_J'],groups_json=json.dumps(p['groups'],sort_keys=True,separators=(',',':')))
        for i in (0,1):
            g=p['groups'][i] if i<len(p['groups']) else {}
            for key in ('modulation','symbols','source_bits','information_bits','transmitted_bits','crc_bits','nominal_rate','effective_information_rate','source_rate'):
                row[f'group{i+1}_{key}']=g.get(key)
        result.append(row)
    return result


def write_csv(path,rows):
    require(rows,'Refusing empty derived table')
    with Path(path).open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_json(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')


def copy_selected(source,light,payload):
    source=Path(source);target=(light if source.stat().st_size<LIMIT else payload)/source.name
    require(not target.exists(),'Duplicate copied artifact')
    shutil.copyfile(source,target);require(sha(target)==sha(source),'Selected result copy changed bytes')
    return target


def require_absent(path):
    """lstat rejects existing directories, files and dangling symbolic links."""
    try:Path(path).lstat()
    except FileNotFoundError:return
    raise FileExistsError('Existing path must not be overwritten: '+str(path))


def filesystem_identity(path,kind):
    info=Path(path).lstat()
    require((stat.S_ISDIR(info.st_mode) if kind=='directory' else stat.S_ISREG(info.st_mode)),
            'Unexpected file type (including symlink): '+str(path))
    return dict(device=info.st_dev,inode=info.st_ino)


def verify_claim(claim):
    checked(claim['file'])
    require(filesystem_identity(claim['file']['path'],'file')==claim['identity'],'Publication claim identity changed')


def fsync_file(path):
    # Only newly created claim/staging files reach this helper. Writable access
    # permits fsync on Windows synthetic fixtures as well as the probed Linux PFS.
    flags=os.O_RDWR|getattr(os,'O_NOFOLLOW',0)
    descriptor=os.open(path,flags)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def fsync_directory(path):
    require(sys.platform.startswith('linux'),'Actual publication requires probed Linux directory fsync')
    descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def exclusive_copy(source,target,directory_identity):
    source,target=Path(source),Path(target)
    require(filesystem_identity(target.parent,'directory')==directory_identity,'Reserved directory changed')
    filesystem_identity(source,'file');expected=pin(source)
    descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(descriptor,'wb') as output,source.open('rb') as incoming:
        shutil.copyfileobj(incoming,output,8<<20);output.flush();os.fsync(output.fileno())
    require(filesystem_identity(target.parent,'directory')==directory_identity,'Reserved directory changed during copy')
    require(target.stat().st_size==expected['bytes'] and sha(target)==expected['sha256'],'Exclusive copy bytes differ')


def publish_directory(source,target,claim,exclude=()):
    """Reserve a new final directory, then exclusively create verified files."""
    verify_claim(claim);require_absent(target)
    target.mkdir(mode=0o700,exist_ok=False);identity=filesystem_identity(target,'directory');fsync_directory(target.parent)
    for path in sorted(source.iterdir()):
        if path.name in exclude:continue
        verify_claim(claim);exclusive_copy(path,target/path.name,identity)
    fsync_directory(target);return identity


def publish_completion(source,target,directory_identity,claim):
    """Exclusive receipt copy, then same-directory atomic completion link."""
    verify_claim(claim)
    require(filesystem_identity(target.parent,'directory')==directory_identity,'Completion directory changed')
    filesystem_identity(source,'file');require_absent(target);expected=pin(source)
    pending=target.with_name('completion.pending.json');require_absent(pending)
    exclusive_copy(source,pending,directory_identity)
    require(pending.stat().st_size==expected['bytes'] and sha(pending)==expected['sha256']
            and sha(source)==expected['sha256'],'Pending completion differs from original receipt')
    fsync_file(pending);fsync_directory(target.parent);verify_claim(claim)
    # tmp and code are distinct PFS areas (cross-area link returns EXDEV).
    # The fully copied/fsynced pending receipt and marker share this directory.
    os.link(pending,target,follow_symlinks=False)  # No fallback; existing destination is never replaced.
    require(filesystem_identity(target.parent,'directory')==directory_identity,'Completion directory changed during link')
    require(sha(target)==expected['sha256'],'Published completion bytes differ');fsync_directory(target.parent)


def verify_published_files(manifest_path,completion_source):
    manifest=read(manifest_path);complete=read(completion_source)
    require(complete['manifest_sha256']==sha(manifest_path) and complete['publication_mode']==PUBLICATION_MODE
            and manifest['publication_mode']==PUBLICATION_MODE,'Completion or publication manifest changed')
    for value in manifest['files']:checked(value)


def report_text(data):
    selected=data['selected'];gate=selected['gate'];lines=[]
    for method,label in (('S_robust','the robust single-group policy'),('S_back3','the fixed backoff policy')):
        c=gate['comparisons'][method]
        lines.append(f"The two-group minus {label} DINO J difference was {c['mean']:+.6f}, with per-estimate differences "
                     +', '.join(f"{int(h)} dB: {v:+.6f}" for h,v in sorted(c['per_hat'].items(),key=lambda x:int(x[0])))+'.')
    engineering=' '.join(f"At estimated {x['snr_hat']} dB, the matched two-group minus nominal-single PSNR difference "
        f"was {x['matched_U_minus_S_nom_psnr_db']:+.6f} dB and the same two-group policy's 3 dB-overestimate PSNR loss "
        f"was {x['same_U_overestimate_psnr_loss_db']:+.6f} dB."
        for x in selected['separate_engineering_targets']['per_hat'])
    return ('R1 calibration200 delivery\n\n'
        'On the original fixed 200 calibration sources, each condition used three predetermined noise realizations. '
        'The completed experiment contains 32,400 physical frames and 43,200 logical receiver outputs. '
        'Each source was averaged over its three noises before the fixed objective '
        'J = 0.75 DINO(matched) + 0.25 DINO(transmitter overestimates SNR by 3 dB) was evaluated. '
        'The opposite 3 dB mismatch was retained as a pressure condition and did not affect selection.\n\n'
        +' '.join(lines)+f" The stored calibration gate is {gate['status']}. "
        'This gate controls the prespecified follow-up budget; it is not a publication or generalization claim. '
        'The separately reported PSNR engineering targets do not replace this gate. '+engineering+'\n\n'
        'All channel rejections, fixed-gray outputs, legal wrong-format receptions, negative results and exact-input reuse '
        'remain in the denominators. Conditional receive-state quality is descriptive and is not a source-paired causal effect. '
        'These are calibration results without confidence intervals; no bootstrap, inference or policy selection was repeated '
        'for delivery. ConvNeXt was not evaluated in R1. R2 is not registered or run by this entry.\n')


def tex_fragment(report):
    return '% R1 calibration results; no new statistical analysis.\n'+report.replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def publish(root,temporary,registration,data,closure):
    output=root/'results/layered_mismatch_raw_20261013/r1_delivery_v1';heavy=temporary/'r1_delivery_v1'
    require_absent(output);require_absent(heavy)
    join={x['view_id']:x for x in data['joins']};flat=[]
    for row in data['rows']:
        j=join[row['view_id']];frame=data['frames'][j['actual_RX']['sha256']];flat.append(flatten(row,j,frame))
    require(len(flat)==43200 and len({r['view_id'] for r in flat})==43200,'Flattened frame grid differs')
    winner_rows=[]
    for (method,hat),winner in sorted(data['winners'].items()):
        subset=[dict(r,method=method) for r in flat if r['snr_hat']==hat and r['candidate_id']==winner['candidate_id'] and r['receiver_rule']==winner['receiver_rule']]
        require(len(subset)==1800,'Frozen-method full frame grid differs');winner_rows.extend(subset)
    candidate_states=state_summary(flat,('snr_hat','snr_true','candidate_id','receiver_rule'))
    winner_states=state_summary(winner_rows,('method','snr_hat','snr_true'))
    physical=physical_rows(flat);require(len(physical)==32400,'Derived physical event grid differs')
    parameters=method_parameters(data,flat);report=report_text(data)
    # All scientific and derived arithmetic audits are now complete. Only here
    # may output staging directories be created. No prior files are overwritten.
    claim=temporary/'r1_delivery_v1.claim.json'
    write_json(claim,dict(registration=registration,process=process_identity(os.getpid()),created_unix=time.time(),
        purpose='Exclusive once-only publication claim after all scientific closure audits',automatic_retry=False,
        publication_mode=PUBLICATION_MODE))
    fsync_file(claim);claim_binding=dict(file=pin(claim),identity=filesystem_identity(claim,'file'))
    fsync_directory(temporary);require_absent(output);require_absent(heavy)
    token=str(os.getpid())+'_'+str(time.time_ns());draft=temporary/('r1_delivery_v1.preparing_'+token)
    draft.mkdir();light=draft/'light';payload=draft/'payload';light.mkdir();payload.mkdir()
    write_csv(payload/'physical_events.csv',physical);write_csv(payload/'logical_frames.csv',flat);write_csv(payload/'frozen_method_frames.csv',winner_rows)
    write_csv(light/'candidate_receive_states.csv',candidate_states);write_csv(light/'frozen_method_receive_states.csv',winner_states)
    write_csv(light/'frozen_method_parameters.csv',parameters)
    copies={}
    for key in ('source_means','source_means_csv','candidate_summary_csv','frozen_winner_summary_csv','selection'):
        source=checked(data['selection'][key]);copy_selected(source,light,payload);copies[key]=source.name
    (light/'REPORT.md').write_text(report,encoding='utf8',newline='\n')
    (light/'manuscript_results.tex').write_text(tex_fragment(report),encoding='utf8',newline='\n')
    write_json(light/'closure_audit.json',dict(status='PASS_R1_COMPLETED_CPU_DELIVERY',registration=registration,
        process_closure=closure,scientific_closure=data['audit'],scientific_calls=0,bootstrap_calls=0,reselection_calls=0,
        R2='NOT_REGISTERED_NOT_RUN',old_scientific_results_modified=False))
    manifest=[]
    for folder,destination in ((light,output),(payload,heavy)):
        for path in sorted(folder.iterdir()):
            require(path.is_file() and not path.is_symlink(),'Unexpected delivery artifact')
            if folder==light:require(path.stat().st_size<LIMIT,'Publication file exceeds unchanged size filter: '+path.name)
            manifest.append(dict(path=str(destination/path.name),sha256=sha(path),bytes=path.stat().st_size,
                                 storage='publication' if folder==light else 'personal_tmp'))
    write_json(light/'delivery_manifest.json',dict(registration=registration,files=manifest,copy_keys=copies,
        heavy_not_for_git=str(heavy),scientific_arrays_copied=False,standalone_recovery=False,
        publication_mode=PUBLICATION_MODE,directory_visibility_is_not_completion=True,
        completion_rule='Require actual delivery exit zero, no publication failure, valid final completion.json and its manifest SHA plus every artifact SHA; absent marker means incomplete',
        dependencies='Original registered R0/R1 inputs, model/environment recovery packages and complete raw R1 run remain required.'))
    write_json(light/'completion.json',dict(status='COMPLETE_R1_CPU_DELIVERY',registration=registration,
        manifest_sha256=sha(light/'delivery_manifest.json'),heavy_directory=str(heavy),scientific_calls=0,bootstrap_calls=0,
        reselection_calls=0,R2='NOT_REGISTERED_NOT_RUN',publication_mode=PUBLICATION_MODE,
        requires_actual_delivery_exit_zero=True,pending_receipt_name='completion.pending.json',
        publication_order='Heavy files first, light files second, exclusive pending receipt copy and same-directory no-clobber completion hardlink last'))
    phase='heavy_directory';heavy_complete=False;light_complete=False
    try:
        heavy_identity=publish_directory(payload,heavy,claim_binding);heavy_complete=True;phase='light_directory'
        light_identity=publish_directory(light,output,claim_binding,exclude=('completion.json',));light_complete=True;phase='final_artifact_verification'
        require(filesystem_identity(heavy,'directory')==heavy_identity and filesystem_identity(output,'directory')==light_identity,
                'Final reserved directory identity changed')
        verify_published_files(output/'delivery_manifest.json',light/'completion.json');phase='completion_marker'
        publish_completion(light/'completion.json',output/'completion.json',light_identity,claim_binding)
    except BaseException as error:
        if draft.exists():write_json(draft/'publication_failure.json',dict(error=repr(error),
            error_type=type(error).__name__,errno=getattr(error,'errno',None),phase=phase,
            heavy_directory_exists=heavy.exists(),light_directory_exists=output.exists(),
            heavy_files_complete=heavy_complete,light_files_complete=light_complete,
            completion_marker_exists=(output/'completion.json').exists(),automatic_retry=False,claim=str(claim),
            publication_mode=PUBLICATION_MODE,original_staging_preserved=str(draft)))
        raise
    return dict(status='DELIVERED_R1_CALIBRATION_ONLY',output=str(output),heavy=str(heavy),manifest=pin(output/'delivery_manifest.json'),
                stored_gate=data['selected']['gate']['status'],scientific_calls=0,bootstrap_calls=0,reselection_calls=0,
                publication_mode=PUBLICATION_MODE,original_staging_preserved=str(draft))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-registration',required=True)
    args=parser.parse_args();require(args.expected_registration==MASTER_SHA,'Explicit fixed master SHA required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run with CUDA hidden; no GPU work is permitted')
    registration=dict(path=str(ROOT/'results/layered_mismatch_raw_20261013/r1_execution_v1/registered/registration.json'),sha256=MASTER_SHA)
    require_absent(ROOT/'results/layered_mismatch_raw_20261013/r1_delivery_v1');require_absent(TMP/'r1_delivery_v1')
    guard=CPUReadOnly();require(not any(x in sys.modules for x in ('torch','tensorflow','sionna','lpips')),'Clean CPU interpreter required')
    sys.meta_path.insert(0,guard);sys.addaudithook(guard.audit)
    reg,closure=preflight(ROOT,TMP,registration);data=deep_audit(ROOT,TMP,registration,reg)
    preflight(ROOT,TMP,registration)  # Final live-state check immediately before creating output.
    guard.active=False
    try:
        result=publish(ROOT,TMP,registration,data,closure)
    except BaseException as error:
        failure=TMP/'r1_delivery_v1_publication_failure.json'
        if not failure.exists():write_json(failure,dict(registration=registration,error=repr(error),
            scientific_closure_passed=True,automatic_retry=False,all_original_science_untouched=True))
        raise
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
