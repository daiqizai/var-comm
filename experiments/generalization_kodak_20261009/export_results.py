"""Read completed Kodak24 JSON evidence and export compact resource tables.

No NPZ, source image, BPG candidate cache, codec, model, channel or statistics
routine is opened or executed. Existing inputs and scientific results are intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

METHODS = ('VAR_UNCONDITIONAL','P1024','SWIN80K','BPG_ADAPTIVE')
SNRS, SEEDS = (4,10,19), (2001,2002,2003)
DISPLAY = {'VAR_UNCONDITIONAL':'Partial-scale digital + unconditional VAR',
           'P1024':'Latent continuous JSCC', 'SWIN80K':'SwinJSCC-80k (adapted)',
           'BPG_ADAPTIVE':'Adaptive-resolution BPG + LDPC'}


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Reader:
    def __init__(self):
        self.bindings = {}

    def read(self,path,expected=None):
        path = Path(path).resolve()
        require(path.suffix == '.json','This export reads JSON metadata only')
        actual = sha(path)
        require(expected is None or actual == expected,'Changed input: '+str(path))
        self.bindings[str(path)] = actual
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def pin(self,d):
        return self.read(d['path'],d['sha256'])


def write_json(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')


def csv_write(path,rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',encoding='utf8',newline='') as f:
        writer = csv.DictWriter(f,fields,lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def copy_json(reader,source,target,role):
    value = reader.read(source)
    target = Path(target); require(not target.exists(),'Never overwrite an exported receipt')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    require(sha(target) == reader.bindings[str(Path(source).resolve())],'Copied evidence changed')
    return dict(path=str(target.resolve()),sha256=sha(target),original_path=str(Path(source).resolve()),role=role)


def frame_key(row):
    return row['source_index'],row['snr_db'],row['noise_seed']


def check_table(table,method,records):
    expected = {(i,s,n) for i in range(24) for s in SNRS for n in SEEDS}
    require(len(table) == 216 and {frame_key(r) for r in table} == expected,'Complete 216-frame method grid')
    require(all(r['method'] == method and r['source_id'] == records[r['source_index']]['source_id']
                and r['status'] and r['reconstruction']['key'] == 'rgb' for r in table), 'Method/source/frame contract')


def common_row(row):
    return dict(method=row['method'],method_label=DISPLAY[row['method']],source_index=row['source_index'],
        source_id=row['source_id'],snr_db=row['snr_db'],noise_seed=row['noise_seed'],N=1024,status=row['status'],
        reconstruction_path=row['reconstruction']['path'],reconstruction_archive_sha256=row['reconstruction']['sha256'],
        diagnostics_role='post-reception audit only; never receiver input',true_class_labels_used=False)


def neural_row(reader,row,done):
    require(done['outputs'].get(row['reception']['path']) == row['reception']['sha256'], 'Completion-bound RX JSON')
    require(done['outputs'].get(row['reconstruction']['path']) == row['reconstruction']['sha256'], 'Completion-bound RGB descriptor')
    receipt = reader.pin(row['reception'])
    result = {**common_row(row),'gray':row['gray'],'packet_decoder_calls':row['packet_decode_count'],**row['resource']}
    result['reception_json_sha256'] = row['reception']['sha256']
    if row['method'] == 'VAR_UNCONDITIONAL':
        actual,tx = receipt['actual_reception'],receipt['transmitter']
        result.update(profile_id=tx['point']['profile_id'],candidate_id=tx['point']['candidate_id'],wire_key=tx['point']['wire_key'],
            public_frame_counter=actual['public_frame_counter'],header_accepted=actual['header']['header_ok'],
            received_header_profile_id=actual['header'].get('profile_id'),header_correct=receipt['header_correct'],
            body_crc_accepted=actual['body_crc_accept'],receiver_state_kind=receipt['receiver_state']['kind'],
            received_tokens_equal_source_prefix=receipt['received_tokens_equal_source_prefix'],
            crc_accepted_incorrect_information=receipt['crc_accepted_incorrect_information'],
            VAR_null_embedding_index=receipt['null_embedding_index'],header_energy_scope='actual saved waveform reduction',
            body_energy_scope='actual saved waveform reduction')
        require(receipt['true_class_supplied'] is False and receipt['audit_comparison_is_after_reconstruction'] is True,
                'No source-class or diagnostic feedback into reconstruction')
    elif row['method'] == 'SWIN80K':
        context = receipt['actual_context']
        result.update(header_accepted=context['accepted'],header_crc_accepted=context['crc_accepted'],
            header_fields_legal=context['fields_legal'],header_correct=receipt['received_context_matches_transmitted'],
            received_normalization_power=context['power'],transmitted_normalization_power=receipt['transmitter']['power'],
            received_active_channels_json=json.dumps(context['indices']),
            outside_training_and_calibration_range=receipt['outside_training_and_calibration_range'])
        require(receipt['outside_training_and_calibration_range'] == (row['snr_db'] == 19),'Swin 19 dB range annotation')
    return result


def bpg_row(row,full):
    require(frame_key(row) == frame_key(full) and row['source_id'] == full['source_id']
            and row['status'] == full['link_status'], 'BPG compact/source checkpoint pairing')
    require(row['reconstruction']['path'] == full['float_reconstruction']['path']
            and row['reconstruction']['sha256'] == full['float_reconstruction']['sha256'], 'BPG RGB descriptor pairing')
    head,body = full['header'] or {},full['body'] or {}
    result = dict(**common_row(row),gray=full['gray_substitution'],packet_decoder_calls=full['packet_decoder_calls'],
        source_encoding_fit=full['source_encoding_fit'],source_unfit=full['source_unfit'],
        actual_link_executed=full['actual_link_executed'],profile_id=full['profile_id'],
        Qm=full['q'],nominal_rate=full['rate'],k=full['k'],n=full['n'],
        effective_rate_fraction=f"{full['k']}/{full['n']}",capacity_bytes=full['capacity_bytes'],
        header_complex_uses=full['header_symbols'],body_complex_uses=full['body_symbols'],idle_complex_uses=full['padding_symbols'],
        selected_resolution=full['selected_resolution'],selected_qp=full['selected_qp'],complete_BPG_bytes=full['complete_BPG_bytes'],
        transmitted_complete_BPG_sha256=full['transmitted_complete_BPG_sha256'],
        received_complete_BPG_sha256=full['received_complete_BPG_sha256'],
        undetected_payload_difference=full['undetected_payload_difference'],actual_frame_energy=full['actual_frame_energy'],
        header_accepted=head.get('header_ok'),received_header_profile_id=head.get('profile_id'),
        body_crc_accepted=body.get('crc_accepted'),body_parser_accepted=body.get('parser_accepted'),
        public_frame_counter=full['counter'],header_state_json=json.dumps(head,sort_keys=True,separators=(',',':')),
        body_state_json=json.dumps({k:v for k,v in body.items() if k not in ('hard_payload','decoded_bits','payload_b64')},
            sort_keys=True,separators=(',',':')),
        header_energy_scope='not separately saved in the original per-frame BPG JSON; no waveform reopened',
        body_energy_scope='not separately saved in the original per-frame BPG JSON; total energy retained')
    require(full['source_truth_used_by_receiver'] is False and full['policy_selection'] is False,
            'BPG diagnostics cannot select RX or policy')
    return result


def source_config(rows):
    keys = ('profile_id','Qm','nominal_rate','k','n','effective_rate_fraction','capacity_bytes',
            'source_encoding_fit','source_unfit','selected_resolution','selected_qp','complete_BPG_bytes',
            'transmitted_complete_BPG_sha256','header_complex_uses','body_complex_uses','idle_complex_uses')
    result = []
    for i in range(24):
        for s in SNRS:
            group = [r for r in rows if r['method'] == 'BPG_ADAPTIVE' and r['source_index'] == i and r['snr_db'] == s]
            require(len(group) == 3 and all(tuple(r[k] for k in keys) == tuple(group[0][k] for k in keys) for r in group),
                    'One TX source encoding configuration, unchanged across all three noises')
            result.append(dict(method='BPG_ADAPTIVE',source_index=i,source_id=group[0]['source_id'],snr_db=s,
                **{k:group[0][k] for k in keys},noise_count=3,
                selection_role='Frozen TX source coding rule; no selection on received/noisy quality'))
    return result


def frozen_configs(neural,bpg,reader,frames):
    catalogue = reader.pin(neural['frozen_raw_catalogue'])['profiles']
    original = reader.pin(bpg['original_adaptive_request'])
    a3 = reader.pin(neural['original_a3_request']); baseline = reader.pin(a3['baseline_request'])
    result = []
    for method in METHODS:
        for s in SNRS:
            row = dict(method=method,method_label=DISPLAY[method],snr_db=s,N=1024,
                       selected_on_Kodak=False,training_updates=0,true_class_labels_used=False)
            if method == 'VAR_UNCONDITIONAL':
                point = neural['raw_points'][str(s)]; p = catalogue[point['profile_id']]; g = p['groups'][0]
                row.update(profile_id=p['profile_id'],candidate_id=point['candidate_id'],wire_key=p['wire_key'],
                    m=p['m'],K=p['K'],source_token_count=p['token_count'],modulation=g['modulation'],
                    nominal_rate=g['nominal_rate'],k=g['layout']['k'],n=g['layout']['n'],
                    effective_rate=g['layout']['actual_effective_rate'],header_complex_uses=p['header_symbols'],
                    body_complex_uses=p['used_body_symbols'],idle_complex_uses=p['idle_symbols'],VAR_class='null embedding 1000',
                    policy_sha256=neural['frozen_raw_policy']['sha256'])
            elif method == 'BPG_ADAPTIVE':
                pid = bpg['frozen_selected_profile_ids'][str(s)]
                p = next(x for x in original['catalogue'] if x['profile_id'] == pid)
                row.update(profile_id=pid,Qm=p['q'],nominal_rate=p['rate'],k=p['k'],n=p['n'],capacity_bytes=p['capacity_bytes'],
                    effective_rate_fraction=f"{p['k']}/{p['n']}",header_complex_uses=p['header_symbols'],
                    body_complex_uses=p['body_symbols'],idle_complex_uses=0,
                    source_configuration='Frozen adaptive per-source resolution/QP; see 72-row table',
                    policy_sha256=bpg['original_adaptive_freeze']['sha256'])
            elif method == 'SWIN80K':
                saved = next(f for f in frames if f['method'] == method and f['snr_db'] == s)
                row.update(selected_step=baseline['Swin_identity']['selected_step'],
                    checkpoint_sha256=baseline['Swin_identity']['selected_checkpoint_sha256'],
                    body_channels=saved['body_channels'],header_complex_uses=saved['header_complex_uses'],
                    body_complex_uses=saved['body_complex_uses'],idle_complex_uses=saved['idle_complex_uses'],
                    outside_training_and_calibration_range=saved['outside_training_and_calibration_range'],
                    adaptation='Paid normalization-power/mask header; training/calibration SNR 1 through 13 dB')
            else:
                saved = next(f for f in frames if f['method'] == method and f['snr_db'] == s)
                selected = baseline['P_identity']['original_selected_record']
                row.update(selected_step=selected['step'],checkpoint_sha256=selected['checkpoint_sha256'],
                    header_complex_uses=saved['header_complex_uses'],body_complex_uses=saved['body_complex_uses'],
                    idle_complex_uses=saved['idle_complex_uses'],
                    source_configuration='Frozen continuous latent JSCC with the same frozen Dc')
            result.append(row)
    return result


def run(root,out):
    root,out = Path(root).resolve(),Path(out).resolve()
    runtime = root/'outputs/GENERALIZATION-KODAK-20261009'
    if root.name == 'GENERALIZATION-KODAK-20261009':
        runtime = root
    require(runtime.is_dir() and out != runtime and not out.is_relative_to(runtime), 'Independent publication output required')
    names = ('frame_resources.csv','bpg_source_configs.csv','frozen_configs.csv','RESOURCE_EXPORT.md',
             'resource_export_completion.json','evidence')
    require(all(not (out/name).exists() for name in names),'Existing resource export retained; choose a new destination')
    reader = Reader(); material = runtime/'materials'
    neural = reader.read(material/'neural_request_r2.json'); bpg = reader.read(material/'bpg_request.json')
    manifest = reader.pin(neural['source_manifest'])
    require(bpg['source_manifest'] == neural['source_manifest'] and len(manifest['records']) == 24,'Same frozen Kodak24 dataset')
    records = manifest['records']; tables = {}; completions = {}; rows = []; copy_plan = []
    for method in METHODS:
        method_out = runtime/'BPG_ADAPTIVE' if method == 'BPG_ADAPTIVE' else runtime/'neural_v1'/method
        cp_path,frame_path = method_out/'completion.json',method_out/'frames.json'
        done = reader.read(cp_path); table = reader.read(frame_path,done['outputs'].get(str(frame_path)))
        require(done['outputs'].get(str(frame_path)) == reader.bindings[str(frame_path)],'Actual completion binds frame index')
        require(done['source_count'] == 24 and done.get('frame_count',done.get('frames')) == 216,'Actual full method completion')
        if method == 'BPG_ADAPTIVE':
            require(done['status'] == 'KODAK24_ADAPTIVE_BPG_ACTUAL_CHILDREN_WAIT_ZERO'
                    and done['actual_children_waited'] and done['worker_exit_codes'] == [0,0]
                    and done['root_budget_before'] == done['root_budget_after'], 'BPG children/root ledger closure')
            require(done['request_sha256'] == sha(material/'bpg_request.json'),'BPG executed request')
            full = {}
            for i in range(24):
                path = method_out/'sources'/f'{i:04d}.json'
                require(str(path) in done['outputs'],'Completion must bind every source checkpoint')
                source = reader.read(path,done['outputs'][str(path)])
                require(source['source_index'] == i and len(source['rows']) == 9,'Nine actual BPG frames per source')
                for row in source['rows']:
                    require(frame_key(row) not in full,'Duplicate BPG source checkpoint row')
                    full[frame_key(row)] = row
            check_table(table,method,records)
            rows += [bpg_row(row,full[frame_key(row)]) for row in table]
        else:
            require(done['status'] == 'COMPLETE' and done['method'] == method
                    and done['request']['sha256'] == sha(material/'neural_request_r2.json'),'Actual neural/request closure')
            check_table(table,method,records)
            rows += [neural_row(reader,row,done) for row in table]
        tables[method],completions[method] = table,done
        copy_plan.extend([(cp_path,Path('methods')/method/'completion.json','ACTUAL_COMPLETE'),
                          (frame_path,Path('methods')/method/'frames.json','ACTUAL_216_FRAMES')])
    require(len(rows) == 864 and sum(r['packet_decoder_calls'] for r in rows) <= 1080,'864 frames within independent PHY budget')
    for method in METHODS:
        expected_calls = (completions[method]['independent_ledger']['total'] if method == 'BPG_ADAPTIVE'
                          else completions[method]['packet_attempts'])
        require(sum(r['packet_decoder_calls'] for r in rows if r['method'] == method) == expected_calls,'Actual per-method decode counts close')
    frozen = frozen_configs(neural,bpg,reader,rows); configs = source_config(rows)
    require(len(frozen) == 12 and len(configs) == 72,'Three SNRs times four methods and 24-source BPG configurations')
    for job in ('bpg','neural','score'):
        directory = runtime/(job+'_supervision_v1')
        completion = directory/'completion.json'
        if not completion.exists():
            require(job == 'score','Actual reconstruction supervision must already be complete')
            continue
        closed = reader.read(completion)
        require(closed['status'] == 'FINITE_JOB_ACTUAL_WAIT_ZERO' and closed['children']
                and all(c['exit_code'] == 0 for c in closed['children']), 'Actual child wait-zero evidence')
        require(closed['request_sha256'] == sha(material/(job+'_job.json')),'Supervisor request binding')
        for name in ('start.json','completion.json'):
            copy_plan.append((directory/name,Path('supervision')/(job+'_supervision_v1')/name,'ACTUAL_SUPERVISION'))
        for child in closed['children']:
            for tail in ('launch','wait'):
                name = f"step_{child['index']:02d}_{tail}.json"
                receipt = reader.read(directory/name)
                require(receipt['pid'] == child['pid'] and (tail != 'wait' or receipt['exit_code'] == 0),'Child wait/launch identity')
                copy_plan.append((directory/name,Path('supervision')/(job+'_supervision_v1')/name,'ACTUAL_CHILD_'+tail.upper()))
    for name in ('bpg_request.json','bpg_job.json','neural_request_r2.json','neural_job.json',
                 'score_request.json','score_job.json','no_inherited_kodak_codec_cache.json'):
        path = material/name
        if path.exists():
            copy_plan.append((path,Path('materials')/name,'AUTHORITATIVE_EXECUTION_INPUT'))
    for name in ('bpg_launch.json','neural_launch.json','score_launch.json'):
        path = runtime/name
        if path.exists():
            copy_plan.append((path,Path('supervision')/name,'LAUNCH_METADATA_NOT_COMPLETION'))
    copy_plan.append((runtime/'data/dataset_manifest.json',Path('dataset_manifest.json'),'FIXED_DATASET_MANIFEST'))
    out.mkdir(parents=True,exist_ok=True); copied = []
    for source,target,role in copy_plan:
        copied.append(copy_json(reader,source,out/'evidence'/target,role))
    write_json(out/'evidence/index.json',dict(files=copied,large_binary_caches_copied=False,
        original_server_paths_preserved=True,prepared_neural_request_r1_omitted=True))
    csv_write(out/'frame_resources.csv',rows); csv_write(out/'bpg_source_configs.csv',configs); csv_write(out/'frozen_configs.csv',frozen)
    note = '''# Kodak24 compact resource export

This is a read-only export of actual completed reconstruction JSON records: 864 frames (four methods, 24 sources, three SNRs, three noises), 72 BPG source/SNR configurations and 12 frozen method/SNR configurations. Failure states are retained. No model, channel, codec, new statistical test or resampling is executed.

`frame_resources.csv` preserves actual frame status, paid channel uses, code parameters, saved total energy and receiver diagnostics. Empty fields mean not applicable or not separately recorded, never zero failure. In particular BPG header/body energy was not separately saved in its per-frame JSON; the export retains measured total energy without reopening waveforms or inventing a split. Neural component energies are the values already saved by the inference job. Code-rate fractions use the recorded k and n, not the nominal rate label.

`bpg_source_configs.csv` contains each selected complete-stream size, resolution, QP and hash once per source/SNR, with consistency checked across all three noises. It retains the previously frozen adaptive source-encoding rule; Kodak does not select a new MCS. This is distinct from the earlier fixed native-resolution BPG baseline.

`frozen_configs.csv` covers all four methods at 4, 10 and 19 dB. VAR uses the null embedding, with no real class input. Swin's 19 dB point remains outside its training/calibration range. Post-reception correctness checks are audit diagnostics and never receiver inputs.

`evidence/` contains byte-identical copies of all four actual frame indexes and completion receipts, the executed neural request r2 and other metadata requests, the common dataset manifest, and available completed supervisory child-wait receipts. A launch record alone is not completion. If score supervision was not yet complete when this export ran, it is omitted; this resource export does not claim that scoring or plots completed. The earlier prepared but unused neural request r1 is omitted.

This compact delivery does not include NPZ reconstructions, source PNGs, model checkpoints, all BPG search candidates or an executable migration environment. Original server paths and hashes remain in the receipts. The exporter authenticates the JSON consumed and checks completion-bound image descriptors; it does not reopen or rehash large binary caches. It must not be described as a complete server migration.

Reproduce into a fresh export destination: `python experiments/generalization_kodak_20261009/export_results.py --root /home/liulu/projects/VAR_COMM --out NEW_EXPORT_DIRECTORY`.
'''
    with (out/'RESOURCE_EXPORT.md').open('x',encoding='utf8',newline='\n') as f:
        f.write(note)
    require(all(sha(p) == h for p,h in reader.bindings.items()),'Input JSON changed during read-only export')
    outputs = {str(out/name):sha(out/name) for name in names[:4]}
    outputs.update({str(p):sha(p) for p in (out/'evidence').rglob('*.json')})
    done = dict(status='KODAK24_COMPACT_RESOURCES_EXPORTED_V1',source_count=24,noise_count=3,
        frame_count=864,source_configuration_rows=72,frozen_configuration_rows=12,
        actual_packet_calls=sum(r['packet_decoder_calls'] for r in rows),
        frames_by_method={m:216 for m in METHODS},gray_frames_by_method={m:sum(bool(r['gray']) for r in rows if r['method']==m) for m in METHODS},
        input_json_bindings=reader.bindings,outputs=outputs,script_sha256=sha(__file__),
        no_NPZ_or_candidate_streams_read=True,new_model_calls=0,new_PHY_calls=0,new_channel_simulations=0,
        new_bootstrap=0,old_scientific_results_modified=False,complete_server_migration_claimed=False)
    write_json(out/'resource_export_completion.json',done)
    print(json.dumps(dict(status=done['status'],frames=864,source_configurations=72,frozen_configurations=12,
                         outputs=len(outputs),actual_packet_calls=done['actual_packet_calls'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path); parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args(); run(args.root,args.out)
