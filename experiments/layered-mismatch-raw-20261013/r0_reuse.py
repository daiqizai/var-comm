"""Read-only R0 reuse and pixel/token inventory. Never imports a model or PHY."""
from __future__ import annotations
import argparse, collections, csv, hashlib, importlib.util, json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
RT = R.parent.parent / 'var_comm_runtime_20261010'
T = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
E = R / 'experiments/layered-mismatch-raw-20261013'
O = R / 'results/layered_mismatch_raw_20261013/r0_reuse'
EV = T / 'layered_mismatch_raw_20261013/evidence_seed_v1'
DR = T / 'layered_mismatch_raw_20261013/dev100_restore_v1'
DM = DR / 'VAR_COMM/outputs/CONTENT-REAL-64QAM-20261006/H/development100_assets/manifest.json'
CM = RT / 'assets/fixed_existing32_seed_v1/VAR_COMM/outputs/CONTENT-REAL-64QAM-20261006/H/full1000_assets/manifest.json'
CR = RT / 'qualification/h800_ep_full_source900_v1_attempt1/registered/request.json'
RR = RT / 'qualification/h800_mismatch_phy_v1_attempt1/registered/request.json'


def require(v, message):
    if not v:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pin(path, expected=None):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Missing regular input: ' + str(p))
    d = dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)
    require(expected is None or d['sha256'] == expected, 'SHA mismatch: ' + str(p))
    return d


def read(path, expected=None):
    p = pin(path, expected)
    return json.loads(Path(path).read_text()), p


def dump(name, data):
    p = O / name
    require(not p.exists(), 'Refusing overwrite: ' + str(p))
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return pin(p)


def validate_array(record):
    a = pin(record['archive']['path'], record['archive']['sha256'])
    c = pin(record['checkpoint']['path'], record['checkpoint']['sha256'])
    cp = json.loads(Path(c['path']).read_text())
    for key in ('source_index', 'source_id', 'preprocessing_id', 'tokens_sha256'):
        require(cp[key] == record[key], 'Checkpoint identity mismatch: ' + key)
    with np.load(a['path'], allow_pickle=False) as z:
        require(set(z.files) == {'tokens', 'pixels'}, 'Unexpected source archive fields')
        pixels, tokens = z['pixels'], z['tokens']
        require(pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256), 'Pixel format changed')
        require(tokens.dtype == np.int64 and tokens.shape == (680,), 'Token format changed')
        require(int(tokens.min()) >= 0 and int(tokens.max()) < 4096, 'Token outside fixed12 codebook')
        require(hashlib.sha256(pixels.tobytes(order='C')).hexdigest() == record['preprocessing_id'], 'Pixel SHA mismatch')
        require(hashlib.sha256(b'int64:680\0' + tokens.astype('<i8').tobytes()).hexdigest() == record['tokens_sha256'], 'Token SHA mismatch')
    return dict(record, archive=a, checkpoint=c,
                original_archive_path=cp['archive'], original_checkpoint_path=record.get('original_checkpoint_path'),
                array_verification=dict(pixels_shape=[3, 256, 256], pixels_dtype='uint8', tokens_shape=[680], tokens_dtype='int64', exact_original_bytes=True))


def selected_model_identity(identity):
    return {k: identity['models'][k] for k in ('vae', 'var', 'decoder')}


def run():
    require(not O.exists(), 'R0 output exists; inspect completion and do not rerun')
    creq, crpin = read(CR, '39bc1c524450ecd981a5c2dafdeb5c898bde06b2723e24f08da3efdf711354ca')
    cm, cmpin = read(CM, '49bb6b8dfb45ac9fec5968481fe7f1691959a80964972bad1c0bde4e468e3cae')
    dm, dmpin = read(DM, 'f12a583c7eae0cc613343c3c981a5264df6a0669797b978f6192140e28550793')
    hydration, hydration_pin = read(DR/'hydration_completion.json', '386c60a5d0137d4558de97872e699d42c023526e00afe763f6b63a0097ed343e')
    for r in hydration['mapping']:
        pin(r['path'], r['sha256'])
    require(hydration['file_count'] == 202 and hydration['source_count'] == 100, 'Incomplete development restore')
    original_map = {r['original_path']: r for r in hydration['mapping']}
    inventory, evidence_pin = read(EV/'inventory.json')
    for r in inventory['files']:
        pin(EV/r['member'], r['sha256'])
    cal = [validate_array(r) for r in creq['records']]
    require(len(cal) == 1000 and [r['source_index'] for r in cal] == list(range(1000)), 'Original cal1000 grid changed')
    require([r['source_id'] for r in cal] == cm['source_ids'], 'Original calibration order changed')
    dev = []
    src_manifest, src_manifest_pin = read(EV/'development100_source/manifest.json')
    src_complete, src_complete_pin = read(EV/'development100_source/completion.json')
    require(src_manifest['encoder_tokens_verified'] is True and src_manifest['source_codec_complete'] is True, 'Historical development source encoding incomplete')
    for old in dm['records']:
        r = {k: old[k] for k in ('source_index', 'source_id', 'preprocessing_id', 'tokens_sha256', 'evaluation_class_index')}
        r['archive'] = {k: original_map[old['archive']][k] for k in ('path','sha256','bytes')}
        r['checkpoint'] = {k: original_map[old['checkpoint']][k] for k in ('path','sha256','bytes')}
        r['original_checkpoint_path'] = old['checkpoint']
        v = validate_array(r)
        sc = src_manifest['records'][old['source_index']]
        p = EV/'development100_source/source_checkpoints'/Path(sc['checkpoint']).name
        cp, cppin = read(p, sc['sha256'])
        for key in ('source_index', 'source_id', 'preprocessing_id', 'tokens_sha256'):
            require(cp[key] == old[key], 'Historical fresh encoder identity mismatch')
        require(cp['encoder_tokens_verified'] is True and cp['independent_roundtrip'] is True, 'Historical fresh encoder proof missing')
        require(cp['source_assets_checkpoint']['sha256'] == r['checkpoint']['sha256'], 'Follow-up encoder receipt does not bind asset checkpoint')
        v['historical_frozen_encoder_verification'] = dict(checkpoint=cppin, original_path=sc['checkpoint'], verified=True, new_call_this_audit=False)
        dev.append(v)
    require(len(dev) == 100 and [r['source_index'] for r in dev] == list(range(100)), 'Original dev100 grid changed')
    require([r['source_id'] for r in dev] == dm['source_ids'] == src_manifest['source_ids'] == src_complete['source_ids'], 'Original development order changed')
    require(not ({r['source_id'] for r in cal} & {r['source_id'] for r in dev}), 'Cal/development overlap')
    require(not ({r['preprocessing_id'] for r in cal} & {r['preprocessing_id'] for r in dev}), 'Cal/development pixel overlap')
    spec = importlib.util.spec_from_file_location('r0_frozen_contract', E/'contract.py')
    contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
    chosen = [dict(r, selection_slot=i) for i,r in enumerate(contract.select_cal200(cal))]
    require(len(chosen) == 200, 'Wrong selected population')
    original_ids = {r['source_id']: i for i,r in enumerate(cal)}
    require(all(r['source_index'] == original_ids[r['source_id']] for r in chosen), 'Source index conflated with selection slot')
    old16 = RT/'assets/timing_fixed16_seed_v1/VAR_COMM/outputs/CONTENT-REAL-64QAM-20261006/H/development100_assets/sources'
    old16_indices = sorted(int(p.stem) for p in old16.glob('*.npz'))
    for i in old16_indices:
        pin(old16/f'{i:04d}.npz', dev[i]['archive']['sha256'])
    require(old16_indices == sorted([0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]), 'Old fixed16 seed changed')
    q, qp = read(EV/'old_uep/quality_bundle.json','58604ab3d1b60d890cb3a0cc91c2eafad03e9b1a228f694a6f753915d22e52c5')
    qr, qrp = read(EV/'old_uep/quality_registration.json','bb7f8d7d08616a1b634885ceb5663a4aba1d570460f222b5ac9237ea30a28e2a')
    qc, qcp = read(EV/'old_uep/quality_completion.json','df5752bc804aa228ad19445fba308d6b3b60fada6d23bce8c04d8cef6919f648')
    require(q['registration_sha256']==qrp['sha256'] and q['completion_sha256']==qcp['sha256'], 'Old Q chain changed')
    require(qr['source_ids'] == cm['source_ids'] == q['source_ids'] == qc['source_ids'], 'Old Q calibration order differs')
    require(qr['preprocessing_ids'] == [r['preprocessing_id'] for r in cal], 'Old Q source pixels differ')
    require(qr['metric'] == 'dinov2_vitl14_cosine' and not qr['label_conditioned'] and not qr['development_read'], 'Old Q wrong metric or conditioning')
    require(selected_model_identity(qr['renderer_identity']['frozen_identity']) == selected_model_identity(cm['visual_identity']), 'Old Q frozen model states changed')
    require(selected_model_identity(dm['visual_identity']) == selected_model_identity(cm['visual_identity']), 'Development model states differ')
    require(qr['renderer_identity']['frozen_identity']['decoder_gate_sha256'] == cm['visual_identity']['decoder_gate_sha256'], 'Dc gate changed')
    coarse = {s['state_id'] for s in qr['states'] if s['K']==0}
    ids = {r['source_id'] for r in chosen}
    subset = [r for r in q['rows'] if r['source_id'] in ids and (r['receiver']=='gray' or (r['receiver']=='VAR' and r['state_id'] in coarse))]
    require(len(subset)==1200 and len({(r['source_id'],r['state_id'],r['receiver']) for r in subset})==1200, 'Historical K0/gray Q grid incomplete')
    metric = qr['score_identity']['model_metadata']['metrics']['dinov2_vitl14']
    rr, rrp = read(RR, '9e587fcec3be465d8d6c43a7123400635441918befbbd45a6339f0f208af2faf')
    bindings = rr['raw_runtime_binding']['files']
    for b in bindings: pin(b['path'],b['sha256'])
    assembly_row = next(b for b in bindings if b['path'].endswith('assemble_v1/completion.json'))
    assembly = json.loads(Path(assembly_row['path']).read_text())
    require(len(assembly['profiles']) == 433 and all(p['order']=='raster' and p['G']==1 for p in assembly['profiles']), 'Raw433 definition changed')
    pp = RT/'assets/mismatch_raw100_seed_v1/VAR_COMM/outputs/MAIN-RAW64-20261007/final_selection_v1/policies.json'
    pol, ppin = read(pp,'7871ac7f9c619bd21c63ba31157738cd56c0dda9344c993e99b935f60ce70c0c')
    anchors=[]
    for hat in (7,10):
        for arm,snr in [('S_nom',hat),('S_back3',hat-3)]:
            w=next(w for w in pol['winners'] if w['family']=='PARTIAL' and w['snr_db']==snr)
            p=next(p for p in assembly['profiles'] if p['candidate_id']==w['candidate_id'])
            anchors.append(dict(arm=arm,snr_hat=hat,lookup_snr_db=snr,candidate_id=w['candidate_id'],m=p['m'],K=p['K'],profile_id=p['profile_id'],order=p['order'],receiver='KEEP',groups=p['groups']))
    bler, bp = read(EV/'old_uep/bler.json','e7e316ba722e2315261b4ba3c75aa0b4d33973fab09ad50978fe4b46909d4f52')
    coverage=collections.Counter((r['kind'],r['snr_db'],r.get('physical_configuration',{}).get('q')) for r in bler['rows'])
    candidates, candidate_pin=read(R/'results/layered_mismatch_raw_20261013/candidates_cpu_v1/catalogue.json')
    endpoints=sorted({(int(p['m']),int(p['K'])) for p in candidates['profiles']})
    u_endpoints=sorted({(int(p['m']),int(p['K'])) for p in candidates['profiles'] if p['G']==2} | {(int(p['j']),0) for p in candidates['profiles'] if p['G']==2})
    historical={(int(s['m']),0) for s in qr['states'] if s['K']==0}
    missing=sorted(set(endpoints)-historical)
    hist=[]
    for name,kind,status in [
      ('reports/parallel_v2_20261005.md','unknown_instantaneous_CSI_Rayleigh_probability_model','F1_UNCERTAIN_F2_NOT_STARTED'),
      ('reports/content_feedback_probe_20261005.md','Rayleigh_feedback_working_model','V_TERMINAL_MISSING_ASSET'),
      ('reports/two_stage_feedback_model_20261005.md','two_stage_feedback_table_integration','NO_NEW_REAL_CHANNEL_RECONSTRUCTIONS'),
      ('reports/tx_csi_uncertainty_literature_preparation_20260916.md','literature_preparation','NO_CSI_EXPERIMENT_IN_THIS_DOCUMENT')]:
        hist.append(dict(binding=pin(R/name),kind=kind,status=status,reuse='historical_reference_only_not_new_AWGN_frames'))
    O.mkdir(parents=True)
    outputs=[]
    outputs.append(dump('calibration1000_sources.json',dict(population='original_calibration1000',records=cal,input=crpin,source_manifest=cmpin)))
    outputs.append(dump('cal200_sources.json',dict(population='original_calibration_SHA200_UEP_MISMATCH_CAL200_V1',records=chosen,selection_domain=contract.CAL_DOMAIN,selection_rule='sort SHA256(canonical JSON [domain, exact source_id]), then exact source_id; first200',contract=pin(E/'contract.py'),uses_quality=False)))
    outputs.append(dump('development100_sources.json',dict(population='original_development100_reused_not_new_blind',records=dev,source_manifest=dmpin,restoration=hydration_pin,historical_fresh_encoder_manifest=src_manifest_pin,historical_fresh_encoder_completion=src_complete_pin)))
    qout=O/'historical_clean_coarse_cal200.csv'
    with qout.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source_id','state_id','receiver','dinov2_vitl14_cosine']);writer.writeheader();writer.writerows(subset)
    outputs.append(pin(qout))
    quality_audit=dict(status='K0_AND_GRAY_HISTORICAL_PRESCREEN_ONLY',bundle=qp,registration=qrp,completion=qcp,
      original_source_count=1000,original_rows=len(q['rows']),all_source_id_and_pixel_identity_match=True,
      frozen_model_states=selected_model_identity(cm['visual_identity']),decoder_gate_sha256=cm['visual_identity']['decoder_gate_sha256'],
      original_runtime=qr['renderer_identity']['numerical_runtime'],generation=qr['renderer_identity']['generation'],
      metric_name=qr['metric'],metric_definition=metric,metric_evaluator_identity=qr['score_identity']['metric_evaluator_identity'],
      reusable_whole_states=sorted(coarse),gray_per_source_reusable=True,reused_for_new_selection=False,
      excluded_positive_K_reason='Old positive-K states use M1 entropy-ranked positions, current protocol uses raster. Same K is not the same received token set.',
      old_Q_source_implementation_sha256=qr['source_sha256'],
      renderer_source=pin(R/'experiments/prior-aware-grouped-mcs-v1/source_quality.py'),
      host_caveat='Historical RTX4090D clean values are allowed only as a disclosed pre-screen reference; no claim of new H800 numerical equivalence or actual R1 quality.',
      candidate_catalogue=candidate_pin,all_unique_terminal_states=[dict(m=m,K=k) for m,k in endpoints],
      all_states_missing_from_compatible_old_UEP_Q=[dict(m=m,K=k) for m,k in missing],
      missing_source_endpoint_pairs_before_other_raw_cache_admission=len(missing)*200,
      U_EEP_states_including_coarse=[dict(m=m,K=k) for m,k in u_endpoints],
      U_EEP_missing_source_endpoint_pairs_before_other_raw_cache_admission=len(set(u_endpoints)-historical)*200,
      other_raw_actualcal_cache='Separate targeted provenance audit required; no positive-K entry admitted in this file.')
    outputs.append(dump('clean_quality_reuse.json',quality_audit))
    outputs.append(dump('raw433_anchors.json',dict(anchors=anchors,policies=ppin,raw_request=rrp,catalogue_sha256=rr['raw_runtime_binding']['catalogue_sha256'],aliases_sha256=rr['raw_runtime_binding']['aliases_sha256'],raw_bindings=bindings)))
    outputs.append(dump('historical_branches.json',dict(branches=hist,old_mismatch=dict(population='posthoc_first100_subset_of_original_common500_holdout_NOT_development100',request=rrp,noise_seeds=[6201,6202,6203],counter='(original_6_SNR_index*500+source_index)*3+noise_index',reuse='frozen models/configurations/API only; no old frame is admitted under new source/noise contract'),new_noise=dict(namespace=contract.NAMESPACE,cal_seeds=list(contract.CAL_SEEDS),dev_seeds=list(contract.DEV_SEEDS),key='canonicalJSON([namespace,split,exact_source_id,hat,seed]); full SHA256 big-endian PCG64; trueSNR and configuration absent',counter='(hat_index*source_count+selection_slot)*3+noise_index'))))
    outputs.append(dump('bler_reuse_inventory.json',dict(binding=bp,rows=len(bler['rows']),point_counts=[dict(kind=k[0],snr_db=k[1],q=k[2],rows=v) for k,v in sorted(coverage.items(),key=str)],
      q6_available=False,source_probability_mode=bler.get('source_probability_mode'),real_image_payload_probability_validation_complete=bler.get('real_image_payload_probability_validation_complete'),
      reuse='Exact same physical layout/settings and measured SNR only; iid scrambled payload approximation for pre-screen, not real-source packet evidence.',
      absent_rows='UNKNOWN_NULL_NEVER_ZERO_OR_ONE',out_of_table_interpolation_admitted=False,exact_candidate_join='See sibling candidates_cpu_v1 coverage audit; candidate layouts and full implementation identities must match, not only k/n.')))
    completion=dict(status='R0_SOURCE_AND_HISTORICAL_REUSE_AUDIT_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
      git_head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),script=pin(Path(__file__)),
      original_calibration_sources_verified=1000,deterministic_calibration_subset=200,original_development_sources_verified=100,
      cal_dev_source_id_intersection=0,cal_dev_exact_pixel_intersection=0,development_restore_files_verified=202,
      prior_timing_seed16_unchanged=True,prior_timing_seed_indices=old16_indices,newly_available_development_sources=84,
      evidence_seed_files_verified=len(inventory['files']),evidence_inventory=evidence_pin,
      model_constructions=0,source_encoding_calls=0,channel_calls=0,packet_decodes=0,reconstructions=0,metric_calls=0,bootstrap_calls=0,
      r1_launch_admitted=False,r1_blocker='Fair shortlist cannot be certified by the present probability/clean-Q coverage. Missing tables remain unknown. Any supplementary science needs an explicitly bounded authorized registration.',
      outputs=outputs)
    dump('completion.json',completion)
    print(json.dumps({k:completion[k] for k in ['status','original_calibration_sources_verified','deterministic_calibration_subset','original_development_sources_verified','r1_launch_admitted']}))


if __name__ == '__main__':
    run()
