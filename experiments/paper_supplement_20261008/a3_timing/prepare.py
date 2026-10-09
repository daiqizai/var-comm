"""Build metadata-only A3 materials from completed local evidence. No SSH/GPU."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / '.research/main_raw64_20261008_paper_supplement/timing_audit'
TAKEOVER = ROOT / '.research/main_raw64_20261007/take_over_v1'
REMOTE = '/home/liulu/projects/VAR_COMM'
RUN = REMOTE + '/outputs/PAPER-SUPPLEMENT-20261008/a3_timing_v1'
SNRS = [7, 13, 19]
METHODS = ['RAW64_PARTIAL', 'RAW64_WHOLE', 'P1024', 'SwinJSCC80k']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def prepare(out, maximum_seconds):
    out = Path(out)
    if out.exists():
        raise RuntimeError('Materials directory already exists; preserve it and choose a fresh directory')
    assert 0 < maximum_seconds <= 7200
    old_path = OLD / 'raw_materials_r3/request.json'
    cost_path = TAKEOVER / 'raw64_online_cost_v2/materials_v2/request.json'
    old, cost = read(old_path), read(cost_path)
    assert len(cost['records']) == 16 and cost['plan']['source_indices'] == [0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
    keep = ['root','python','pythonpath','threads','interop_threads','affinity','nice','visual_lock',
            'baseline_request','raw_config','bindings','executable_bindings','private_sionna_manifest',
            'private_sionna_parent','sionna_source','source_driver','pair_module','timing_module','stop_files']
    r = {key:deepcopy(old[key]) for key in keep}
    r.update(schema='A3_FIXED16_UNIFIED_TIMING_V1', out=RUN+'/results', SNRs=SNRS, methods=METHODS,
        source_indices=cost['plan']['source_indices'], source_ids=cost['plan']['source_ids'], records=cost['records'],
        warmup_per_source_snr=1, measured_per_source_snr=3, batch_size=1,
        max_seconds_per_method=maximum_seconds, training=False, policy_selection=False,
        original_ledger_mutated=False, holdout_used=False, automatic_successor=False,
        packet_caps={'RAW64_PARTIAL':384,'RAW64_WHOLE':384,'P1024':0,'SwinJSCC80k':192},
        total_independent_packet_cap=960, expected_frames_per_method=192,
        development_plan_module=cost['development_plan_module'],
        helper_module=REMOTE+'/outputs/PAPER-SUPPLEMENT-20261008/unified_timing_v1/runtime/unified_endpoint_benchmark_v1.py',
        points={method:{str(snr):next(point for point in cost['plan']['points']
            if point['timing_family'] == method.removeprefix('RAW64_') and point['snr_db'] == snr)
            for snr in SNRS} for method in METHODS if method.startswith('RAW64_')},
        timing_protocol=dict(boundary='CPU CHW uint8 -> complete CPU1024x2 IQ -> received CPU IQ -> one CPU float32 RGB',
            e2e='One measured outer window; channel included and separately subtracted',
            repeated_e2e_second_pass=False, packet_qualification_repeated=False,
            deterministic_fixed_noise_per_source_snr=True, RAW_seed=6201, P_Swin_seed=2001,
            model_loading_disk_IO_metrics_outside_timing=True,
            zero_gray_failures_retained=True, inclusive_components_never_added_to_totals=True))
    assert r['bindings'][r['development_plan_module']] == 'b5c425ded7a43b8bb92f364e4dc0f7e007304b2e174d85e6d63b102551ecf047'
    r['stop_files'] += [RUN+'/STOP']
    r['a3_sources'] = {name:dict(path=RUN+'/runtime/'+name, sha256=sha(Path(__file__).parent/name))
        for name in ['benchmark.py','raw_endpoint.py']}
    r['prior_completed_timing'] = {}
    oldrun = REMOTE+'/outputs/PAPER-SUPPLEMENT-20261008/unified_timing_v1'
    for method in ['RAW64_PARTIAL','P1024','SwinJSCC80k']:
        revision = 'r3' if method == 'RAW64_PARTIAL' else 'v1'
        local = OLD/f'actual_results_{revision}'/method/'completion.json'
        done = read(local)
        assert done['status'] == 'INDEPENDENT_UNIFIED_UM_ENDPOINT_TIMING_COMPLETE_V1' and done['migration_validation_cases'] == 48
        r['prior_completed_timing'][method] = dict(path=f'{oldrun}/results_{revision}/{method}/completion.json',sha256=sha(local))
    p_checkpoint = read(TAKEOVER/'unified500_baseline_r2/gpu_materials_v1/request.json')['P_identity']['original_selected_record']['checkpoint']
    vae = '/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth'
    var = '/home/liulu/projects/VAR-MAP-GATE0/checkpoints/var_d16.pth'
    dc = REMOTE+'/outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_A_v1/checkpoints/update_0038000_1789635001035591298.pt'
    swin = REMOTE+'/outputs/EXTERNAL-COMPARISON-20261004/swin_training/checkpoints/model_080000.pt'
    weights = {'RAW64_PARTIAL':[vae,var,dc], 'RAW64_WHOLE':[vae,var,dc], 'P1024':[vae,p_checkpoint,dc], 'SwinJSCC80k':[swin]}
    r['weight_files'] = {method:[dict(path=p,sha256=r['bindings'][p]) for p in paths] for method,paths in weights.items()}
    out.mkdir(parents=True)
    save(out/'request.json',r)
    uploads = {str(Path(__file__).parent/name): entry['path'] for name,entry in r['a3_sources'].items()}
    uploads[str(out.resolve()/'request.json')] = RUN+'/materials/request.json'
    save(out/'local_preparation.json', dict(status='PREPARED_NOT_RUN', request_sha256=sha(out/'request.json'),
        prior_local_evidence=[dict(path=str(old_path),sha256=sha(old_path)),dict(path=str(cost_path),sha256=sha(cost_path))],
        uploads=uploads, methods=METHODS, frames_per_method=192, total_neural_frames=768,
        packet_caps=r['packet_caps'], total_packet_cap=960, new_qualification_packets=0,
        note='Only preparation. No SSH, model construction, GPU work, or PHY decoding executed.'))
    return dict(status='PREPARED_NOT_RUN', directory=str(out.resolve()), request_sha256=sha(out/'request.json'), uploads=uploads)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'.research/main_raw64_20261008_paper_supplement/a3_fixed16/materials_v1')
    p.add_argument('--max-seconds-per-method',type=int,default=3600)
    a = p.parse_args(); print(json.dumps(prepare(a.out,a.max_seconds_per_method),indent=2))
