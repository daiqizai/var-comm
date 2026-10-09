"""Prepare a finite adaptive-BPG execution request after the actual A1a probe.

Preparation executes no codec, model, PHY or holdout selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from prepare_probe_materials import PROJECT, SUP, REMOTE, OLD_REQUEST_SHA, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True)
    p.add_argument('--probe', required=True)
    p.add_argument('--probe-request', required=True)
    p.add_argument('--source-rule', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    w, here = Path(a.workspace), Path(__file__).resolve().parent
    prior = w / '.research/main_raw64_20261008_paper_supplement'
    old_path = prior / 'bpg_codec_runtime_r2/materials_r2/request.json'
    assert sha(old_path) == OLD_REQUEST_SHA
    old = json.loads(old_path.read_text())
    probe, rule = json.loads(Path(a.probe_request).read_text()), json.loads(Path(a.source_rule).read_text())
    done_path = Path(a.probe) / 'completion.json'; done = json.loads(done_path.read_text())
    assert done['status'] == 'A1A_CALIBRATION32_BPG_CODEC_PROBE_COMPLETE_V1'
    assert done['request_sha256'] == sha(a.probe_request) == rule['probe_request_sha256']
    assert rule['probe_completion_sha256'] == sha(done_path)
    assert rule['status'] == 'SOURCE_RULE_FROZEN_MCS_NOT_SELECTED' and not rule['MCS_frozen']
    assert len(old['records']['calibration']) == 100 and len(old['records']['holdout']) == 500
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    remote_materials = REMOTE + '/a1b_materials_v1'
    modules = ['adaptive_stages.py', 'adaptive_codec.py', 'probe_source_codec.py']
    bindings = {REMOTE + '/' + name: sha(here / name) for name in modules}
    old_reg = json.loads((prior / 'bpg_phy_runtime_v1/materials_v1/phy_registration.json').read_text())
    bindings.update(old['codec_source_bindings']); bindings.update(old_reg['source_bindings'])
    native = {population: {'path': SUP + '/bpg_results_v1/codec/' + population + '/completion.json',
                          'sha256': sha(prior / 'bpg_audit/actual/codec' / population / 'completion.json')}
              for population in ['calibration', 'holdout']}
    inherited_qualification = {'path': SUP + '/bpg_results_v1/qualification/worker_0.json',
                              'sha256': sha(prior / 'bpg_audit/actual/qualification/worker_0.json')}
    inherited_calibration = {'path': SUP + '/bpg_results_v1/calibration/completion.json',
                            'sha256': sha(prior / 'bpg_audit/actual/calibration/completion.json')}
    catalogue = old['catalogue'] + [dict(profile_id=3014, q=6, rate='1/3', k=1912, n=5736,
                                       capacity_bytes=235, N=1024, header_symbols=68, body_symbols=956)]
    destination = PROJECT + '/results/paper_supplement_20261008/a1_bpg_adaptive/a1b_adaptive_v1'
    req = dict(schema='A1B_ADAPTIVE_BPG_REQUEST_V1',
               original_request={'path': SUP + '/bpg_metadata_r2/request.json', 'sha256': sha(old_path)},
               probe_request={'path': REMOTE + '/materials_v1/request.json', 'sha256': sha(a.probe_request)},
               probe_completion={'path': probe['out'] + '/completion.json', 'sha256': sha(done_path)},
               probe_root=probe['out'],
               source_rule={'path': remote_materials + '/source_rule_frozen.json', 'sha256': sha(a.source_rule)},
               inherited_qualification=inherited_qualification, inherited_calibration=inherited_calibration,
               original_native_codec_completion=native, original_native_root=SUP + '/bpg_results_v1/codec',
               records=old['records'], catalogue=catalogue, SNRs=old['SNRs'], noise_seeds=old['noise_seeds'],
               python=old['python'], bpgenc=old['bpgenc'], bpgdec=old['bpgdec'],
               source_bindings=bindings, phy_dependency_bindings=old['phy_dependency_bindings'],
               protocol=old['protocol'], original_root_ledger_readonly=old['root_ledger_readonly'],
               out=destination, independent_ledger=destination + '/ledger.sqlite3',
               phase_caps={'qualification': 44, 'prescreen': 5760, 'calibration': 10800, 'holdout': 18000},
               total_new_PHY_cap=34604,
               calibration_sources=100, actual_MCS_per_SNR_max=3,
               prior_BLER_minimum_actual_frames=32, missing_BLER_fixed_frames=32,
               missing_BLER_noise_seeds=list(range(9001, 9033)),
               prescreen_rule='top3 predicted all-source mean per-image PSNR using32cal source quality and matching actual packet delivery frequency; tie smallest profile_id',
               prescreen_scope='selection proxy only; not reported noisy image quality',
               source_objective='minimum MSE to original256 among legal actual source candidates',
               calibration_objective='maximum all100cal source-mean per-image PSNR over original3noise; tie smallest profile_id',
               root_ledger_mutation=False, old_main_results_modified=False, holdout_used_for_selection=False,
               resources={'workers': 8, 'threads': 2, 'nice': 15,
                          'affinities': [[16 + 2*i, 17 + 2*i] for i in range(8)]},
               deadline_unix=a.deadline_unix, codec_timeout_seconds=600,
               stop_files=old['stop_files'] + [destination + '/STOP'],
               source_candidate_upper_bound={'calibration_new': 68 * 4 * 52, 'holdout_new': 500 * 4 * 52},
               source_probe_reuse='first32cal actual cases reused after SHA and rule identity verification',
               native256_rule='old native256 result remains separate and immutable',
               receiver_truth_rule='only received profile and received length/BPG bytes determine decoding; TX truth diagnostic only')
    write(out / 'request.json', req)
    # Source rule has already been frozen by the previous verified, read-only step.
    (out / 'source_rule_frozen.json').write_bytes(Path(a.source_rule).read_bytes())
    puts = [{'local_path': str(here / name), 'remote_path': REMOTE + '/' + name, 'sha256': sha(here / name)} for name in modules]
    puts += [{'local_path': str(out / name), 'remote_path': remote_materials + '/' + name, 'sha256': sha(out / name)}
             for name in ['request.json', 'source_rule_frozen.json']]
    stages = ['qualification', 'screen', 'codec_calibration', 'calibration', 'freeze', 'codec_holdout', 'holdout']
    argv = {s: [old['python']['path'], '-B', REMOTE + '/adaptive_stages.py', '--request',
                remote_materials + '/request.json', '--stage', s] for s in stages}
    write(out / 'execution.json', {'status': 'PREPARED_NOT_EXECUTED', 'puts': puts, 'stage_order': stages,
                                   'stage_argv': argv, 'request_sha256': sha(out / 'request.json'),
                                   'remote_output': destination, 'phase_caps': req['phase_caps'],
                                   'total_new_PHY_cap': req['total_new_PHY_cap'],
                                   'environment': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
                                                   'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}})
    print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'execution': str(out / 'execution.json')}))


if __name__ == '__main__':
    main()
