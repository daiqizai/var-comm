"""Local preparation only. Takes an explicit authorized deadline; runs no codec."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PROJECT = '/home/liulu/projects/VAR_COMM'
SUP = PROJECT + '/outputs/PAPER-SUPPLEMENT-20261008'
REMOTE = PROJECT + '/experiments/paper_supplement_20261008/a1_bpg_adaptive'
OLD_REQUEST_SHA = 'd4fc512cb0970e5826c79287e7e64bbf3da5d4355fa7cc94951343b0ec4fb833'
OLD_CODEC_DONE_SHA = '1540dd056fe2a053ded3c42c51d736c071ad48c0141daac976ebecb113c1383f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    p = Path(path)
    if p.exists() and p.read_bytes() != data:
        raise ValueError('Refusing to change existing prepared metadata: ' + str(p))
    if not p.exists():
        with p.open('xb') as f:
            f.write(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True)
    p.add_argument('--out', required=True, help='Fresh local materials directory')
    p.add_argument('--deadline-unix', required=True, type=float)
    p.add_argument('--max-seconds', type=int, default=14400)
    a = p.parse_args()
    w = Path(a.workspace).resolve()
    s = w / '.research/main_raw64_20261008_paper_supplement'
    source = Path(__file__).resolve().parent / 'probe_source_codec.py'
    old = s / 'bpg_codec_runtime_r2/materials_r2/request.json'
    old_done = s / 'bpg_audit/actual/codec/calibration/completion.json'
    assert sha(old) == OLD_REQUEST_SHA and sha(old_done) == OLD_CODEC_DONE_SHA
    r = json.loads(old.read_text()); done = json.loads(old_done.read_text())
    assert done['request_sha256'] == sha(old) and done['worker_exit_codes'] == [0] * 8
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    remote_request = REMOTE + '/materials_v1/request.json'
    remote_output = PROJECT + '/results/paper_supplement_20261008/a1_bpg_adaptive/a1a_probe_v1'
    runner = {'path': REMOTE + '/probe_source_codec.py', 'sha256': sha(source)}
    req = dict(schema='A1A_BPG_CALIBRATION_CODEC_PROBE_REQUEST_V1', source_count=32,
               records=r['records']['calibration'][:32], resolutions=[256, 128, 64, 32],
               qp_range=[0, 51], coarse_qps=[0, 8, 16, 24, 32, 40, 48, 51],
               fine_rule='all_integer_QPs_between_adjacent_coarse_points_straddling_any_capacity',
               candidate_count_upper_bound=32 * 4 * 52,
               capacity_bytes=sorted({c['capacity_bytes'] for c in r['catalogue']}), catalogue=r['catalogue'],
               additional_64QAM_rate_1_3='not PHY-qualified here; capacity235 already probed through old16QAM1/2',
               original_request={'path': SUP + '/bpg_metadata_r2/request.json', 'sha256': sha(old)},
               native_codec_completion={'path': SUP + '/bpg_results_v1/codec/calibration/completion.json', 'sha256': sha(old_done)},
               native_codec_root=SUP + '/bpg_results_v1/codec/calibration',
               bpgenc=r['bpgenc'], bpgdec=r['bpgdec'], python=r['python'], runner=runner,
               resources={'affinity': [16, 17], 'threads': 2, 'nice': 15},
               deadline_unix=a.deadline_unix, max_seconds=a.max_seconds,
               codec_timeout_seconds=r['codec_timeout_seconds'],
               out=remote_output, stop_files=[PROJECT + '/STOP', PROJECT + '/outputs/MAIN-RAW64-20261007/STOP',
                                            SUP + '/STOP', remote_output + '/STOP'],
               source_resize='uint8_RGB_Pillow_BICUBIC', restoration='decoded_uint8_RGB_Pillow_BICUBIC_to256',
               source_quality_target='original256_RGB; float64_per_image_MSE_and_PSNR',
               native256_original_results_reused=True, holdout_content_used=False,
               new_model_calls=0, new_PHY_calls=0, policy_frozen=False,
               prospective_adaptive_policy='NOT_SELECTED; inspect actual original-cal results first')
    write(out / 'request.json', req)
    bindings = {runner['path']: runner['sha256'], remote_request: sha(out / 'request.json')}
    for key in ['python', 'bpgenc', 'bpgdec', 'original_request', 'native_codec_completion']:
        bindings[req[key]['path']] = req[key]['sha256']
    for record in req['records']:
        bindings[record['archive']] = record['archive_sha256']
        bindings[record['checkpoint']] = record['checkpoint_sha256']
    argv = [r['python']['path'], '-B', runner['path'], '--request', remote_request]
    delivery = dict(status='LOCAL_PREPARATION_ONLY_A1A_NOT_EXECUTED',
                    request_sha256=sha(out / 'request.json'), argv=argv,
                    put=[{'local_path': str(source), 'remote_path': runner['path'], 'sha256': sha(source)},
                         {'local_path': str(out / 'request.json'), 'remote_path': remote_request,
                          'sha256': sha(out / 'request.json')}],
                    readonly_bindings=bindings, local_output=str(out), remote_output=remote_output,
                    preflight_argv=[r['python']['path'], '-B', '-c',
                                   'import numpy,PIL; print(numpy.__version__,PIL.__version__)'],
                    fetch=[remote_output + '/' + name for name in ['completion.json', 'cases.csv', 'feasibility.csv',
                                                                  'summary.csv', 'runtime.json', 'request_identity.json',
                                                                  'bpgenc_help.json', 'bpgdec_help.json']],
                    external_wait_and_closed_log_required=True, no_remote_execution=True)
    write(out / 'execution.json', delivery)
    print(json.dumps({'execution': str(out / 'execution.json'), 'request_sha256': delivery['request_sha256'],
                      'runner_sha256': runner['sha256'], 'argv': argv}, sort_keys=True))


if __name__ == '__main__':
    main()
