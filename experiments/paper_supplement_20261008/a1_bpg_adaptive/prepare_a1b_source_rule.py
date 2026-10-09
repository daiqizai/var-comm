"""Read an actually completed A1a probe and freeze only the adaptive source rule.

This is read-only with respect to scientific inputs. It does not select an MCS,
run a codec, evaluate a model, simulate a channel, or create holdout results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_bytes(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        require(p.read_bytes() == data, 'Refusing to replace different analysis: ' + str(p))
    else:
        with p.open('xb') as f:
            f.write(data)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def write_csv(path, rows):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
    w.writeheader(); w.writerows(rows)
    write_bytes(path, f.getvalue().encode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe', required=True, help='Local copy or original actual probe directory')
    p.add_argument('--request', required=True, help='Exact A1a request used by the actual process')
    p.add_argument('--out', required=True, help='New source-rule and descriptive-analysis directory')
    a = p.parse_args()
    probe, out = Path(a.probe), Path(a.out)
    request, done = read(a.request), read(probe / 'completion.json')
    require(done['status'] == 'A1A_CALIBRATION32_BPG_CODEC_PROBE_COMPLETE_V1', 'Probe is not complete')
    require(done['request_sha256'] == sha(a.request), 'Probe/request identity differs')
    require(done['source_count'] == 32 and done['source_ids'] == [r['source_id'] for r in request['records']],
            'Expected the fixed 32 original calibration sources')
    require(done['resolutions'] == [256, 128, 64, 32] and done['new_PHY_calls'] == 0
            and done['new_model_calls'] == 0 and not done['holdout_content_used'], 'Unexpected scientific scope')
    copied = ['cases.csv', 'feasibility.csv', 'summary.csv', 'runtime.json', 'request_identity.json',
              'bpgenc_help.json', 'bpgdec_help.json']
    for name in copied:
        original = str(Path(request['out']) / name).replace('\\', '/')
        require(done['outputs'].get(original) == sha(probe / name), 'Changed actual probe output: ' + name)
    rows = list(csv.DictReader((probe / 'cases.csv').open(encoding='utf-8', newline='')))
    require(len(rows) == done['tested_case_count'] <= request['candidate_count_upper_bound'], 'Wrong case count')
    keys, grouped = set(), {}
    for r in rows:
        for k in ['source_index', 'resolution', 'qp', 'complete_BPG_bytes', 'source_bits']:
            r[k] = int(r[k])
        for k in ['mse', 'psnr_db', 'encode_seconds', 'decode_seconds']:
            r[k] = float(r[k])
        key = r['source_index'], r['resolution'], r['qp']
        require(key not in keys, 'Duplicate source/resolution/QP')
        keys.add(key)
        require(0 <= r['source_index'] < 32 and r['source_id'] == request['records'][r['source_index']]['source_id'],
                'Wrong source population')
        require(r['resolution'] in request['resolutions'] and 0 <= r['qp'] <= 51
                and r['source_bits'] == r['complete_BPG_bytes'] * 8 and r['mse'] >= 0,
                'Invalid source-codec result')
        expected = math.inf if r['mse'] == 0 else -10 * math.log10(r['mse'])
        require(r['psnr_db'] == expected or abs(r['psnr_db'] - expected) < 1e-10, 'Per-image PSNR differs')
        grouped.setdefault((r['source_index'], r['resolution']), []).append(r)
    for source_index in range(32):
        for side in request['resolutions']:
            actual = {r['qp'] for r in grouped[source_index, side]}
            require(set(request['coarse_qps']).issubset(actual), 'Missing predeclared coarse point')
    selected, summaries = [], []
    for capacity in request['capacity_bytes']:
        for i in range(32):
            candidates = [r for side in request['resolutions'] for r in grouped[i, side]
                          if r['complete_BPG_bytes'] <= capacity]
            best = min(candidates, key=lambda r: (r['mse'], r['complete_BPG_bytes'], -r['resolution'], r['qp'])) if candidates else None
            native = [r for r in grouped[i, 256] if r['complete_BPG_bytes'] <= capacity]
            original = min(native, key=lambda r: (r['mse'], r['complete_BPG_bytes'], r['qp'])) if native else None
            selected.append(dict(source_index=i, source_id=request['records'][i]['source_id'], capacity_bytes=capacity,
                                 adaptive_fit=best is not None, selected_resolution=best['resolution'] if best else None,
                                 selected_qp=best['qp'] if best else None, selected_bytes=best['complete_BPG_bytes'] if best else None,
                                 selected_mse=best['mse'] if best else None, selected_psnr_db=best['psnr_db'] if best else None,
                                 selected_stream=best['stream'] if best else None,
                                 selected_stream_sha256=best['stream_sha256'] if best else None,
                                 native256_fit=original is not None,
                                 native256_best_tested_psnr_db=original['psnr_db'] if original else None,
                                 delta_psnr_on_both_fit=best['psnr_db'] - original['psnr_db'] if best and original else None))
        sub = [r for r in selected if r['capacity_bytes'] == capacity]
        both = [r['delta_psnr_on_both_fit'] for r in sub if r['delta_psnr_on_both_fit'] is not None]
        summaries.append(dict(capacity_bytes=capacity, source_count=32,
                              native256_fit_sources=sum(r['native256_fit'] for r in sub),
                              adaptive_fit_sources=sum(r['adaptive_fit'] for r in sub),
                              selected_resolution_counts={str(side): sum(r['selected_resolution'] == side for r in sub)
                                                          for side in request['resolutions']},
                              both_fit_sources=len(both), mean_delta_PSNR_both_fit=sum(both) / len(both) if both else None,
                              statistics='descriptive source-only; no confidence interval or noisy-link claim'))
    rule = dict(schema='A1B_ADAPTIVE_BPG_SOURCE_RULE_V1', status='SOURCE_RULE_FROZEN_MCS_NOT_SELECTED',
                name='BPG with adaptive source downsampling + LDPC', probe_completion_sha256=sha(probe / 'completion.json'),
                probe_request_sha256=sha(a.request), probe_source_ids=done['source_ids'],
                resolutions=request['resolutions'], coarse_qps=request['coarse_qps'], qp_range=request['qp_range'],
                capacity_bytes=request['capacity_bytes'], fine_rule=request['fine_rule'],
                encoder_options=['-e', 'x265', '-m', '8', '-f', '420', '-c', 'ycbcr', '-b', '8'],
                bpgenc=request['bpgenc'], bpgdec=request['bpgdec'],
                source_resize='Pillow uint8 RGB bicubic from the same original256',
                receiver_resize='decode complete BPG to its signalled dimensions; Pillow uint8 RGB bicubic to256',
                source_selection='among actually encoded and independently decoded candidates fitting the paid byte capacity: minimum float64 per-image MSE to original256',
                source_tie_break=['smaller complete BPG byte length', 'larger input resolution', 'smaller QP'],
                source_metadata='dimensions are carried inside the complete paid BPG file; no free per-image side information',
                link_failure_output='RGB float32 constant0.5 on source-unfit/header rejection/body CRC rejection/parser rejection/BPG decode failure',
                model_training=False, actual_noise_visible_to_source_selection=False, holdout_used_for_rule_selection=False,
                MCS_frozen=False, original_native256_results='retained as a separately labelled setting; not replaced',
                limits=['This is not a claim of global rate-distortion optimality.',
                        'Noisy calibration and final500 evaluation remain NOT_RUN for this adaptive rule.',
                        'At most three MCS per SNR enter actual calibration after length-matched BLER pre-screening.',
                        'A same-capacity MCS with a different LDPC length cannot reuse another length\'s BLER.'])
    write_csv(out / 'adaptive_probe_choices.csv', selected)
    write_json(out / 'source_feasibility_analysis.json', dict(status='ACTUAL_A1A_DESCRIPTIVE_ANALYSIS_COMPLETE',
               probe_completion_sha256=sha(probe / 'completion.json'), cases=len(rows), summaries=summaries,
               new_codec_calls=0, new_PHY_calls=0, new_model_calls=0, holdout_content_used=False))
    write_json(out / 'source_rule_frozen.json', rule)
    print(json.dumps({'status': rule['status'], 'out': str(out), 'cases': len(rows),
                      'next': 'length-matched BLER pre-screen, at most3 MCS per SNR, actual old-cal link evaluation'}))


if __name__ == '__main__':
    main()
