"""A1a: original-calibration-only BPG rate/distortion probe; no PHY or models.

Keeps the completed native-256 experiment intact. Each new complete stream is
independently decoded. A partial/failed subprocess is never silently retried.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
import zlib

STOP = []
SIDES = [256, 128, 64, 32]
COARSE = [0, 8, 16, 24, 32, 40, 48, 51]
OPTIONS = ['-e', 'x265', '-m', '8', '-f', '420', '-c', 'ycbcr', '-b', '8']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    immutable_bytes(path, (json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False) + '\n').encode())


def immutable_bytes(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        require(p.read_bytes() == data, 'Existing immutable file differs: ' + str(p))
        return
    tmp = p.with_name(p.name + '.partial.' + str(os.getpid()))
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def pin(descriptor):
    require(sha(descriptor['path']) == descriptor['sha256'],
            'Changed input: ' + descriptor['path'])


def outputs_valid(value):
    for p, expected in value['outputs'].items():
        require(sha(p) == expected, 'Changed completed output: ' + p)


def png_bytes(rgb):
    """Same PNG construction as the old codec, generalized only in dimensions."""
    import numpy as np
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3,
            'Expected uint8 HWC RGB')
    h, w, _ = rgb.shape
    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value))
    raw = b''.join(b'\x00' + row.tobytes() for row in np.ascontiguousarray(rgb))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def ppm_rgb(path, side):
    import numpy as np
    data, fields, pos = Path(path).read_bytes(), [], 0
    while len(fields) < 4:
        while pos < len(data) and data[pos] in b' \t\r\n':
            pos += 1
        require(pos < len(data), 'Truncated PPM header')
        if data[pos:pos + 1] == b'#':
            pos = data.index(b'\n', pos) + 1
            continue
        end = pos
        while end < len(data) and data[end] not in b' \t\r\n':
            end += 1
        fields.append(data[pos:end]); pos = end
    require(fields == [b'P6', str(side).encode(), str(side).encode(), b'255'],
            'Unexpected decoder RGB/size/range')
    require(pos < len(data) and data[pos] in b' \t\r\n', 'Missing PPM separator')
    pos += 2 if data[pos:pos + 2] == b'\r\n' else 1
    require(len(data) - pos == side * side * 3, 'Unexpected PPM raster length')
    return np.frombuffer(data[pos:], np.uint8).reshape(side, side, 3).copy()


class Probe:
    def __init__(self, request, request_path):
        self.r, self.rpath = request, str(Path(request_path).resolve())
        self.out = Path(request['out'])
        self.started = time.time()
        self.fresh_encodes = self.native_reuses = self.fresh_decodes = 0
        self.new_case_receipts = 0
        self.original = read(request['original_request']['path'])
        self.old_done = read(request['native_codec_completion']['path'])

    def check(self):
        require(not STOP, 'Requested stop signal: ' + str(STOP))
        require(time.time() < min(self.r['deadline_unix'],
                                  self.started + self.r['max_seconds']), 'Probe deadline reached')
        require(not any(Path(p).exists() for p in self.r['stop_files']), 'STOP file exists')

    def call(self, argv, log, allow_help=False):
        self.check()
        require(not Path(log).exists(), 'Unreceipted codec log exists; no automatic retry')
        wall, monotonic = time.time(), time.perf_counter()
        process = None
        with Path(log).open('xb') as f:
            try:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    self.check()
                    require(time.perf_counter() - monotonic < self.r['codec_timeout_seconds'], 'Codec timeout')
                    time.sleep(0.1)
                code = process.wait()
            except BaseException:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                    process.wait()
                raise
            finally:
                f.flush(); os.fsync(f.fileno())
        result = dict(argv=argv, pid=process.pid, started_unix=wall,
                      ended_unix=time.time(), elapsed_seconds=time.perf_counter() - monotonic,
                      process_waited=True, exit_code=code, log=str(log), log_sha256=sha(log))
        require(code in ([0, 1] if allow_help else [0]), 'Codec failed: ' + str(result))
        return result

    def preflight(self):
        r = self.r
        require(r['schema'] == 'A1A_BPG_CALIBRATION_CODEC_PROBE_REQUEST_V1', 'Wrong request schema')
        require(r['resolutions'] == SIDES and r['coarse_qps'] == COARSE and r['qp_range'] == [0, 51],
                'Changed predeclared resolution/QP grid')
        require(r['fine_rule'] == 'all_integer_QPs_between_adjacent_coarse_points_straddling_any_capacity',
                'Changed refinement rule')
        require(r['source_count'] == 32 and len(r['records']) == 32, 'Expected exactly original 32 cal sources')
        for key in ['original_request', 'native_codec_completion', 'bpgenc', 'bpgdec', 'python', 'runner']:
            pin(r[key])
        require(r['records'] == self.original['records']['calibration'][:32], 'Wrong calibration population/order')
        require(r['bpgenc'] == self.original['bpgenc'] and r['bpgdec'] == self.original['bpgdec'], 'Changed actual codec')
        require(self.old_done['request_sha256'] == r['original_request']['sha256']
                and self.old_done['status'] == 'BPG_CODEC_POPULATION_COMPLETE_V1'
                and self.old_done['population'] == 'calibration'
                and self.old_done['worker_exit_codes'] == [0] * 8, 'Original codec lacks normal completion')
        require(r['catalogue'] == self.original['catalogue'], 'Changed original 14 MCS metadata')
        require(r['native_codec_root'] == str(Path(self.original['out']) / 'codec' / 'calibration'),
                'Wrong original native codec cache')
        expected_capacities = sorted({x['capacity_bytes'] for x in r['catalogue']})
        require(r['capacity_bytes'] == expected_capacities, 'Changed source-size probe capacities')
        require(r['resources'] == {'affinity': [16, 17], 'threads': 2, 'nice': 15}, 'Wrong CPU resource scope')
        require(sys.platform == 'linux', 'CPU probe must run on original Linux codec installation')
        os.sched_setaffinity(0, r['resources']['affinity'])
        os.setpriority(os.PRIO_PROCESS, 0, r['resources']['nice'])
        for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            os.environ[key] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        self.check()
        self.out.mkdir(parents=True, exist_ok=True)
        save(self.out / 'request_identity.json', {'path': self.rpath, 'sha256': sha(self.rpath)})
        for key in ['bpgenc', 'bpgdec']:
            receipt = self.out / (key + '_help.json')
            if receipt.exists():
                done = read(receipt); outputs_valid(done)
                require(done['argv'] == [r[key]['path'], '-h'] and done['process_waited'], 'Changed help receipt')
            else:
                done = self.call([r[key]['path'], '-h'], self.out / (key + '_help.log'), True)
                done['outputs'] = {done['log']: done['log_sha256']}; save(receipt, done)
            help_text = Path(done['log']).read_text(errors='replace')
            require('version 0.9.8' in help_text, 'Unexpected actual libbpg version')
            if key == 'bpgenc':
                require(re.search(r'range:\s*0\s*-\s*51\b', help_text), 'Actual QP range differs')

    def encode(self, directory, source_png, side, index, qp):
        receipt = directory / ('qp%02d.encode.json' % qp)
        stream = directory / ('qp%02d.bpg' % qp)
        source_sha = sha(source_png)
        argv = [self.r['bpgenc']['path']] + OPTIONS + ['-q', str(qp), '-o', str(stream), str(source_png)]
        if receipt.exists():
            done = read(receipt); outputs_valid(done)
            require(done['source_png_sha256'] == source_sha and done['qp'] == qp
                    and done['encoder_sha256'] == self.r['bpgenc']['sha256']
                    and done['process_waited'] and done['exit_code'] == 0, 'Changed encoding receipt')
            require(done['argv'] == argv or done['reused_original_native256'], 'Changed encode argv')
            return done
        require(not stream.exists(), 'Unreceipted stream retained; no automatic retry')
        if side == 256:
            old_dir = Path(self.r['native_codec_root']) / ('%04d' % index)
            old_receipt, old_stream = old_dir / ('qp%02d.json' % qp), old_dir / ('qp%02d.bpg' % qp)
            if old_receipt.exists():
                require(self.old_done['outputs'].get(str(old_receipt)) == sha(old_receipt), 'Unbound native receipt')
                old = read(old_receipt)
                old_argv = [self.r['bpgenc']['path']] + OPTIONS + ['-q', str(qp), '-o', str(old_stream), str(old_dir / 'source.png')]
                require(old['argv'] == old_argv and old['source_png_sha256'] == source_sha
                        and old['encoder_sha256'] == self.r['bpgenc']['sha256']
                        and old['process_waited'] and old['exit_code'] == 0, 'Native codec identity differs')
                require(sha(old_dir / 'source.png') == source_sha
                        and self.old_done['outputs'].get(str(old_dir / 'source.png')) == source_sha,
                        'Original native source PNG changed/unbound')
                outputs_valid(old)
                require(self.old_done['outputs'].get(str(old_stream)) == sha(old_stream), 'Unbound native stream')
                done = dict(old, reused_original_native256=True, original_receipt=str(old_receipt),
                            original_receipt_sha256=sha(old_receipt), stream=str(old_stream),
                            stream_sha256=sha(old_stream), encode_seconds=old['ended_unix'] - old['identity']['started_unix'],
                            encode_time_scope='original_actual_receipt_wall_time_not_fresh_timing')
                self.native_reuses += 1
                save(receipt, done)
                return done
        done = self.call(argv, directory / ('qp%02d.encode.log' % qp))
        require(stream.exists() and stream.stat().st_size > 0, 'Encoder returned no complete stream')
        done.update(qp=qp, encoder_sha256=self.r['bpgenc']['sha256'], source_png_sha256=source_sha,
                    stream=str(stream), stream_sha256=sha(stream), bytes=stream.stat().st_size,
                    encode_seconds=done['elapsed_seconds'], encode_time_scope='fresh_codec_wall_time',
                    reused_original_native256=False, outputs={str(stream): sha(stream), done['log']: done['log_sha256']})
        self.fresh_encodes += 1
        save(receipt, done)
        return done

    def case(self, directory, source_png, side, index, qp, original_rgb):
        import numpy as np
        from PIL import Image
        receipt = directory / ('qp%02d.case.json' % qp)
        if receipt.exists():
            done = read(receipt); outputs_valid(done)
            require(done['request_sha256'] == sha(self.rpath) and done['qp'] == qp and done['resolution'] == side,
                    'Changed completed probe case')
            return done
        enc = self.encode(directory, source_png, side, index, qp)
        ddir = directory / 'decoded' / enc['stream_sha256']; ddir.mkdir(parents=True, exist_ok=True)
        dr = ddir / 'receipt.json'; ppm = ddir / 'decoded.ppm'
        # Different QPs can emit identical bytes. Give each content-addressed
        # decoder cache a canonical input path so an equal stream is reusable.
        decoder_input = ddir / 'input.bpg'
        immutable_bytes(decoder_input, Path(enc['stream']).read_bytes())
        argv = [self.r['bpgdec']['path'], '-o', str(ppm), str(decoder_input)]
        if dr.exists():
            dec = read(dr); outputs_valid(dec)
            require(dec['argv'] == argv and dec['process_waited'] and dec['exit_code'] == 0, 'Changed decode receipt')
            require(dec['decoder_sha256'] == self.r['bpgdec']['sha256']
                    and dec['stream_sha256'] == enc['stream_sha256'], 'Changed actual decoder/input identity')
        else:
            require(not ppm.exists(), 'Unreceipted decoded output retained')
            dec = self.call(argv, ddir / 'decode.log')
            ppm_rgb(ppm, side)
            dec.update(decoder_sha256=self.r['bpgdec']['sha256'], stream_sha256=enc['stream_sha256'],
                       outputs={str(ppm): sha(ppm), str(decoder_input): sha(decoder_input),
                                dec['log']: dec['log_sha256']})
            self.fresh_decodes += 1; save(dr, dec)
        decoded = ppm_rgb(ppm, side)
        restored = decoded if side == 256 else np.asarray(Image.fromarray(decoded).resize((256, 256), Image.Resampling.BICUBIC))
        restored = np.ascontiguousarray(restored, dtype=np.uint8)
        restored_path = ddir / 'restored256.png'; immutable_bytes(restored_path, png_bytes(restored))
        mse = float(np.mean(((restored.astype(np.float64) - original_rgb.astype(np.float64)) / 255.0) ** 2))
        # Infinity is represented explicitly in strict JSON, and as inf in CSV.
        row = dict(source_index=index, source_id=self.r['records'][index]['source_id'], resolution=side, qp=qp,
                   source_bits=enc['bytes'] * 8, complete_BPG_bytes=enc['bytes'], mse=mse,
                   psnr_db=None if mse == 0 else -10.0 * math.log10(mse), psnr_is_infinite=mse == 0,
                   encode_seconds=enc['encode_seconds'], encode_time_scope=enc['encode_time_scope'],
                   reused_original_native256=enc['reused_original_native256'], decode_seconds=dec['elapsed_seconds'],
                   stream=enc['stream'], stream_sha256=enc['stream_sha256'], restored_png=str(restored_path),
                   independent_decode=True, decoder_size=[side, side], restore='identity' if side == 256 else 'Pillow_uint8_RGB_BICUBIC_to_256',
                   over_budget={str(c): enc['bytes'] > c for c in self.r['capacity_bytes']},
                   request_sha256=sha(self.rpath), outputs={str(receipt.with_name('qp%02d.encode.json' % qp)): sha(receipt.with_name('qp%02d.encode.json' % qp)),
                   str(dr): sha(dr), str(restored_path): sha(restored_path), str(source_png): sha(source_png)})
        row['outputs'].update(enc['outputs']); row['outputs'].update(dec['outputs'])
        save(receipt, row)
        self.new_case_receipts += 1
        if self.new_case_receipts % 20 == 0:
            print(json.dumps({'progress': 'codec_cases', 'cases_this_invocation': self.new_case_receipts,
                              'elapsed_seconds': time.time() - self.started,
                              'fresh_encodes': self.fresh_encodes, 'native_stream_reuses': self.native_reuses,
                              'fresh_decodes': self.fresh_decodes, 'source_index': index,
                              'resolution': side, 'qp': qp}), flush=True)
        return row

    def run(self):
        import numpy as np
        import PIL
        from PIL import Image
        self.preflight()
        runtime = dict(python=sys.version, executable=sys.executable, numpy=np.__version__, Pillow=PIL.__version__,
                       affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0),
                       CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], source_resize='Pillow_uint8_RGB_BICUBIC',
                       PSNR='float64 mean of RGB squared error / 255**2; per image; reference original256',
                       time_scope='codec subprocess wall time; includes codec file IO, not deployment timing')
        save(self.out / 'runtime.json', runtime)
        all_rows, completed = [], []
        for index, record in enumerate(self.r['records']):
            self.check()
            pin({'path': record['archive'], 'sha256': record['archive_sha256']})
            pin({'path': record['checkpoint'], 'sha256': record['checkpoint_sha256']})
            with np.load(record['archive'], allow_pickle=False) as a:
                pixels = a['pixels'].copy()
            require(pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256)
                    and hashlib.sha256(pixels.tobytes()).hexdigest() == record['preprocessing_id'], 'Original source RGB differs')
            rgb = np.ascontiguousarray(pixels.transpose(1, 2, 0))
            source_dir = self.out / 'sources' / ('%04d' % index)
            source_rows = []
            for side in SIDES:
                directory = source_dir / ('r%d' % side); directory.mkdir(parents=True, exist_ok=True)
                resized = rgb if side == 256 else np.asarray(Image.fromarray(rgb).resize((side, side), Image.Resampling.BICUBIC))
                png = directory / 'source.png'; immutable_bytes(png, png_bytes(resized))
                rows = {q: self.case(directory, png, side, index, q, rgb) for q in COARSE}
                # No monotonic-size assumption: refine every actual crossing in either direction.
                fine = set()
                for left, right in zip(COARSE, COARSE[1:]):
                    sizes = sorted([rows[left]['complete_BPG_bytes'], rows[right]['complete_BPG_bytes']])
                    if any(sizes[0] <= c < sizes[1] for c in self.r['capacity_bytes']):
                        fine.update(range(left + 1, right))
                for qp in sorted(fine):
                    rows[qp] = self.case(directory, png, side, index, qp, rgb)
                source_rows.extend(rows[q] for q in sorted(rows))
            cp = source_dir / 'completion.json'
            cp_outputs = {}
            for row in source_rows: cp_outputs.update(row['outputs'])
            for p in source_dir.rglob('qp*.case.json'): cp_outputs[str(p)] = sha(p)
            save(cp, dict(status='A1A_SOURCE_CODEC_PROBE_COMPLETE_V1', source=record, rows=source_rows,
                          training_updates=0, new_PHY_calls=0, outputs=cp_outputs))
            all_rows.extend(source_rows); completed.append(str(cp))
            print(json.dumps({'completed_sources': index + 1, 'source_count': 32, 'cases': len(all_rows)}), flush=True)
        fields = ['source_index', 'source_id', 'resolution', 'qp', 'complete_BPG_bytes', 'source_bits', 'mse', 'psnr_db',
                  'encode_seconds', 'encode_time_scope', 'decode_seconds', 'reused_original_native256', 'stream', 'stream_sha256']
        write_csv(self.out / 'cases.csv', fields, [{**r, 'psnr_db': 'inf' if r['psnr_is_infinite'] else r['psnr_db']} for r in all_rows])
        feasibility = []
        for side in SIDES:
            for capacity in self.r['capacity_bytes']:
                for index in range(32):
                    rows = [r for r in all_rows if r['resolution'] == side and r['source_index'] == index]
                    fits = [r for r in rows if r['complete_BPG_bytes'] <= capacity]
                    best = min(fits, key=lambda r: (r['mse'], r['complete_BPG_bytes'], r['qp'])) if fits else None
                    feasibility.append(dict(source_index=index, resolution=side, capacity_bytes=capacity,
                                            fit=bool(fits), unfit=not fits, tested_candidates=len(rows),
                                            minimum_tested_bytes=min(r['complete_BPG_bytes'] for r in rows),
                                            best_tested_qp=None if best is None else best['qp'],
                                            best_tested_mse=None if best is None else best['mse'],
                                            best_tested_psnr=None if best is None else ('inf' if best['psnr_is_infinite'] else best['psnr_db'])))
        write_csv(self.out / 'feasibility.csv', list(feasibility[0]), feasibility)
        summary = []
        for side in SIDES:
            for capacity in self.r['capacity_bytes']:
                subset = [r for r in feasibility if r['resolution'] == side and r['capacity_bytes'] == capacity]
                summary.append(dict(resolution=side, capacity_bytes=capacity, source_count=32,
                                    fit_sources=sum(r['fit'] for r in subset), unfit_sources=sum(r['unfit'] for r in subset),
                                    over_budget_fraction=sum(r['unfit'] for r in subset) / 32.0,
                                    scope='source-codec feasibility; no PHY success or all-frame noisy quality claim'))
        write_csv(self.out / 'summary.csv', list(summary[0]), summary)
        files = completed + [str(self.out / p) for p in ['cases.csv', 'feasibility.csv', 'summary.csv', 'runtime.json', 'request_identity.json',
                                                              'bpgenc_help.json', 'bpgenc_help.log', 'bpgdec_help.json', 'bpgdec_help.log']]
        all_outputs = {p: sha(p) for p in files}
        for p in completed: all_outputs.update(read(p)['outputs'])
        done = dict(status='A1A_CALIBRATION32_BPG_CODEC_PROBE_COMPLETE_V1', source_count=32,
                    source_ids=[r['source_id'] for r in self.r['records']], resolutions=SIDES, tested_case_count=len(all_rows),
                    request_sha256=sha(self.rpath), new_model_calls=0, new_PHY_calls=0, holdout_content_used=False,
                    native256_experiment_unchanged=True, training_updates=0, policy_frozen=False,
                    fresh_encodes_this_invocation=self.fresh_encodes, native_stream_reuses_this_invocation=self.native_reuses,
                    fresh_decodes_this_invocation=self.fresh_decodes, outputs=all_outputs)
        save(self.out / 'completion.json', done)
        print(json.dumps({k: v for k, v in done.items() if k != 'outputs'}, sort_keys=True), flush=True)


def write_csv(path, fields, rows):
    import io
    f = io.StringIO(newline=''); w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    w.writeheader(); w.writerows(rows); immutable_bytes(path, f.getvalue().encode())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True)
    a = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, lambda s, frame: STOP.append(s))
    r = read(a.request)
    try: Probe(r, a.request).run()
    except BaseException as exc:
        out = Path(r['out']); out.mkdir(parents=True, exist_ok=True)
        save(out / ('failure_%d.json' % time.time_ns()), dict(error=type(exc).__name__, message=str(exc),
             request_sha256=sha(a.request), new_PHY_calls=0, holdout_content_used=False, completed=False))
        raise


if __name__ == '__main__':
    main()
