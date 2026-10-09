"""Complete-file adaptive BPG source coding and received-byte-only restoration."""
from __future__ import annotations
import hashlib
from pathlib import Path
import time
import numpy as np
from PIL import Image
from probe_source_codec import Probe, COARSE, SIDES, png_bytes, ppm_rgb, immutable_bytes
from bpg_codec import read, require, sha, verify_outputs, write


def choice(rows, capacity):
    fits = [r for r in rows if r['complete_BPG_bytes'] <= capacity]
    return min(fits, key=lambda r: (r['mse'], r['complete_BPG_bytes'], -r['resolution'], r['qp'])) if fits else None


def make_probe(cfg, population, out):
    original = read(cfg['original_request']['path'])
    probe = dict(out=str(out), original_request=cfg['original_request'],
                 native_codec_completion=cfg['original_native_codec_completion'][population],
                 native_codec_root=cfg['original_native_root'] + '/' + population,
                 records=cfg['records'][population], bpgenc=cfg['bpgenc'], bpgdec=cfg['bpgdec'],
                 deadline_unix=cfg['deadline_unix'], max_seconds=max(1, cfg['deadline_unix'] - time.time()),
                 stop_files=cfg['stop_files'], codec_timeout_seconds=cfg['codec_timeout_seconds'],
                 capacity_bytes=sorted({p['capacity_bytes'] for p in cfg['catalogue']}))
    require(original['bpgenc'] == cfg['bpgenc'] and original['bpgdec'] == cfg['bpgdec'], 'Original codec changed')
    return Probe(probe, cfg['_path'])


def source_rows(cfg, population, index, rgb_chw):
    record = cfg['records'][population][index]
    out = Path(cfg['out']) / 'codec' / population / ('%04d' % index)
    cp = out / 'completion.json'
    if cp.exists():
        done = read(cp); verify_outputs(done)
        require(done['request_sha256'] == cfg['_sha256'] and done['source_id'] == record['source_id'], 'Changed adaptive codec source')
        return done
    if population == 'calibration' and index < 32:
        old_path = Path(cfg['probe_root']) / 'sources' / ('%04d' % index) / 'completion.json'
        old = read(old_path); verify_outputs(old)
        top = read(cfg['probe_completion']['path'])
        require(top['outputs'][str(old_path)] == sha(old_path) and old['source'] == record,
                'Actual A1a source does not match calibration source')
        rows = old['rows']
        outputs = {str(old_path): sha(old_path)}
        # Keep the old output directory untouched and retain original provenance.
        outputs.update(old['outputs'])
        origin = 'actual_A1a_source_cases_reused'
    else:
        probe = make_probe(cfg, population, out)
        rgb = np.ascontiguousarray(rgb_chw.transpose(1, 2, 0))
        rows, outputs = [], {}
        for side in SIDES:
            directory = out / ('r%d' % side); directory.mkdir(parents=True, exist_ok=True)
            resized = rgb if side == 256 else np.asarray(Image.fromarray(rgb).resize((side, side), Image.Resampling.BICUBIC))
            source = directory / 'source.png'; immutable_bytes(source, png_bytes(resized))
            measured = {q: probe.case(directory, source, side, index, q, rgb) for q in COARSE}
            fine = set()
            for left, right in zip(COARSE, COARSE[1:]):
                low, high = sorted([measured[left]['complete_BPG_bytes'], measured[right]['complete_BPG_bytes']])
                if any(low <= c < high for c in probe.r['capacity_bytes']):
                    fine.update(range(left + 1, right))
            for q in sorted(fine):
                measured[q] = probe.case(directory, source, side, index, q, rgb)
            rows.extend(measured[q] for q in sorted(measured))
            for q, value in measured.items():
                path = directory / ('qp%02d.case.json' % q)
                outputs[str(path)] = sha(path); outputs.update(value['outputs'])
        origin = 'new_actual_source_codec_cases_with_available_native256_stream_reuse'
    require(len(rows) <= 4 * 52 and {r['resolution'] for r in rows} == set(SIDES), 'Incomplete source candidate dimensions')
    fits = {}
    for profile in cfg['catalogue']:
        selected = choice(rows, profile['capacity_bytes'])
        fits[str(profile['profile_id'])] = {'status': 'FIT' if selected else 'SOURCE_UNFIT',
                                           'capacity_bytes': profile['capacity_bytes'], 'selected': selected}
    done = dict(status='ADAPTIVE_BPG_SOURCE_CODEC_COMPLETE_V1', source_index=index, source_id=record['source_id'],
                preprocessing_id=record['preprocessing_id'], request_sha256=cfg['_sha256'],
                fits=fits, rows=rows, source_case_origin=origin, outputs=outputs)
    write(cp, done)
    return done


class ReceiverCodec:
    """Only received bytes enter this decoder; source references never enter it."""
    def __init__(self, cfg, population):
        self.cfg = cfg
        # Separate workers avoid racing on a content-identical stream; within a
        # worker the real received-byte cache is reused without re-decoding.
        self.out = Path(cfg['out']) / 'received_codec' / population / ('worker_%d' % cfg.get('_worker', 0))
        self.probe = make_probe(cfg, population, self.out)

    def decode(self, received_bytes):
        key = hashlib.sha256(received_bytes).hexdigest()
        directory = self.out / key
        directory.mkdir(parents=True, exist_ok=True)
        cp, stream, ppm = directory / 'receipt.json', directory / 'received.bpg', directory / 'decoded.ppm'
        argv = [self.cfg['bpgdec']['path'], '-o', str(ppm), str(stream)]
        if cp.exists():
            done = read(cp); verify_outputs(done)
            require(done['argv'] == argv and done['received_bytes_sha256'] == key
                    and done['decoder_sha256'] == self.cfg['bpgdec']['sha256'] and done['process_waited'],
                    'Received codec cache changed')
        else:
            require(not stream.exists() and not ppm.exists(), 'Unreceipted BPG decoder output retained')
            immutable_bytes(stream, received_bytes)
            # libbpg uses status1 for actual malformed streams. Any other status
            # or an I/O failure is an execution failure, not an image outcome.
            done = self.probe.call(argv, directory / 'process.log', allow_help=True)
            status, size = 'BPG_DECODED', None
            if done['exit_code'] == 1:
                text = Path(done['log']).read_text(errors='replace')
                require('Could not decode image' in text and 'I/O error' not in text
                        and 'Error while reading file' not in text, 'Unexpected decoder process failure')
                status = 'BPG_DECODER_REJECT'
            else:
                try:
                    with Image.open(ppm) as im:
                        size = list(im.size)
                        require(im.mode == 'RGB' and size[0] == size[1] and size[0] in SIDES,
                                'Unsupported received image format')
                    ppm_rgb(ppm, size[0])
                except (ValueError, OSError):
                    status = 'BPG_SOURCE_FORMAT_REJECT'
            done.update(received_bytes_sha256=key, decoder_sha256=self.cfg['bpgdec']['sha256'],
                        status=status, decoded_size=size, outputs={str(stream): sha(stream), done['log']: done['log_sha256']})
            if ppm.exists():
                done['outputs'][str(ppm)] = sha(ppm)
            write(cp, done)
        if done['status'] != 'BPG_DECODED':
            return None, done['status'], str(cp)
        side = done['decoded_size'][0]
        decoded = ppm_rgb(ppm, side)
        restored = decoded if side == 256 else np.asarray(Image.fromarray(decoded).resize((256, 256), Image.Resampling.BICUBIC))
        image = np.ascontiguousarray(restored.transpose(2, 0, 1), dtype=np.float32) / np.float32(255)
        return image, 'BPG_DECODED', str(cp)
