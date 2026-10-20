"""Fresh frozen four-resolution BPG search, with real CLI time included."""
import subprocess
import tempfile
import shutil
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image

SIDES = (256, 128, 64, 32)
COARSE = (0, 8, 16, 24, 32, 40, 48, 51)
OPTIONS = ('-e', 'x265', '-m', '8', '-f', '420', '-c', 'ycbcr', '-b', '8')
_THREAD_MESSAGES = set()


@contextmanager
def workspace(root):
    """Clean successful transient codec files; preserve any failed real call."""
    from contract import T
    path = Path(tempfile.mkdtemp(dir=root))
    try:
        yield str(path)
    except BaseException:
        evidence = T/'BPG/failed_codec_evidence'/path.name
        evidence.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, evidence)
        raise
    else:
        shutil.rmtree(path)


def png_bytes(a):
    return legacy().png_bytes(a)


@lru_cache(maxsize=1)
def legacy():
    from native import load
    from contract import T
    return load(T/'assets/bpg_support/probe_source_codec.py', '_timing_original_BPG_pixel_io')


def run(argv, allow_malformed=False):
    p = subprocess.run(argv, capture_output=True, timeout=120)
    for line in (p.stdout+p.stderr).decode(errors='replace').splitlines():
        if re.search(r'thread|pool', line, re.IGNORECASE):
            _THREAD_MESSAGES.add(line)
    if p.returncode == 1 and allow_malformed:
        text = (p.stdout+p.stderr).decode(errors='replace')
        assert 'Could not decode image' in text and 'I/O error' not in text and 'Error while reading file' not in text
        return False
    assert p.returncode == 0, (argv, p.returncode, p.stderr.decode(errors='replace'))
    return True


def read_rgb(path, expected_side=None):
    with Image.open(path) as im:
        assert im.mode == 'RGB' and im.size[0] == im.size[1] and im.size[0] in SIDES
        if expected_side is not None:
            assert im.size == (expected_side, expected_side)
        side = im.size[0]
    return legacy().ppm_rgb(path, side)


def restore(a):
    return a if len(a) == 256 else np.asarray(Image.fromarray(a).resize((256, 256), Image.Resampling.BICUBIC))


def encode(pixels, cfg, capacity, temp_root):
    """No cross-call cache. Fine search uses ALL frozen catalogue capacities."""
    _THREAD_MESSAGES.clear()
    rgb = np.ascontiguousarray(pixels.transpose(1, 2, 0))
    capacities = sorted({p['capacity_bytes'] for p in cfg['catalogue']})
    rows = []
    with workspace(temp_root) as name:
        d = Path(name)
        for side in SIDES:
            original = rgb if side == 256 else np.asarray(Image.fromarray(rgb).resize((side, side), Image.Resampling.BICUBIC))
            src = d / f'r{side}.png'
            src.write_bytes(png_bytes(original))
            def candidate(qp):
                target, decoded = d/f'r{side}q{qp}.bpg', d/f'r{side}q{qp}.ppm'
                run([cfg['bpgenc']['path'], *OPTIONS, '-q', str(qp), '-o', str(target), str(src)])
                run([cfg['bpgdec']['path'], '-o', str(decoded), str(target)])
                image = restore(read_rgb(decoded, side))
                mse = float(np.mean(((image.astype(np.float64)-rgb.astype(np.float64))/255.0)**2))
                data = target.read_bytes()
                return dict(qp=qp, resolution=side, complete_BPG_bytes=len(data),
                            mse=mse, data=data)
            measured = {q: candidate(q) for q in COARSE}
            fine = set()
            for left, right in zip(COARSE, COARSE[1:]):
                lo, hi = sorted([measured[left]['complete_BPG_bytes'],
                                 measured[right]['complete_BPG_bytes']])
                if any(lo <= c < hi for c in capacities):
                    fine.update(range(left+1, right))
            for q in sorted(fine):
                measured[q] = candidate(q)
            rows.extend(measured[q] for q in sorted(measured))
        assert len(rows) <= 208
        fits = [v for v in rows if v['complete_BPG_bytes'] <= capacity]
        selected = min(fits, key=lambda v: (v['mse'], v['complete_BPG_bytes'], -v['resolution'], v['qp'])) if fits else None
        return (None if selected is None else selected['data']), dict(
            candidate_encodes=len(rows), candidate_decodes=len(rows),
            selected=None if selected is None else {k: v for k, v in selected.items() if k != 'data'},
            all_capacity_boundaries=capacities, cached_source_calls=0,
            native_thread_messages=sorted(_THREAD_MESSAGES),
            thread_policy='Original x265 defaults inside inherited six-CPU affinity; OMP/MKL6 is not asserted to be an x265 thread count')


def decode(payload, cfg, temp_root):
    with workspace(temp_root) as name:
        d = Path(name)
        src, target = d/'received.bpg', d/'received.ppm'
        src.write_bytes(payload)
        if not run([cfg['bpgdec']['path'], '-o', str(target), str(src)], allow_malformed=True):
            return None, 'BPG_DECODER_REJECT'
        try:
            rgb = restore(read_rgb(target))
        except (ValueError, OSError):
            return None, 'BPG_SOURCE_FORMAT_REJECT'
        return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)/np.float32(255), 'BPG_DECODED'
