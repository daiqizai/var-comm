"""Isolated frozen-asset adapter for the two authorized October 2 studies."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = 'SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
OUT = ROOT / 'outputs' / RUN
RESULT = ROOT / 'results/scale_causal_partial_residual_20261002'
OLD = ROOT / 'experiments/extreme-bandwidth-20261001-N1024'
sys.path.insert(0, str(OLD))
import assets
import digital as legacy
import numpy as np
import torch
import yaml
from var_comm.quality import dino_features
SNRS = [1, 4, 7, 13, 19]
CAL_SEEDS = [4101, 4102, 4103]
DEV_SEEDS = [2001, 2002, 2003]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(part)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(t, p)

def seal(path, obj):
    p = Path(path)
    if p.exists():
        if read(p) != obj: raise RuntimeError('Immutable identity changed: ' + str(p))
    else: write(p, obj)

def csv_rows(path, rows): legacy.csv_rows(path, rows)
def read_csv(path):
    with Path(path).open(newline='') as h: return list(csv.DictReader(h))
def check(): assets.check()
def setup(): return assets.setup()
def assert_frozen(loaded): assets.assert_frozen(loaded)
def status(stage, **kw): write(OUT / (stage + '_status.json'), dict(stage=stage, time=time.time(), pid=os.getpid(), **kw))

@torch.no_grad()
def data(role, loaded):
    if role == 'development': return legacy.development_data(loaded)
    if role != 'calibration': raise ValueError('Only original calibration/development populations')
    d = assets.load_calibration()
    refs = []
    for r in d['records']:
        check()
        x = torch.as_tensor(r['pixels'][None], dtype=torch.float32, device=loaded['device']) / 255.
        refs.append(dino_features(loaded['dino'], x)[0].cpu())
    d['reference'] = torch.stack(refs)
    return d

def split_tokens(t): return legacy.split_tokens(np.asarray(t, dtype=np.int64))
def prefix_latent(loaded, prefix): return legacy.prefix_latent(loaded['vae'], prefix).cpu()

@torch.no_grad()
def quality(loaded, record, latent, clean, reference=None, misreference=None, images=False):
    if latent is not None: latent = latent.detach().cpu()
    scored, image = legacy.quality_outputs(loaded, record, {'output': (latent, 'Dc')}, reference, misreference, images)['output']
    return dict(**scored, **legacy.latent_fields(clean, latent)), image

def source_bindings():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.json')) + list(HERE.glob('*.md'))
    # Freeze tracked repository Python too: the old asset adapter has several
    # transitive experiment imports, not just the directly imported entry files.
    tracked = subprocess.check_output(['git','ls-files','*.py','*.cpp','configs/*.yaml','configs/*.json'],cwd=ROOT,text=True).splitlines()
    paths += [ROOT/p for p in tracked]
    paths += list(OLD.glob('*.py')) + [ROOT/'src/var_comm/next_scale_prior.py', ROOT/'src/var_comm/scale_channel.py', ROOT/'src/var_comm/token_trellis.cpp', ROOT/'src/var_comm/study.py', ROOT/'src/var_comm/quality.py']
    paths += [ROOT/'experiments/rx-posterior-step1-20260929'/n for n in ['probe.py','rx_v3_common.py','run_preflight.py']]
    paths += [ROOT/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py']
    vp = Path(assets.old.b.model_paths()['var_source'])
    paths += list((vp/'models').glob('*.py'))
    quality_config=ROOT/'configs/progressive_channel.yaml'
    dp=Path(yaml.safe_load(quality_config.read_text())['quality']['dino_source'])
    paths += list((dp/'dinov2').rglob('*.py'))
    return {str(p): sha(p) for p in paths}

def verify_bindings(bindings):
    for p, h in bindings.items():
        if sha(p) != h: raise RuntimeError('Bound source changed: '+p)

def registration(loaded, d, stage, inputs=None):
    cfg = dict(stage=stage, identity=loaded['identity'], source_bindings=source_bindings(),
               source_ids=[r['image_id'] for r in d['records']],
               preprocessing_ids=[r['preprocessing_id'] for r in d['records']],
               data_bindings=d.get('bindings',{}),input_artifacts={str(p):sha(p) for p in (inputs or [])},
               calibration_or_development=stage, training_updates=0,
               protocol_sha256=sha(HERE/'protocol.json'))
    p=OUT/(stage+'_registration.json');seal(p,cfg)
    return sha(p)

def save_image(path, image):
    from PIL import Image
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.clip(image.transpose(1,2,0)*255,0,255).astype(np.uint8)).save(p)
