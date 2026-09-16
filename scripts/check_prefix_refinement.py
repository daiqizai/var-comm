#!/usr/bin/env python3
"""Run a disposable, no-update engineering check on two existing training images."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

from var_comm.next_scale_prior import state_sha256
from var_comm.prefix_learning_support import build_codec, load_frozen_training_models
from var_comm.prefix_refinement import engineering_check
from var_comm.prefix_training_data import HeaderProtocol, read_image_population
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/VAR-PREFIX-REFINEMENT-PREFLIGHT-001')
    arguments = parser.parse_args()
    config = yaml.safe_load((ROOT / 'configs/prefix_refinement.yaml').read_text())
    base = yaml.safe_load((ROOT / config['base_config']).read_text())
    previous = ROOT / config['starting_training']
    receipt = json.loads((previous / 'completion.json').read_text())
    if sha256(previous / 'completion.json') != config['starting_training_receipt_sha256']:
        raise RuntimeError('starting training receipt changed')
    verify_snapshot(receipt['source_hashes'])
    if sha256(previous / config['starting_checkpoint']) != config['starting_checkpoint_sha256']:
        raise RuntimeError('starting checkpoint changed')
    output = create_output(arguments.output_dir)
    sources = snapshot(output, [Path(__file__), ROOT / 'src/var_comm/prefix_refinement.py', ROOT / 'configs/prefix_refinement.yaml'])
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(arguments.device)
    write_json(output / 'status.json', {'status': 'ENGINEERING_CHECK_RUNNING', 'device': str(device), 'local_started': datetime.now().astimezone().isoformat()})
    vae, var, perceptual = load_frozen_training_models(base, device)
    codec = build_codec(base, vae, 'next_scale', device)
    checkpoint = torch.load(previous / config['starting_checkpoint'], map_location='cpu', weights_only=True)
    codec.load_state_dict(checkpoint['model'], strict=True)
    models = {'codec': codec, 'vae': vae, 'var': var, 'lpips': perceptual}
    before = {name: state_sha256(model) for name, model in models.items()}
    images, labels, identifiers, bindings = read_image_population('train')
    with np.load(ROOT / base['data_cache'] / 'train.npz', allow_pickle=False) as cache:
        positions = cache['source_indices'][:2]
        tokens = torch.tensor(cache['tokens'][:2, 0].astype(np.int64), device=device)
    images = images[positions].to(device).float() / 255
    labels = labels[positions].numpy()
    noise = torch.randn((2, 3060, 2), generator=torch.Generator().manual_seed(config['training']['gradient_monitor_seed']))
    decoded, valid, false = HeaderProtocol().decode(labels, np.array([4., 7.]), noise[:, :68].numpy())
    result = engineering_check(codec, tokens, torch.tensor(decoded, device=device), torch.tensor([4., 7.], device=device),
                               noise[:, 68:].to(device), torch.tensor(valid, device=device), images, vae, var, perceptual, base)
    if before != {name: state_sha256(model) for name, model in models.items()}:
        raise RuntimeError('no-update engineering check changed weights')
    verify_snapshot(sources)
    write_json(output / 'status.json', {'status': result['status'], 'device': str(device), 'local_completed': datetime.now().astimezone().isoformat()})
    write_json(output / 'check.json', {**result, 'population': 'two_existing_training_images', 'model_weights_unchanged': True,
               'source_hashes': sources, 'starting_checkpoint_sha256': config['starting_checkpoint_sha256'], 'output_hashes': artifact_hashes(output)})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
