"""CPU verification of exact four-model/Adam restoration; no new optimizer updates."""

import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

import torch

from sufficiency.common import load_parent, load_sources, output_path, restore_systems, settings
from joint_sender.runtime import tensor_sha256
from wetok_comm.common import now, snapshot, write_json
from wetok_comm.training import module_sha256, paired_batches


def main():
    torch.set_num_threads(2)
    config, original, grid, reference, base, parent_record = settings()
    sources = load_sources(config)
    parent = load_parent(reference, base, 'cpu')
    if module_sha256(parent) != sources['frozen']['parent']:
        raise RuntimeError('the frozen original parent changed')
    systems, optimizers, restored = restore_systems(config, reference, base, sources, parent, 'cpu')
    batch = next(paired_batches(base, 20000, config['initial_global_data_step'], config['initial_global_data_step'] + 1))
    if batch['step'] != 12000:
        raise RuntimeError('R2 would restart the old data segment')
    output = output_path(config, 'preparation')
    path = output / 'resume_initialization.json'
    if path.exists():
        raise RuntimeError('CPU restoration qualification already exists')
    source_hashes = snapshot(output / 'resume_checks', [Path(__file__), EXPERIMENT / 'configs/study.yaml',
        EXPERIMENT / 'docs/protocol.md', *sorted((EXPERIMENT / 'src/sufficiency').glob('*.py')),
        JOINT / 'src/joint_sender/model.py', JOINT / 'src/joint_sender/runtime.py', GRID / 'src/grid_controls/model.py'])
    write_json(path, {'status': 'FOUR_MODEL_AND_ADAM_RESTORATION_PASS_NO_NEW_UPDATES', 'completed_local': now(),
        'restored': restored, 'source_endpoint_hashes': sources['hashes'], 'frozen': sources['frozen'],
        'first_new_global_batch_index': batch['step'], 'first_new_global_update': batch['step'] + 1,
        'first_new_batch_sha256': batch['fingerprint'], 'first_new_noise_sha256': tensor_sha256(batch['noise']),
        'inherited_selected_steps': {name: row['step'] for name, row in sources['selected'].items()},
        'inherited_calibration_files': sources['calibration_files'], 'source_hashes': source_hashes,
        'GPU_used': False, 'optimizer_updates': 0, 'research_goal_complete': False})
    print(json.dumps({'restored': restored, 'first_new_global_batch_index': batch['step'], 'output': str(path)}, indent=2))


if __name__ == '__main__':
    main()
