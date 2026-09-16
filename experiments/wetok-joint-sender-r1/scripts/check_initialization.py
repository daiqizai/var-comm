"""Verify production-weight initialization against the live study's frozen initial manifest on CPU."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from joint_sender.common import initial_system, load_parent, output_path, settings
from wetok_comm.common import PROJECT, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.training import module_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: common initial model hashes and trainable parameter policy, CPU only; not final control qualification')
        return
    configure_torch()
    config, reference, base, parent_milestone = settings()
    training = PROJECT / 'outputs' / config['reference_training']
    metadata = json.loads((training / 'metadata.json').read_text())
    verify_sources(metadata['source_hashes'])
    initial_path = training / 'initialization.json'
    initial = json.loads(initial_path.read_text())
    parent = load_parent(reference, base, 'cpu')
    rows = {}
    for variant in config['variants']:
        system = initial_system(parent, variant, reference, 'cpu')
        digest = module_sha256(system)
        if digest != initial['model_hashes'][variant] or module_sha256(system.encoder) != initial['frozen']['encoder']:
            raise RuntimeError('joint and receiver-only starting weights differ')
        rows[variant] = {'model_sha256': digest, 'frozen_RX_control_trainable': initial['trainable_parameters'][variant],
            'joint_communication_trainable': sum(value.numel() for value in system.parameters() if value.requires_grad),
            'sender_newly_trainable': sum(value.numel() for value in system.encoder.parameters() if value.requires_grad)}
    output = output_path(config, 'preparation')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml',
        *sorted((EXPERIMENT / 'src/joint_sender').glob('*.py'))])
    report = {'status': 'JOINT_SENDER_INITIAL_WEIGHT_POLICY_CPU_CHECK_PASS', 'completed_local': now(), 'models': rows,
        'initial_manifest_sha256': sha256(initial_path), 'parent_checkpoint_sha256': reference['parent_model_sha256'],
        'source_hashes': sources, 'final_5000_control_qualification_done': False,
        'optimizer_updates': 0, 'GPU_used': False, 'new_development_access': False, 'research_goal_complete': False}
    write_json(output / 'initialization.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
