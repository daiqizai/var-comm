"""Preserve an owned paired checkpoint before an explicitly recorded research stop."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import torch

from wetok_comm.bit_support import arm_definitions, load_repair, repair_output
from wetok_comm.common import now, sha256, verify_sources, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', required=True, type=int)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: preserve a complete paired checkpoint and interrupt only the owned bit-support trainer')
        return
    repair, base, parent, receipt = load_repair()
    output = repair_output(repair, 'training')
    destination = output / 'exploratory_early_stop'
    if destination.exists():
        raise RuntimeError('early-stop evidence already exists; inspect it instead of signalling twice')
    process = Path(f'/proc/{arguments.training_pid}')
    command = (process / 'cmdline').read_bytes().split(b'\0')
    expected = str(EXPERIMENT / 'scripts/train_bit_support.py').encode()
    if expected not in command or b'--execute' not in command or process.stat().st_uid != os.getuid():
        raise RuntimeError('PID is not the owned registered bit-support trainer')
    started = (process / 'stat').read_text().split(') ', 1)[1].split()[19]
    metadata = json.loads((output / 'metadata.json').read_text())
    verify_sources(metadata['source_hashes'])
    destination.mkdir()
    checkpoint = destination / 'paired_optimizer.pt'
    os.link(output / 'resume.pt', checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    completed = saved['completed_additional_updates']
    definitions = arm_definitions(repair)
    if completed < 1000 or completed % 250 or saved['global_step'] != repair['parent_step'] + completed:
        raise RuntimeError('latest checkpoint is not a complete registered paired boundary')
    fingerprints = []
    for name in definitions:
        history = saved['histories'][name]
        if len(history) != completed or history[-1]['additional_step'] != completed:
            raise RuntimeError('per-arm retained training budgets differ')
        if any(int(value['step']) != saved['global_step'] for value in saved['optimizers'][name]['state'].values()):
            raise RuntimeError('Adam counter differs from the retained paired history')
        fingerprints.append([row['batch_sha256'] for row in history])
    if any(value != fingerprints[0] for value in fingerprints[1:]):
        raise RuntimeError('data/channel pairing differs across retained arms')
    evidence = {'status': 'CHECKPOINT_PRESERVED_BEFORE_OWNED_SIGINT', 'decision_local': now(),
                'training_pid': arguments.training_pid, 'process_start_ticks': started,
                'retained_additional_updates_per_arm': completed, 'retained_global_step': saved['global_step'],
                'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                'original_planned_additional_updates': 5000, 'planned_milestone_completed': False,
                'reason': 'Exploratory stop after severe deterioration of all six arms at fixed calibration monitor; not a preregistered significance decision.',
                'decision_document_sha256': sha256(EXPERIMENT / 'docs/bit_support_early_stop.md'),
                'data_pairing_and_Adam_counters_verified': True, 'research_goal_complete': False,
                'uncommitted_updates_excluded_from_comparison': True}
    write_json(destination / 'termination.json', evidence)
    if (process / 'stat').read_text().split(') ', 1)[1].split()[19] != started:
        raise RuntimeError('PID identity changed; refusing to signal')
    os.kill(arguments.training_pid, signal.SIGINT)
    deadline = time.monotonic() + 120
    while process.exists() and time.monotonic() < deadline:
        try:
            if (process / 'stat').read_text().split(') ', 1)[1].split()[0] == 'Z':
                break
        except FileNotFoundError:
            break
        time.sleep(1)
    else:
        if process.exists():
            evidence['status'] = 'SIGINT_SENT_PROCESS_STILL_ALIVE_NEEDS_INSPECTION'
            write_json(destination / 'termination.json', evidence)
            raise RuntimeError('owned process did not exit after SIGINT; not escalating automatically')
    evidence.update(status='CANDIDATE_STOPPED_EARLY_NOT_COMPLETE', stopped_local=now(),
                    trainer_terminal_status=json.loads((output / 'status.json').read_text()))
    write_json(destination / 'termination.json', evidence)
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
