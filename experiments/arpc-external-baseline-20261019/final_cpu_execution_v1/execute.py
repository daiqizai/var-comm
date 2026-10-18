"""Finite, once-only CPU artifact execution after the actual final science gate."""
import argparse
import json
import os
import pathlib
import sys
import time
import traceback

E = pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
sys.path.insert(0, str(E))
import a1_owner as receipt

PY = '/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
ROLES = {
    'figures': ('final_figures_v1/render.py', 600),
    'report': ('final_report_v1/create.py', 300),
    'publication': ('final_publication_v1/publish.py', 2400),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=ROLES)
    parser.add_argument('--expected-entry', required=True)
    parser.add_argument('--expected-closure', required=True)
    raw = sys.argv[1:]
    assert '--' in raw, 'Separate supervisor arguments from child arguments with --'
    split = raw.index('--')
    args = parser.parse_args(raw[:split])
    relative, limit = ROLES[args.role]
    entry = E / relative
    assert receipt.desc(entry)['sha256'] == args.expected_entry
    closure = receipt.O / 'a3_metrics_v1/root_closure_review.json'
    assert receipt.desc(closure)['sha256'] == args.expected_closure
    state = json.loads(closure.read_text())
    assert state['status'] == 'EVAL100_FINAL_ANALYSIS_CLOSED' and state['all_processes_gone'] and state['unresolved'] == 0
    for pin in state['controls'] + state['outputs']:
        receipt.verify(pin)
    extra = raw[split + 1:]
    assert extra
    root = receipt.L / ('final_' + args.role + '_execution_v1')
    assert not root.exists()
    root.mkdir()
    receipt.put(root / 'supervisor.json', {
        'process': receipt.identity(os.getpid()), 'parent_process': receipt.identity(os.getppid()),
        'argv': sys.argv, 'source': receipt.desc(pathlib.Path(__file__)),
        'entry': receipt.desc(entry), 'closure': receipt.desc(closure), 'started_unix': time.time()})
    receipt.RUN = root
    os.chdir(entry.parent)
    try:
        receipt.run_stage('child', [PY, '-B', str(entry), *extra],
            dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1'), limit)
        receipt.put(root / 'completion.json', {'status': 'PASS', 'actual_wait': receipt.desc(root / 'child.actual_wait.json')})
    except BaseException:
        receipt.put(root / 'failure.json', {'error': traceback.format_exc(), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
