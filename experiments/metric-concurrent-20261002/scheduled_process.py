"""Apply registered scheduling permission, then run the frozen M2 wrapper."""
import argparse
from pathlib import Path
import runpy
import sys
from scheduling_guard import install_admission, read, sha, verify, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--admission', required=True)
    p.add_argument('--stage', required=True, choices=('calibration','evaluation','actual','timing'))
    a = p.parse_args(); root = Path(a.root).resolve()
    target = root / 'experiments/metric-speed-20261002/wrapper.py'
    admission = read(a.admission)
    verify(admission['original_execution_bindings'])
    if admission['original_execution_bindings'].get(str(target)) != sha(target):
        raise RuntimeError('Original M2 wrapper was not admitted')
    record = install_admission(root, a.admission, role='m2', exclusive=a.stage == 'timing')
    write(root / 'outputs/METRIC-CONCURRENT-R4-20261002' / ('m2_' + a.stage + '_execution.json'),
        dict(status='SCHEDULING_ONLY_EXECUTION', stage=a.stage, **record,
             original_script=str(target), original_script_sha256=sha(target),
             exclusive_timing=a.stage == 'timing', training_updates=0, policy_selection_updates=0))
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), '--stage', a.stage]
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__': main()
