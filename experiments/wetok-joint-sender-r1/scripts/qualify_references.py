"""Seal the completed RX experiment for later Joint comparisons, without GPU or new-model inference."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from joint_sender.evaluation import load_evaluation
from joint_sender.references import qualify_reference
from wetok_comm.common import PROJECT, now, snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, reference, base, parent = load_evaluation()
    if not arguments.execute:
        print('PLAN ONLY: completed RX5000/analysis and all source/noise/resource/image artifacts; no GPU or new-model inference')
        return
    output = PROJECT / 'outputs' / evaluation['qualification_file']
    if output.exists():
        raise FileExistsError('preserve the previous reference qualification')
    report = qualify_reference(evaluation, config, reference, base)
    sources = snapshot(output.parent / 'reference_binding', [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml',
        EXPERIMENT / 'docs/evaluation_protocol.md', EXPERIMENT / 'src/joint_sender/references.py', EXPERIMENT / 'src/joint_sender/evaluation.py'])
    report.update(completed_local=now(), source_hashes=sources)
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
