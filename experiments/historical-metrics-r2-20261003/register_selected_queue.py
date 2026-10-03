"""Register independent R2 provenance repair with the exact original scope."""
import argparse
from copy import deepcopy
from pathlib import Path
from audit_selected_runtime import ROSTER
from history_common import read, write, sha, source_bindings, verify
from history_controller import validate_queue


def selected_identity(audit):
    if audit.get('status') != 'PASS' or audit.get('GPU') is not False:
        raise RuntimeError('Complete CPU original-table audit is required')
    rows = audit.get('studies', [])
    if [(r.get('adapter'), r.get('study'), r.get('frames')) for r in rows] != ROSTER:
        raise RuntimeError('Constructor audit scope or ordering differs')
    fields = ('adapter', 'study', 'frames', 'sources', 'methods', 'selected_rows_sha256')
    for row in rows:
        if (row.get('status') != 'PASS' or row.get('sources') != 100
                or not row.get('methods') or not row.get('selected_rows_sha256')):
            raise RuntimeError('Incomplete original-table constructor evidence')
    return [{key:row[key] for key in fields} for row in rows]


def continuation_queue(original, source_map, *, original_path, original_sha, audit_path, audit_sha):
    """Keep all scope, comparisons, original proofs and cache admission unchanged."""
    queue = deepcopy(original)
    queue['source_bindings'] = dict(source_map)
    for path, digest in ((str(original_path), original_sha), (str(audit_path), audit_sha)):
        previous = queue['input_proof_bindings'].get(path)
        if previous is not None and previous != digest:
            raise RuntimeError('R2 evidence conflicts with an original proof binding')
        queue['input_proof_bindings'][path] = digest
    return queue


def register(root, audit_path):
    root = Path(root).resolve(); here = Path(__file__).resolve().parent
    original_path = root/'outputs/HISTORICAL-METRICS-20261003/queue_registration.json'
    original = read(original_path)
    if (original.get('status') != 'REGISTERED' or original.get('training_updates') != 0
            or original.get('policy_selection_updates') != 0):
        raise RuntimeError('Original zero-training registered queue is required')
    for field in ('source_bindings', 'shared_metric_bindings', 'input_proof_bindings'):
        verify(original[field])
    coverage = original['coverage_manifest']
    verify({coverage['path']:coverage['sha256']})
    candidates = [Path(p) for p in original['input_proof_bindings']
                  if Path(p).name == 'selected_constructor_audit_final.json']
    if len(candidates) != 1:
        raise RuntimeError('Exactly one bound original constructor audit is required')
    audit_path = Path(audit_path).resolve(); audit = read(audit_path)
    own = source_bindings(here)
    if audit.get('source_bindings') != own:
        raise RuntimeError('R2 runtime differs from the CPU-admitted source inventory')
    if selected_identity(audit) != selected_identity(read(candidates[0])):
        raise RuntimeError('R2 selected rows, source counts, methods or scope differ from v1')
    queue = continuation_queue(original, own, original_path=original_path,
        original_sha=sha(original_path), audit_path=audit_path, audit_sha=sha(audit_path))
    path = root/'outputs/HISTORICAL-METRICS-R2-20261003/queue_registration.json'
    if path.exists() and read(path) != queue:
        raise RuntimeError('Existing R2 queue differs; never overwrite active registration')
    write(path, queue)
    validate_queue(root, here, path)
    print('REGISTERED_R2', len(queue['jobs']), 'studies;',
          sum(count for _,_,count in ROSTER), 'rows;', sha(path), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--audit', required=True)
    args = parser.parse_args(); register(args.root, args.audit)
