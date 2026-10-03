"""Register R3: inherit seven completed R2 studies and run only five remaining."""
import argparse
from copy import deepcopy
from pathlib import Path
from audit_selected_runtime import ROSTER
from history_common import read, write, sha, source_bindings, verify
from history_controller import validate_queue
from phase2_completed import admit_completed, NAME, STUDY
from phase2_inheritance import admit_inherited
from historical_phase2_alias import create_adapter


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
            raise RuntimeError('R3 evidence conflicts with an original proof binding')
        queue['input_proof_bindings'][path] = digest
    return queue


def register(root, audit_path):
    root = Path(root).resolve(); here = Path(__file__).resolve().parent
    original_path = root/'outputs/HISTORICAL-METRICS-R3-20261003/queue_registration.json'
    original = read(original_path)
    if (original.get('status') != 'REGISTERED' or original.get('training_updates') != 0
            or original.get('policy_selection_updates') != 0):
        raise RuntimeError('Original zero-training registered queue is required')
    for field in ('source_bindings', 'shared_metric_bindings', 'input_proof_bindings'):
        verify(original[field])
    coverage = original['coverage_manifest']
    verify({coverage['path']:coverage['sha256']})
    candidates = []
    for path in original['input_proof_bindings']:
        if Path(path).suffix != '.json':
            continue
        value = read(path)
        if (isinstance(value, dict) and value.get('status') == 'PASS'
                and value.get('source_bindings') == original['source_bindings'] and 'studies' in value):
            candidates.append(Path(path))
    if len(candidates) != 1:
        raise RuntimeError('Exactly one bound R2 constructor audit is required')
    audit_path = Path(audit_path).resolve(); audit = read(audit_path)
    own = source_bindings(here)
    if audit.get('source_bindings') != own:
        raise RuntimeError('R3 runtime differs from the CPU-admitted source inventory')
    if selected_identity(audit) != selected_identity(read(candidates[0])):
        raise RuntimeError('R3 selected rows, source counts, methods or scope differ from R2')
    queue = continuation_queue(original, own, original_path=original_path,
        original_sha=sha(original_path), audit_path=audit_path, audit_sha=sha(audit_path))
    queue['completed_r3_studies'] = admit_completed(root,original)
    for item in queue['completed_r3_studies'].values():
        queue['input_proof_bindings'][item['receipt_path']] = item['receipt_sha256']
    parents={Path(p).parent for p in original['source_bindings']}
    if len(parents)!=1:raise RuntimeError('Original R3 runtime must be one frozen flat bundle')
    adapter=create_adapter(root,STUDY)
    inherited=admit_inherited(root,adapter,next(iter(parents)))
    inherited_path=root/'outputs'/NAME/'phase2_first81_admission.json'
    if inherited_path.exists() and read(inherited_path)!=inherited:raise RuntimeError('Earlier partial admission differs')
    write(inherited_path,inherited)
    queue['input_proof_bindings'][str(inherited_path)]=sha(inherited_path)
    queue['input_proof_bindings'].update(adapter.bindings)
    queue['replay_adapter_overrides']={STUDY:'historical_phase2_alias'}
    queue['phase2_recovery']=dict(version='PHASE2-HISTORICAL-ALIAS-20261004-R1',
        original_queue_path=str(original_path),original_queue_sha256=sha(original_path),
        inheritance_path=str(inherited_path),inheritance_sha256=sha(inherited_path),
        constructor_audit_path=str(audit_path),constructor_audit_sha256=sha(audit_path),
        inherited_sources=81,remaining_sources=19,new_metric_offset_applied=False,
        scope_changed=False,strict_grid_parity_tolerances_changed=False)
    path = root/'outputs'/NAME/'queue_registration.json'
    if path.exists() and read(path) != queue:
        raise RuntimeError('Existing R3 queue differs; never overwrite active registration')
    write(path, queue)
    validate_queue(root, here, path)
    print('REGISTERED_PHASE2_RECOVERY',len(queue['jobs']),'studies; 11 complete studies retained; Phase2 81 inherited, 19 remaining;',sha(path),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--audit', required=True)
    args = parser.parse_args(); register(args.root, args.audit)
