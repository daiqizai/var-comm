"""CPU-only ancestry adaptation for R1 scoring after the diagnosed visual failure.

The original master and completed projection retain their original identities.
Only three metadata statements in the frozen scorer change; metric and selection
functions, input identities, caches, budgets and mathematical expressions do not.
This module is not a launcher and cannot resume an experiment by itself.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
OLD_ROOT = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013/r1_execution_v1')
ORIGINAL_REGISTRATION = dict(path=str(ROOT/'results/layered_mismatch_raw_20261013/r1_execution_v1/registered/registration.json'),
    sha256='70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47')
SCORE_SHA = '1bd5d9b016ed9d4bd8044acf67942a2fc06bd0e23450fda5d04039543ff48947'
SCORE_SUFFIX = ('experiments', 'layered-mismatch-raw-20261013', 'r1_execution_resource_v2', 'score_selection.py')
SCIENTIFIC_FIELDS = ('plan_id', 'namespace', 'caps', 'expected_physical_frames',
    'expected_logical_var_images', 'models', 'numeric_flags', 'metric_names', 'metric_config',
    'resource_policy', 'training_updates', 'additional_qualification_frames', 'bootstrap_calls',
    'automatic_retry', 'automatic_R2')
PIN_FIELDS = ('cal200_sources', 'common_ledger', 'grouped_catalogue', 'nominations', 'original_R0_audit')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def same_pin(a, b):
    return isinstance(a, dict) and isinstance(b, dict) and a.get('path') == b.get('path') and a.get('sha256') == b.get('sha256')


def original_master(reg):
    """Read only the exact frozen master, never source pixels or result quality."""
    value = reg['recovery']['original_registration']
    require(same_pin(value, ORIGINAL_REGISTRATION), 'Unexpected projection ancestor registration')
    path = Path(value['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and sha(path) == ORIGINAL_REGISTRATION['sha256'], 'Original master bytes changed')
    if 'bytes' in value:
        require(type(value['bytes']) is int and value['bytes'] == path.stat().st_size, 'Original master size differs')
    old = json.loads(path.read_text(encoding='utf-8-sig'))
    require(old['schema'] == 'LAYERED_RAW_R1_EXECUTION_REGISTRATION_V1'
            and old['status'] == 'REGISTERED_R1_EXECUTION_V1', 'Original master status differs')
    require(all(reg[k] == old[k] for k in SCIENTIFIC_FIELDS), 'Recovery changed original scientific scope')
    require(old['expected_physical_frames'] == 32400 and old['expected_logical_var_images'] == 43200,
            'Different fixed R1 grid')
    require(same_pin(reg['phy_registration'], old['phy_registration'])
            and all(reg['pins'][k] == old['pins'][k] for k in PIN_FIELDS), 'Recovery changed PHY or original source bindings')
    require(Path(old['runtime']['root']) == OLD_ROOT, 'Original runtime root differs')
    for stage in ('phy', 'projection'):
        expected = str(OLD_ROOT/stage)
        require(old['runtime'][stage+'_directory'] == expected == reg['runtime'][stage+'_directory'],
                'Completed stage must retain its original directory')
    for key in ('deadline_unix', 'max_seconds', 'python_cpu', 'python_visual', 'target_gpu',
                'cpu_affinity', 'phy_cpu_affinity'):
        require(reg['runtime'][key] == old['runtime'][key], 'Recovery changed original deadline or runtime: '+key)
    return old


def ancestor_for_wait(reg, registration, stage):
    if stage != 'projection':
        return reg, registration
    old = original_master(reg)
    return old, dict(ORIGINAL_REGISTRATION)


def projection_registration(reg):
    original_master(reg)
    return dict(ORIGINAL_REGISTRATION)


def projection_root(reg):
    return Path(original_master(reg)['runtime']['root'])


SUBSTITUTIONS = (
    ("    require(stage in ('projection', 'visual', 'score'), 'Unsupported score prerequisite stage')",
     "    require(stage in ('projection', 'visual', 'score'), 'Unsupported score prerequisite stage')\n"
     "    reg, registration = _recovery_ancestor_for_wait(reg, registration, stage)"),
    ("            and same_pin(projection['registration'], registration)",
     "            and same_pin(projection['registration'], _recovery_projection_registration(reg))"),
    ("        projection_dispatch=pin(Path(reg['runtime']['root']) / 'projection_launch.json'),",
     "        projection_dispatch=pin(_recovery_projection_root(reg) / 'projection_launch.json'),"),
)


def adapted_source(source):
    changed = source
    for old, new in SUBSTITUTIONS:
        require(changed.count(old) == 1, 'Frozen score metadata statement changed: '+old)
        changed = changed.replace(old, new)
    inverse = changed
    for old, new in reversed(SUBSTITUTIONS):
        require(inverse.count(new) == 1, 'Ambiguous inverse metadata statement')
        inverse = inverse.replace(new, old)
    original_ast = ast.dump(ast.parse(source), include_attributes=False)
    restored_ast = ast.dump(ast.parse(inverse), include_attributes=False)
    require(original_ast == restored_ast, 'Scoring inverse AST proof failed')
    before = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    after = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(changed).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    permitted = {'successful_wait', 'closed_visual'}
    require(set(before) == set(after) and all(before[k] == after[k] for k in before.keys()-permitted),
            'Non-provenance scorer code changed')
    return changed, dict(status='PASS_EXACT_ANCESTRY_METADATA_ONLY', original_score_sha256=SCORE_SHA,
        substitutions=[dict(old=a,new=b) for a,b in SUBSTITUTIONS],
        original_AST_sha256=hashlib.sha256(original_ast.encode()).hexdigest(),
        inverse_restored_AST_sha256=hashlib.sha256(restored_ast.encode()).hexdigest(),
        unchanged_functions_and_classes=sorted(before.keys()-permitted), scientific_calls=0,
        original_projection_relabelled=False, metric_or_selection_math_changed=False)


def install(module):
    """Install only two metadata functions in an already imported private scorer."""
    path = Path(module.__file__).resolve()
    require(path.parts[-4:] == SCORE_SUFFIX and sha(path) == SCORE_SHA, 'Unbound original scorer')
    require(module.__name__ != '__main__', 'Install metadata before invoking any original CLI')
    source, proof = adapted_source(path.read_text(encoding='utf-8-sig'))
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
             and node.name in ('successful_wait', 'closed_visual')]
    require(len(nodes) == 2, 'Missing original scorer metadata functions')
    module.__dict__.update(_recovery_ancestor_for_wait=ancestor_for_wait,
        _recovery_projection_registration=projection_registration, _recovery_projection_root=projection_root)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    module.RECOVERY_SCORE_PROVENANCE_PROOF = proof
    return module
