"""CPU-only, fail-closed inverse-AST proof of the quality resource amendment.

This is source evidence, not scientific numerical qualification. Reading these
files never imports a model, PHY runtime, torch, or historical result.
"""
import ast
import hashlib
from pathlib import Path

BASELINE_SHA = {
'shared.py':'5f8354aee84435f5eb3611d81b15c57bb1c69e40bfca42103e01047b4dd18d51',
'owner.py':'1cc1ba2a64ad73303deb992438da6fae669836228894cec0613b0e9570608ab6',
'registration_builder.py':'a33dd1197500c89ce497c0987437a7150f2ade9db66996b1909b9f08cc170b65',
'receiver_visual.py':'6fba9e6434688347184ff9188b74ae96d41a9635ac6c8d9b3c02fa8aa512fbbc',
'score_selection.py':'006b82957558ec222294e91a53cb1ba9f03f64e41ef3cfa282279d10d9c05771'}
ALLOWED_SUBSTITUTIONS = {'owner.py': [], 'receiver_visual.py': [['(16<<30)/props.total_memory,0', '(12<<30)/props.total_memory,0'], ['allocator_cap_bytes=16<<30', 'allocator_cap_bytes=12<<30']], 'score_selection.py': [['(16 << 30) / props.total_memory', '(12 << 30) / props.total_memory']], 'shared.py': [['prelaunch_free_GiB=18, runtime_free_plus_own_GiB=18, allocator_cap_GiB=16,', 'prelaunch_free_GiB=15, runtime_free_plus_own_GiB=14, allocator_cap_GiB=12,'], ['absolute_free_GiB=4, math_threads=6', 'absolute_free_GiB=2, math_threads=6'], ['def resource_device(reg, out):', 'def resource_adapter_sources(resource):\n    """Fail closed if the original current-process-only resource guard changes."""\n    assessor = inspect.getsource(resource.assess_snapshot)\n    replacements = {\n        \'free>=20*GiB if prelaunch else free>=4*GiB\': \'free>=15*GiB if prelaunch else free>=2*GiB\',\n        \'free+own>=20*GiB\': \'free+own>=14*GiB\',\n        \'0<=own<=16*GiB\': \'0<=own<=12*GiB\'}\n    for old, new in replacements.items():\n        require(assessor.count(old) == 1, \'Original assessor shape changed: \' + old)\n        assessor = assessor.replace(old, new)\n    observer = inspect.getsource(resource.own_allocator_observation)\n    require(observer.count(\'<=16*GiB\') == 1, \'Original own-allocator observation changed\')\n    observer = observer.replace(\'<=16*GiB\', \'<=12*GiB\').replace(\'frozen16GiB cap\', \'reviewed12GiB cap\')\n    return assessor, observer\n\n\ndef resource_device(reg, out):'], ["schema='R1_RAW_SHARED_RESOURCE_V1', prelaunch_free_bytes=18 << 30,\n                  runtime_free_plus_own_reserved_bytes=18 << 30)", "schema='R1_RAW_SHARED_RESOURCE_V2', prelaunch_free_bytes=15 << 30,\n                  runtime_free_plus_own_reserved_bytes=14 << 30,\n                  runtime_free_bytes=2 << 30, own_allocator_cap_bytes=12 << 30)"], ["    original = inspect.getsource(resource.assess_snapshot)\n    require(original.count('20*GiB') == 2, 'Qualified resource adapter shape changed')\n    exec(compile(original.replace('20*GiB', '18*GiB'), resource.__file__, 'exec'), namespace)", "    assessor, observer = resource_adapter_sources(resource)\n    exec(compile(assessor + '\\n' + observer, resource.__file__, 'exec'), namespace)"]], 'registration_builder.py': [['import r0_finalize\n', 'import r0_finalize_resource_v2 as r0_finalize\nimport resource_revision\n'], ["    s.checked(audit['finalizer'])", "    r0_finalize.validate_adapter_evidence(audit)\n    s.checked(audit['finalizer'])"], ['    review = s.readpin(review_pin)', '    revision_proof = resource_revision.verify()\n    review = s.readpin(review_pin)'], ["review['status'] == 'R1_IMPLEMENTATION_REVIEW_PASS'", "review['status'] == 'R1_RESOURCE_V2_IMPLEMENTATION_REVIEW_PASS'"], ["and review['scientific_calls'] == 0 and review['caps'] == s.contract.R1_CAPS,", "and review['scientific_calls'] == 0 and review['caps'] == s.contract.R1_CAPS\n              and review['resource_policy'] == s.RESOURCE\n              and review['resource_revision_proof'] == revision_proof,"], ["'receiver_visual.py', 'score_selection.py')", "'receiver_visual.py', 'score_selection.py',\n             'resource_revision.py', 'dispatch_once.py')"], ["s.EXPERIMENT/'policy_selection.py', s.EXPERIMENT/'r0_finalize.py'):", "s.EXPERIMENT/'policy_selection.py', s.EXPERIMENT/'r0_finalize.py',\n                 Path(r0_finalize.__file__)):"], ['caps=s.contract.R1_CAPS, numeric_flags=s.FLAGS, resource_policy=s.RESOURCE, automatic_retry=False,', 'caps=s.contract.R1_CAPS, numeric_flags=s.FLAGS, resource_policy=s.RESOURCE,\n        resource_revision=revision_proof, automatic_retry=False,']]}

def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree_sha(source):
    return sha(ast.dump(ast.parse(source), include_attributes=False).encode('utf8'))


def verify(directory=None):
    directory = Path(__file__).parent if directory is None else Path(directory)
    old = directory.with_name('r1_execution_v1')
    records = {}
    for name, expected_sha in BASELINE_SHA.items():
        before = (old/name).read_bytes()
        require(sha(before) == expected_sha, 'Frozen baseline source changed: ' + name)
        original = before.decode('utf8').replace('\r\n', '\n')
        current_bytes = (directory/name).read_bytes()
        current = current_bytes.decode('utf8').replace('\r\n', '\n')
        projected = original
        for left, right in ALLOWED_SUBSTITUTIONS[name]:
            require(projected.count(left) == 1, 'Ambiguous original substitution: ' + name)
            projected = projected.replace(left, right)
        require(current == projected, 'Undeclared change to resource revision: ' + name)
        restored = current
        for left, right in reversed(ALLOWED_SUBSTITUTIONS[name]):
            require(restored.count(right) == 1, 'Ambiguous inverse substitution: ' + name)
            restored = restored.replace(right, left)
        require(tree_sha(restored) == tree_sha(original), 'Inverse AST differs: ' + name)
        records[name] = dict(original_sha256=expected_sha, revised_sha256=sha(current_bytes),
            original_AST_sha256=tree_sha(original), inverse_restored_AST_sha256=tree_sha(restored),
            permitted_substitutions=len(ALLOWED_SUBSTITUTIONS[name]))
    return dict(status='EXACT_RESOURCE_AND_CLOSURE_BINDING_REVISION_PASS', source_records=records,
        prelaunch_free_GiB=15, runtime_free_plus_own_GiB=14, allocator_cap_GiB=12,
        absolute_free_GiB=2, scientific_calls=0, mathematical_code_changed=False,
        source_free_RX_projection_changed=False, timing_v6_resource_changed=False,
        qualification='CPU source proof only; no promise of bitwise equivalence or that every real frame fits')
