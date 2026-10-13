"""Finish only missing registration metadata after the documented zero-call stop.

The existing nominations and PHY registration stay byte-identical. The original
absolute deadline is inherited, never restarted. No model or PHY execution.
"""
from __future__ import annotations
import argparse
import ast
import copy
from pathlib import Path
import sys
import time
import types

import pin_adapter as a

HERE = Path(__file__).resolve().parent
RESOURCE = HERE.parent / 'r1_execution_resource_v2'
BUILDER = RESOURCE/'registration_builder.py'
BUILDER_SHA = 'a83457f7ac430d39128018f0c333ea61a71ecd0f98862f5a63b418cf8d606475'
PARTIAL = {
    'nominations.json': '498572dc2cacb8772ffe86e77c02e9a90027a3fe8bba5c37106821f2b237bfba',
    'phy_registration.json': '19207028a83c96e0b99bb1aa3109b2f9a5ff22f0d557132fcde8899dd19749e2',
}
DEADLINE = 1791954813.2221007
MAX_SECONDS = 43200
FAILURE_SHA = '578be63f7a3a28bc01985cdbb6d636f022235cfebcc1577b1325ff15af4d6edd'
REPLACEMENTS = (
    ("s.require(directory == s.OUT / 'registered' and not directory.exists(), 'Fresh authoritative registration directory required')",
     '_verify_partial(directory)'),
    ('deadline_unix=time.time() + max_seconds', 'deadline_unix=_original_deadline'),
    ('bindings.update(review[\'code_bindings\'])', "bindings.update(review['code_bindings'])\n    bindings.update(_additional_bindings)"),
    ('directory.mkdir(parents=True)', '_verify_partial_directory_again(directory)'),
)


def adapted_builder_source(text):
    changed = text
    for old, new in REPLACEMENTS:
        a.require(changed.count(old) == 1 and new not in changed, 'Original builder shape differs: ' + old)
        changed = changed.replace(old, new)
    inverse = changed
    for old, new in reversed(REPLACEMENTS):
        inverse = inverse.replace(new, old)
    a.require(ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(text)), 'Builder inverse AST differs')
    return changed


def proof():
    a.require(a.sha(BUILDER) == BUILDER_SHA, 'Original resource builder differs')
    adapted_builder_source(BUILDER.read_text(encoding='utf-8-sig'))
    return dict(original_builder=dict(path=str(BUILDER), sha256=BUILDER_SHA),
                partial_file_sha256=PARTIAL, original_deadline_unix=DEADLINE,
                original_max_seconds=MAX_SECONDS, inverse_AST_equal=True,
                substitutions=[dict(old=x, new=y) for x, y in REPLACEMENTS],
                existing_files_rewritten=False, science_calls=0)


def verify_partial(directory, s):
    directory = Path(directory)
    a.require(directory == s.OUT/'registered' and directory.is_dir() and not directory.is_symlink(),
              'Original partial registration directory required')
    a.require({p.name for p in directory.iterdir()} == set(PARTIAL), 'Only the original two partial files are permitted')
    for name, expected in PARTIAL.items():
        path = directory/name
        a.require(path.is_file() and not path.is_symlink() and a.sha(path) == expected,
                  'Original partial file changed: ' + name)
    phy = s.read(directory/'phy_registration.json')
    a.require(phy['runtime']['deadline_unix'] == DEADLINE
              and phy['runtime']['max_seconds'] == MAX_SECONDS
              and phy['runtime']['cpu_affinity'] == [2, 3], 'Original finite runtime differs')
    a.require(time.time() < DEADLINE, 'Original deadline exhausted')
    a.require(not any((s.TMP/name).exists() for name in
              ('owner_launch.json', 'owner_failure.json', 'completion.json', 'phy', 'projection', 'visual', 'score', 'selection')),
              'R1 already attempted; no registration recovery')
    dispatch = s.TMP.parent/'r1_resource_dispatch_v2'
    a.require(not any((dispatch/name).exists() for name in
              ('launch.json', 'owner_identity.json', 'actual_wait.json', 'failure.json')),
              'Original dispatch already attempted')
    return phy


def immutable_or_new_write(s, directory, original_write, metadata):
    def write(path, value):
        path = Path(path)
        if path.parent == directory and path.name in PARTIAL:
            a.require(a.sha(path) == PARTIAL[path.name], 'Partial file changed before verification')
            expected = (s.canonical(value)+'\n').encode('utf8')
            a.require(path.read_bytes() == expected, 'Reconstructed original partial bytes differ')
            return s.pin(path)
        a.require(path.parent == directory and path.name in
                  ('model_bundle.json', 'visual_registration.json', 'registration.json'),
                  'Unexpected registration recovery write')
        if path.name == 'registration.json':
            value = copy.deepcopy(value)
            value['metadata_identity_revision'] = metadata
            value['pins']['metadata_adapter_review'] = metadata['review']
            value['pins']['metadata_failure_audit'] = metadata['original_failure']
        return original_write(path, value)
    return write


def recover(audit_pin, original_review_pin, adapter_review_pin, directory):
    a.install_phy()
    import shared as s
    directory = Path(directory).absolute()
    verify_partial(directory, s)
    adapter_proof, recovery_proof = a.proof(), proof()
    review = s.readpin(adapter_review_pin)
    a.require(review['status'] == 'R1_METADATA_IDENTITY_ADAPTER_REVIEW_PASS'
              and review['approved'] is True and review['scientific_calls'] == 0
              and review['adapter_proof'] == adapter_proof and review['recovery_proof'] == recovery_proof,
              'Independent metadata adapter review required')
    code = {str(HERE/name): a.sha(HERE/name) for name in
            ('pin_adapter.py', 'runner.py', 'recover_registration.py', 'test_metadata_revision.py')}
    a.require(review['code_bindings'] == code, 'Reviewed adapter source bindings differ')
    failure = dict(path=str(s.OUT.parent/'r1_resource_revision_v2/metadata_failure_audit.json'), sha256=FAILURE_SHA)
    s.checked(failure)
    metadata = dict(review=adapter_review_pin, original_failure=failure,
                    adapter_proof=adapter_proof, recovery_proof=recovery_proof,
                    explicit_runner=str(HERE/'runner.py'), no_new_scientific_calls=True)
    namespace = {'__name__': '_explicit_R1_recovery_builder', '__file__': str(BUILDER),
                 '_verify_partial': lambda p: verify_partial(p, s), '_original_deadline': DEADLINE,
                 '_verify_partial_directory_again': lambda p: verify_partial(p, s),
                 '_additional_bindings': code}
    exec(compile(adapted_builder_source(BUILDER.read_text(encoding='utf-8-sig')), str(BUILDER), 'exec'), namespace)
    # This affects this private builder namespace only; original shared module,
    # receiver bundle builder and all frozen source files remain unchanged.
    private_s = types.ModuleType('_R1_recovery_shared'); private_s.__dict__.update(vars(s))
    private_s.write = immutable_or_new_write(s, directory, s.write, metadata)
    namespace['s'] = private_s
    result = namespace['build'](audit_pin, original_review_pin, directory, 3, MAX_SECONDS)
    for name, expected in PARTIAL.items():
        a.require(a.sha(directory/name) == expected, 'Original partial bytes changed during recovery')
    master = s.readpin(result['registration'])
    a.require(master['runtime']['deadline_unix'] == DEADLINE
              and master['runtime']['max_seconds'] == MAX_SECONDS, 'Deadline was reset')
    result['metadata_identity_revision'] = metadata
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('r0-audit', 'original-review', 'adapter-review'):
        p.add_argument('--'+name, required=True); p.add_argument('--'+name+'-sha256', required=True)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    pin = lambda n: dict(path=getattr(args,n), sha256=getattr(args,n+'_sha256'))
    result = recover(pin('r0_audit'), pin('original_review'), pin('adapter_review'), args.directory)
    import json
    print(json.dumps(result, sort_keys=True))
