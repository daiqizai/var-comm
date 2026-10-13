"""Read-only R0 finalizer adapter for the declared quality-resource revision.

The original audit and ranking function bodies are reused unchanged. Only the
clean registration validator is bound to the separately reviewed resource-v2
entry, and a metadata descriptor is added when the original closure is saved.
Importing this module does not audit results, register work or call a model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import types

import r0_finalize as original
import r0_clean_resource_v2 as revised_clean

ORIGINAL_SHA = 'fe3ab159c5f24fc3b78bb1f62d0bb316ad2435a7e0a029afe8e956ae6d3ce63e'
CLEAN_SHA = 'a06240e23aa8206f36aa6fa2a904efe023994f8fa46e4e8a334957fb70023045'
CLEAN_REGISTRATION_SHA = '8d6517b0c7b12b6f149b8c99cd36bb013aedc408da2c2c91eedbfc188fb79dfc'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/layered_mismatch_raw_20261013'
CLEAN_REGISTRATION = OUT / 'r0_clean_resource_revision_v2/registration.json'
NOMINATION_OUTPUT = OUT / 'r0_nomination_v1'
SCHEMA = original.SCHEMA
ADAPTER_SCHEMA = 'R0_AUTHENTICATED_RESOURCE_VALIDATOR_ADAPTER_V2'
require = original.require
sha = original.sha
pin = original.pin
checkpin = original.checkpin
readpin = original.readpin
audit_bler = original.audit_bler


def source_bindings():
    """Metadata hashes only; never invokes either scientific registration."""
    require(sha(original.__file__) == ORIGINAL_SHA, 'Original read-only finalizer changed')
    require(sha(revised_clean.__file__) == CLEAN_SHA, 'Reviewed clean resource entry changed')
    return dict(original_finalizer=pin(original.__file__), clean_entry=pin(revised_clean.__file__))


def check_clean_registration(value):
    source_bindings()
    require(Path(value['path']).absolute() == CLEAN_REGISTRATION.absolute()
            and value['sha256'] == CLEAN_REGISTRATION_SHA,
            'Only the actual resource-v2 clean registration is admitted')
    checkpin(value)
    return value


def adapter_evidence():
    binding = source_bindings()
    registration = pin(CLEAN_REGISTRATION)
    check_clean_registration(registration)
    return dict(schema=ADAPTER_SCHEMA, **binding,
                actual_clean_registration=registration,
                resource_policy=dict(revised_clean.RESOURCE_POLICY),
                clean_validator='r0_clean_resource_v2.registration',
                audit_function_bodies_unchanged=True,
                nomination_algorithm_unchanged=True,
                source_grid_and_scientific_caps_unchanged=True,
                scientific_calls=0, bootstrap_calls=0)


def clone(function, namespace):
    copied = types.FunctionType(function.__code__, namespace, function.__name__,
                                function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    copied.__annotations__ = function.__annotations__
    return copied


def audit_clean(registration):
    check_clean_registration(registration)
    namespace = dict(vars(original))
    namespace['clean'] = revised_clean
    # Same function code, same deep read-only ledger/pixel/feature checks.
    return clone(original.audit_clean, namespace)(registration)


def save_with_adapter_evidence(path, value):
    """Append provenance only, before the original once-only closure write."""
    if Path(path).name == 'closure_audit.json':
        require(value.get('schema') == SCHEMA
                and value.get('status') == 'R0_COMPLETE_AUTHENTICATED_R1_NOT_REGISTERED'
                and value.get('no_new_science') is True and 'resource_adapter' not in value,
                'Unexpected closure metadata')
        value = dict(value, resource_adapter=adapter_evidence())
    return original.bler.save(path, value)


def validate_adapter_evidence(audit):
    """For the later R1 builder: verify only the closure adapter provenance."""
    require(audit.get('schema') == SCHEMA
            and audit.get('status') == 'R0_COMPLETE_AUTHENTICATED_R1_NOT_REGISTERED',
            'Authenticated original R0 closure schema required')
    require(original.same_pin(audit['finalizer'], pin(__file__)),
            'Closure was not produced by this pinned adapter')
    checkpin(audit['finalizer'])
    require(audit.get('resource_adapter') == adapter_evidence(),
            'Resource adapter provenance changed')
    require(original.same_pin(audit['clean']['registration'],
                              audit['resource_adapter']['actual_clean_registration']),
            'Clean audit and resource adapter registration differ')
    return audit['resource_adapter']


def finalize(bler_registration, clean_registration, output):
    check_clean_registration(clean_registration)
    require(Path(output).absolute() == NOMINATION_OUTPUT.absolute(),
            'The single authoritative R0 nomination output is required')
    namespace = dict(vars(original))
    namespace.update(clean=revised_clean, audit_clean=audit_clean, __file__=__file__)
    # No mutation of the original module or original bler module globals.
    proxy = types.SimpleNamespace(**vars(original.bler))
    proxy.save = save_with_adapter_evidence
    namespace['bler'] = proxy
    return clone(original.finalize, namespace)(bler_registration, clean_registration, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bler', 'clean'):
        parser.add_argument('--' + name + '-registration', required=True)
        parser.add_argument('--' + name + '-registration-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(original.bler.canonical(finalize(
        dict(path=args.bler_registration, sha256=args.bler_registration_sha256),
        dict(path=args.clean_registration, sha256=args.clean_registration_sha256), args.output)))
