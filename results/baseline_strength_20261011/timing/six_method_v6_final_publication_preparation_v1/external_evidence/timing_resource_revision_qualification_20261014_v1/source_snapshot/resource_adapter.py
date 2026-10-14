"""Pure, source-derived metadata adapters. No Torch imports or scientific calls."""
import ast
import hashlib

STARTUP_OLD = 'free >= 18432'
STARTUP_NEW = 'free >= 8192'
POST_OLD = "admission['selected_GPU']['memory_free_MiB'] >= 18432"
POST_NEW = "admission['selected_GPU']['memory_free_MiB'] >= 8192"


def changed_function(source, name, before, after):
    tree = ast.parse(source)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, 'Expected one original function'
    text = ast.get_source_segment(source, matches[0]) + '\n'
    assert text.count(before) == 1 and after not in text, 'Unexpected original comparator'
    changed = text.replace(before, after)
    restored = changed.replace(after, before)
    assert restored == text
    assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(text), include_attributes=False)
    proof = dict(function=name, before=before, after=after, inverse_AST_equal=True,
        original_function_sha256=hashlib.sha256(text.encode()).hexdigest(),
        adapted_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        scientific_function_body_changes=0, resource_comparator_changes=1)
    return changed, proof


def compile_function(source, namespace, filename, name, before, after):
    changed, proof = changed_function(source, name, before, after)
    ns = dict(namespace)
    exec(compile(changed, filename, 'exec'), ns)
    return ns[name], proof


def install_snapshot(v6, source, filename):
    replacement, proof = compile_function(source, vars(v6.v2), filename,
        'resource_snapshot', STARTUP_OLD, STARTUP_NEW)
    v6.v2.resource_snapshot = replacement
    v6.old.exclusive_snapshot = replacement
    return proof


def resource_validator(original, relaxed, bound_registration, bound_admission, bound_worker_pid):
    """Only the authenticated current segment gets 8 GiB. Prior/unknown stays original."""
    def validate(history, numerical, device, reg, admission, worker_pid, expected_boundaries=1730):
        if reg.get('schema') == 'FROZEN_SIX_METHOD_FIXED16_TIMING_V6_PARTIAL_RECOVERY':
            assert reg == bound_registration, 'Unbound v6 registration'
            assert admission == bound_admission, 'Unbound v6 admission'
            assert type(worker_pid) is int and worker_pid == bound_worker_pid, 'Unbound v6 worker'
            result = relaxed(history, numerical, device, reg, admission, worker_pid, expected_boundaries)
            result[0]['startup_free_threshold_MiB'] = 8192
            result[0]['historical_registration_startup_threshold_MiB'] = 18432
            result[0]['startup_resource_revision_applied'] = True
            return result
        return original(history, numerical, device, reg, admission, worker_pid, expected_boundaries)
    return validate
