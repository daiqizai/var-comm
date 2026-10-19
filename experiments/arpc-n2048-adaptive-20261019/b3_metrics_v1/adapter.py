"""Extract unchanged metric mathematics with two audited bookkeeping edits."""
import ast
import copy
import hashlib

ORIGINAL_SHA256 = 'dcb9c38d374238ba22125c7fb24a2f2268ddabde40827898777e217c302dde7b'
FUNCTIONS = ('csvwrite', 'load_image', 'ScoreLedger', 'score')


def selected_tree(source):
    assert hashlib.sha256(source).hexdigest() == ORIGINAL_SHA256
    tree = ast.parse(source.decode())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in FUNCTIONS]
    assert tuple(n.name for n in nodes) == FUNCTIONS
    return ast.Module(body=nodes, type_ignores=[])


def adapt(source, score_seconds):
    assert type(score_seconds) is int and 600 <= score_seconds <= 14400
    original = selected_tree(source)
    tree = copy.deepcopy(original)
    changed = []
    score = tree.body[-1]
    for n in ast.walk(score):
        if isinstance(n, ast.Constant) and type(n.value) is int and n.value in (1800, 3500):
            changed.append((n, n.value))
            n.value = 1200 if n.value == 1800 else score_seconds - 60
    assert sorted(v for _, v in changed) == [1800, 1800, 3500]
    inverse = copy.deepcopy(tree)
    inverse_nodes = [n for n in ast.walk(inverse.body[-1]) if isinstance(n, ast.Constant)
                     and type(n.value) is int and n.value in (1200, score_seconds - 60)]
    assert len(inverse_nodes) == 3
    for n in inverse_nodes:
        n.value = 1800 if n.value == 1200 else 3500
    assert ast.dump(inverse, include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(tree)


def install(source, namespace, seconds):
    exec(compile(adapt(source, seconds), '<frozen_N2048_metric_functions>', 'exec'), namespace)

