"""Explicit metadata-only compatibility for optional checkpoint byte counts.

No scientific function or received information changes. Original PHY source and
its registration pin remain intact; the master separately binds this adapter.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PHY = EXPERIMENT / 'r1_phy.py'
PHY_SHA = 'cd4e25660d637d82632622d765fcb0234a642c369a9f6801755b589ad551856b'
OLD = 'all(row[key] == selected[key] for key in selected)'
NEW = '_strict_selected_identity(row, selected)'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_checkpoint(left, right):
    for value in (left, right):
        require(type(value) is dict and set(value) in
                ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}), 'Unexpected checkpoint pin keys')
        require(type(value['path']) is str and type(value['sha256']) is str,
                'Checkpoint pin identity must be text')
        path = Path(value['path'])
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Invalid checkpoint path')
        require(sha(path) == value['sha256'], 'Checkpoint content SHA differs')
        if 'bytes' in value:
            require(type(value['bytes']) is int and value['bytes'] >= 0
                    and path.stat().st_size == value['bytes'], 'Checkpoint byte count differs')
    require(left['path'] == right['path'] and left['sha256'] == right['sha256'],
            'Checkpoint path or SHA identity differs')
    return True


def strict_selected_identity(row, selected):
    require(type(row) is dict and type(selected) is dict and 'checkpoint' in selected,
            'Missing original selected source identity')
    require(set(selected) <= set(row), 'Selected source keys missing')
    for key in selected:
        if key == 'checkpoint':
            strict_checkpoint(row[key], selected[key])
        else:
            require(row[key] == selected[key], 'Selected source identity differs: ' + key)
    return True


def adapted_source(text):
    require(text.count(OLD) == 1 and NEW not in text, 'Original selected comparison shape differs')
    result = text.replace(OLD, NEW)
    require(ast.dump(ast.parse(result.replace(NEW, OLD))) == ast.dump(ast.parse(text)),
            'Inverse AST source identity failed')
    return result


def install_phy():
    """Private explicit module namespace; no interpreter-wide startup hook."""
    require(sha(PHY) == PHY_SHA, 'Frozen PHY source changed')
    text = PHY.read_text(encoding='utf-8-sig')
    changed = adapted_source(text)
    for path in (EXPERIMENT, EXPERIMENT / 'r1_execution_resource_v2'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = types.ModuleType('r1_phy')
    module.__file__ = str(PHY)
    module.__package__ = ''
    module.__dict__['_strict_selected_identity'] = strict_selected_identity
    # __name__ is r1_phy, so the original CLI block does not execute here.
    exec(compile(changed, str(PHY), 'exec'), module.__dict__)
    sys.modules['r1_phy'] = module
    return module


def proof():
    require(sha(PHY) == PHY_SHA, 'Frozen PHY source changed')
    text = PHY.read_text(encoding='utf-8-sig'); adapted_source(text)
    return dict(original_phy=dict(path=str(PHY), sha256=PHY_SHA),
                replacement_old=OLD, replacement_new=NEW, inverse_AST_equal=True,
                optional_checkpoint_keys=['bytes'], path_sha_exact=True,
                declared_bytes_type='int_not_bool', real_stat_and_SHA_required=True,
                other_selected_fields_exact=True, science_changes=0)
