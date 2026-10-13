"""One explicit, pinned adapter runner for every R1 process boundary.

This loads code only before the original entry validates its own registration.
The RX model process receives exactly its original source-free arguments.
"""
from __future__ import annotations
import argparse
import ast
import importlib.util
from pathlib import Path
import sys
import types

import pin_adapter as a

HERE = Path(__file__).resolve().parent
RESOURCE = HERE.parent / 'r1_execution_resource_v2'
ORIGINAL = {
    str(a.PHY): a.PHY_SHA,
    str(RESOURCE/'owner.py'): '1cc1ba2a64ad73303deb992438da6fae669836228894cec0613b0e9570608ab6',
    str(RESOURCE/'receiver_visual.py'): '55cd58863018e30ce92e501148c225fd4b412ac1c99c274d3ea0b63f964e2a9f',
    str(RESOURCE/'score_selection.py'): '1bd5d9b016ed9d4bd8044acf67942a2fc06bd0e23450fda5d04039543ff48947',
    str(RESOURCE/'dispatch_once.py'): 'f4b635edc808f4337133b09eaace2e72d34df2698f8490db8e95c3d8214edd18',
}


def route(argv, reg):
    """Preserve original script/action adjacency and every original argument."""
    argv = list(argv)
    a.require(len(argv) >= 4 and argv[1:3] == ['-B', '-u'] and argv[3] in ORIGINAL,
              'Unexpected R1 child command')
    own = str(Path(__file__).resolve()); adapter = str(Path(a.__file__).resolve())
    a.require(reg['code_bindings'][own] == a.sha(own)
              and reg['code_bindings'][adapter] == a.sha(adapter), 'Unbound metadata runner')
    return argv[:3] + [own, '--runner-sha256', a.sha(own), '--adapter-sha256', a.sha(adapter),
                      '--original'] + argv[3:]


def _dispatch_namespace(module):
    text = Path(module.__file__).read_text(encoding='utf-8-sig')
    old = "argv = [s.PY_CPU, '-B', '-u', str(ENTRY), '--registration', str(REG),\n                '--registration-sha256', expected]"
    new = old + '\n        argv = _metadata_route(argv, reg)'
    a.require(text.count(old) == 1, 'Original dispatch argv shape differs')
    changed = text.replace(old, new)
    a.require(ast.dump(ast.parse(changed.replace(new, old))) == ast.dump(ast.parse(text)),
              'Dispatcher inverse AST proof failed')
    module.__dict__['_metadata_route'] = route
    exec(compile(changed, module.__file__, 'exec'), module.__dict__)


def load_entry(path):
    path = str(Path(path).resolve())
    a.require(path in ORIGINAL and a.sha(path) == ORIGINAL[path], 'Original R1 entry binding differs')
    phy = a.install_phy()
    if path == str(a.PHY):
        module = phy
    else:
        spec = importlib.util.spec_from_file_location('_explicit_R1_entry', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if path == str(RESOURCE/'owner.py'):
        original_command = module.stage_command
        def stage_command(*args, **kwargs):
            # Original signature starts (reg, registration, stage, parent).
            reg = args[0] if args else kwargs['reg']
            return route(original_command(*args, **kwargs), reg)
        module.stage_command = stage_command
    elif path == str(RESOURCE/'dispatch_once.py'):
        _dispatch_namespace(module)
    return module


def parse_envelope(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    a.require(raw.count('--original') == 1, 'Exactly one explicit original entry required')
    boundary = raw.index('--original')
    a.require(boundary+1 < len(raw), 'Original entry path missing')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner-sha256', required=True)
    parser.add_argument('--adapter-sha256', required=True)
    parser.add_argument('--original', required=True)
    args = parser.parse_args(raw[:boundary+2])
    args.original_arguments = raw[boundary+2:]
    return args


def main():
    args = parse_envelope()
    a.require(a.sha(__file__) == args.runner_sha256 and a.sha(a.__file__) == args.adapter_sha256,
              'Explicit metadata runner SHA mismatch')
    module = load_entry(args.original)
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig'))
    blocks = [node for node in tree.body if isinstance(node, ast.If)
              and ast.unparse(node.test) == "__name__ == '__main__'"]
    a.require(len(blocks) == 1, 'Original CLI main block shape differs')
    sys.argv = [module.__file__, *args.original_arguments]
    module.__dict__['__name__'] = '__main__'
    # Only the original CLI block executes; loaded private functions keep their
    # private namespace, including the explicit validator/stage-command adapter.
    exec(compile(ast.Module(body=blocks[0].body, type_ignores=[]), module.__file__, 'exec'), module.__dict__)


if __name__ == '__main__':
    main()
