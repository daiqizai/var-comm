"""Independent bounded synthetic review; no deployment, registration or timing."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

p = argparse.ArgumentParser()
p.add_argument('--entry', required=True)
p.add_argument('--v2', required=True)
p.add_argument('--post', required=True)
p.add_argument('--original-owner', required=True)
a = p.parse_args()
entry = Path(a.entry).resolve()
expected = {
    'resource_adapter.py': 'e3bfcbba9dfb31ed90909c59c54855de177a85896505a1bea23411f1e62a46bc',
    'timing_resource_owner.py': '613a0c8fb65c3e2b20f82af79f8821bae845a23427d9c723fa79808ef3bf4f3f',
    'test_resource_adapter.py': '17c14b4e4f1fd1939b8b55b74288dadb4eb3ae45850b28990eac9a5b2c162db5',
    'README.md': 'b5f44b73918f9daf2ae06ef6f3333011e9bdb6f6f8a91688afb6eade79fc338e',
}
originals = [
    (Path(a.v2).resolve(), 'ef35e3c0b1f1ed50840ca6473b596bad865e791ce28743e197b49fae88f0dc50'),
    (Path(a.post).resolve(), '7a3414ff2088c00d8a7d4fff61d6baf93c67640f4ccc0861ecbb8c3449dbb8f2'),
    (Path(a.original_owner).resolve(), 'c59d4cf22c1c8da36861d9af8e28ade8d2c08d6219013ab4f03d6d88e26997d8'),
]
source_pins = []
for path, digest in [(entry / n, h) for n, h in expected.items()] + originals:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, str(path)
    source_pins.append(dict(local_path=str(path), sha256=digest, bytes=len(data)))
original_text = [path.read_text(encoding='utf8') for path, _ in originals]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(entry))
blocked = []
exact_reads = {str(path).casefold() for path, _ in originals}
runtime_root = Path(sys.base_prefix).resolve()
model_names = {'torch', 'numpy', 'tensorflow', 'sionna', 'lpips', 'transformers', 'timm', 'jax'}

def stop(kind, value):
    blocked.append(dict(kind=kind, value=str(value)))
    raise RuntimeError('Independent synthetic boundary: ' + kind)

def audit(event, args):
    if event.startswith(('subprocess.', 'socket.')) or event in {'os.system', 'os.fork', 'os.posix_spawn', 'os.spawn'}:
        stop(event, args)
    if event == 'import' and str(args[0]).split('.')[0] in model_names:
        stop(event, args[0])
    if event == 'open' and not isinstance(args[0], int):
        target = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(v in mode for v in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            stop('filesystem_write', target)
        allowed = target.is_relative_to(entry) or target.is_relative_to(runtime_root) or str(target).casefold() in exact_reads
        if not allowed:
            stop('out_of_scope_read', target)

sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location('new_timing_resource_tests', entry / 'test_resource_adapter.py')
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)

def function_source(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node) + '\n'

class IndependentFixtureTests(unittest.TestCase):
    def test_both_embedded_functions_are_exact_original_source(self):
        self.assertEqual(test_module.RESOURCE_SOURCE, function_source(original_text[0], 'resource_snapshot'))
        self.assertEqual(test_module.POST_SOURCE, function_source(original_text[1], 'validate_resources'))

    def test_file_override_preserves_original_getsource_and_code(self):
        tree = ast.parse(original_text[2])
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'run', 'worker'}]
        self.assertEqual(len(selected), 2)
        namespace = {'__file__': str(originals[2][0])}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(originals[2][0]), 'exec'), namespace)
        before = {name: (namespace[name].__code__, inspect.getsource(namespace[name])) for name in ('run', 'worker')}
        namespace['__file__'] = str(entry / 'timing_resource_owner.py')
        for name, (code, source) in before.items():
            self.assertIs(namespace[name].__code__, code)
            self.assertEqual(inspect.getsource(namespace[name]), source)
            self.assertEqual(source, function_source(original_text[2], name))
        self.assertEqual(namespace['run'].__globals__['__file__'], str(entry / 'timing_resource_owner.py'))

suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromModule(test_module),
    unittest.defaultTestLoader.loadTestsFromTestCase(IndependentFixtureTests),
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
models = sorted(name for name in sys.modules if name.split('.')[0] in model_names)
record = dict(schema='TIMING_RESOURCE_INDEPENDENT_SYNTHETIC_RESULT_V1', testsRun=result.testsRun,
    failures=len(result.failures), errors=len(result.errors), blocked=blocked, models_imported=models,
    CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), source_pins=source_pins,
    real_registration_or_readiness_called=False, real_ledger_or_GPU_read=False,
    scientific_or_timing_calls=0, remote_writes=0)
print(json.dumps(record, sort_keys=True))
assert result.wasSuccessful() and result.testsRun == 34 and not blocked and not models
