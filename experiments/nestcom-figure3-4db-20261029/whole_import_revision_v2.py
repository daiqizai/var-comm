"""Resolve colliding legacy module names before the first model call.

The first WHOLE attempt stopped during import before loading a model or ledger.
Keep that failed source, registration and output directory unchanged.
"""
from pathlib import Path
import sys
import missing_worker as frozen
assert sys.modules['common'].__file__.endswith('adatoken-response-20261026/common.py')
sys.modules.pop('common')
frozen.native.paths()
import tb_native
assert 'pid_identity' in vars(sys.modules['common'])
frozen.L=frozen.L/'whole_import_revision_v2'
if __name__=='__main__':
    frozen.L.mkdir(exist_ok=True)
    frozen.main(*sys.argv[1:])
