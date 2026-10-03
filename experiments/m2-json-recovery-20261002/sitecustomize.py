"""Inert outside the exact R4 M2 entry; eligible startup failures exit 78."""
from pathlib import Path
import importlib.util
import os
import sys

ENTRY = Path('/home/liulu/projects/VAR_COMM/outputs/METRIC-CONCURRENT-R4-20261002/runtime/scheduled_process.py')

if sys.argv and Path(sys.argv[0]).resolve() == ENTRY.resolve():
    try:
        here = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location('_m2_boolean_json_recovery', here / 'bool_json.py')
        adapter = importlib.util.module_from_spec(spec); sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        manifest = os.environ.get(adapter.ENV)
        if not manifest: raise RuntimeError('M2 JSON recovery manifest was not supplied')
        adapter.install(manifest, sys.argv, here)
    except BaseException as error:
        sys.stderr.write('M2 boolean JSON recovery startup refused: ' + str(error) + '\n')
        sys.stderr.flush()
        # Python normally ignores errors in sitecustomize; refuse eligible
        # execution explicitly rather than silently running without the fix.
        os._exit(78)
