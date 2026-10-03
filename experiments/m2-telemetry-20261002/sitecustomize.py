"""Apply bound M2 telemetry and chain the original Boolean startup repair."""
import importlib.util
import os
from pathlib import Path
import sys

ENTRY=Path('/home/liulu/projects/VAR_COMM/outputs/METRIC-CONCURRENT-R4-20261002/runtime/scheduled_process.py')
if sys.argv and Path(sys.argv[0]).resolve()==ENTRY.resolve():
    try:
        here=Path(__file__).resolve().parent
        spec=importlib.util.spec_from_file_location('_m2_telemetry_startup',here/'telemetry.py')
        adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
        path=os.environ.get(adapter.ENV)
        if not path:raise RuntimeError('Telemetry manifest was not supplied')
        adapter.install(path,sys.argv,here)
    except BaseException as error:
        sys.stderr.write('M2 telemetry startup refused: '+str(error)+'\n');sys.stderr.flush();os._exit(78)
