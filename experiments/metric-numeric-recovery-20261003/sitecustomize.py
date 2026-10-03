"""Inert except at the two explicitly bound frozen final-cache entries."""
import importlib.util
import os
from pathlib import Path
import sys

FOLDER=Path('/home/liulu/projects/VAR_COMM/outputs/METRIC-FINAL-CACHE-20261002/runtime').resolve()
if sys.argv and Path(sys.argv[0]).resolve() in {FOLDER/'controller.py',FOLDER/'final_runner.py'}:
    try:
        here=Path(__file__).resolve().parent
        spec=importlib.util.spec_from_file_location('_numeric_recovery_startup',here/'launch_recovery.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        path=os.environ.get(module.ENV)
        if not path:raise RuntimeError('Metric numeric recovery manifest missing')
        module.install(path,sys.argv,here)
    except BaseException as error:
        sys.stderr.write('Metric numeric startup refused: '+str(error)+'\n');sys.stderr.flush();os._exit(78)
