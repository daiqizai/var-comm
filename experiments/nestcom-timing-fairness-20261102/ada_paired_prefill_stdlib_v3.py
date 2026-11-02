"""Resolve stdlib statistics before historical modules extend sys.path.

The complete, already-qualified numerical worker is executed unchanged.
"""
from pathlib import Path
import sys,sysconfig,importlib.util,runpy

p=Path(sysconfig.get_path('stdlib'))/'statistics.py'
spec=importlib.util.spec_from_file_location('statistics',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert module.median([1,3,5])==3 and Path(module.__file__).resolve()==p.resolve()
sys.modules['statistics']=module
runpy.run_path(str(Path(__file__).with_name('ada_paired_prefill_headers_v2.py')),run_name='__main__')
