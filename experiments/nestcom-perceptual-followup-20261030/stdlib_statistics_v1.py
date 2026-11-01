"""Resolve Python's standard statistics before historical paths are prepended."""
import hashlib
import importlib.util
import sys
import sysconfig
from pathlib import Path


def bind_stdlib_statistics():
    path=(Path(sysconfig.get_path('stdlib'))/'statistics.py').resolve(strict=True)
    current=sys.modules.get('statistics')
    if current is not None:
        if Path(current.__file__).resolve()!=path:
            raise RuntimeError('A nonstandard statistics module was already imported')
    else:
        spec=importlib.util.spec_from_file_location('statistics',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['statistics']=module
    module=sys.modules['statistics']
    assert Path(module.__file__).resolve()==path and module.median([1,3,2])==2
    return dict(path=str(path),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
