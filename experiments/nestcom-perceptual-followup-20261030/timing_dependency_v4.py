"""Selective pure-Python dependency closure; preserve NumPy, SciPy and Torch routes."""
from followup_common import *
from timing_dependency_v3 import activate as activate_sionna
import importlib.util


def activate(manifest):
    record = cr(manifest)
    package = record['importlib_resources']
    root = Path(package['package_root']).resolve()
    for p in package['files']:
        checked(p)
        assert Path(p['path']).resolve().is_relative_to(root)
        assert Path(p['path']).suffix not in ['.so', '.dll', '.pyd']
    actual = {str(p.resolve()) for p in root.rglob('*') if p.is_file() and p.suffix != '.pyc'}
    assert actual == {v['path'] for v in package['files']}
    before = list(sys.path)
    if 'importlib_resources' not in sys.modules:
        spec = importlib.util.spec_from_file_location('importlib_resources', root/'__init__.py',
                                                      submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['importlib_resources'] = module
        spec.loader.exec_module(module)
    assert Path(sys.modules['importlib_resources'].__file__).resolve() == root/'__init__.py'
    assert before == sys.path
    activate_sionna(record['sionna_package'])
    return record
