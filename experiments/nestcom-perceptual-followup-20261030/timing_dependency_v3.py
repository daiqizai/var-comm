"""Bind only the existing pure-Python Sionna package, not another Python environment."""
from followup_common import *
import importlib.util


def activate(manifest):
    spec_record = cr(manifest)
    root = Path(spec_record['package_root']).resolve()
    assert spec_record['version'] == '2.2.0' and spec_record['compiled_extensions'] == []
    for item in spec_record['files']:
        checked(item)
        assert Path(item['path']).resolve().is_relative_to(root)
    actual = {str(p.resolve()) for p in root.rglob('*') if p.is_file() and p.suffix != '.pyc'}
    assert actual == {v['path'] for v in spec_record['files']}
    before = list(sys.path)
    if 'sionna' not in sys.modules:
        spec = importlib.util.spec_from_file_location('sionna', root/'__init__.py',
                                                      submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['sionna'] = module
        spec.loader.exec_module(module)
    import sionna
    assert Path(sionna.__file__).resolve() == root/'__init__.py'
    assert sionna.__version__ == '2.2.0' and before == sys.path
    return spec_record
