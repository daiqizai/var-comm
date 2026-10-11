"""No-model tests of the exact native-library addendum boundary."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from timing_runtime import register_native_addendum, sha

class NativeAddendumTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'token_trellis.cpp';source.write_bytes(b'fake-source-for-test')
        library=root/'token_trellis.so';library.write_bytes(b'not-a-real-library')
        descriptor=root/'build.json'
        descriptor.write_text(json.dumps(dict(source_sha256=sha(source),library_sha256=sha(library))))
        return dict(native_header={k:dict(path=str(p),sha256=sha(p)) for k,p in
            [('source',source),('library',library),('descriptor',descriptor)]})

    def test_adds_only_prebound_library_without_mutating_original_inventory(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);reg=self.fixture(root);original=[]
            env=SimpleNamespace(host_native=original)
            register_native_addendum(reg,env,root)
            self.assertEqual(original,[]);self.assertEqual(len(env.host_native),1)
            self.assertEqual(env.host_native[0]['sha256'],reg['native_header']['library']['sha256'])

    def test_changed_library_rejected_before_inventory_extension(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);reg=self.fixture(root);env=SimpleNamespace(host_native=[])
            (root/'token_trellis.so').write_bytes(b'changed')
            with self.assertRaises(AssertionError):register_native_addendum(reg,env,root)
            self.assertEqual(env.host_native,[])

if __name__=='__main__':unittest.main()
