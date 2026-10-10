"""Tiny local file/path regressions for ldd aliases; no scientific imports."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import leo_whole_gate_v1 as g


class Tests(unittest.TestCase):
    def fixture(self,root):
        base=root/'personal';base.mkdir()
        packages=base/'packages';(packages/'torch/lib').mkdir(parents=True)
        target=packages/'nvidia/cublas/lib/libcublas.so.12'
        target.parent.mkdir(parents=True);target.write_bytes(b'sealed native library')
        lexical=packages/'torch/lib/../../nvidia/cublas/lib/libcublas.so.12'
        fact=dict(path=str(lexical),sha256=g.sha(target),bytes=target.stat().st_size)
        mapping={str(target):dict(original_path='/old/libcublas.so.12',path=str(target),
                                 sha256=fact['sha256'],bytes=fact['bytes'])}
        return base,target,lexical,fact,mapping

    def test_actual_ldd_parent_segments_preserve_observation_and_reverify(self):
        with tempfile.TemporaryDirectory() as td:
            base,target,lexical,fact,mapping=self.fixture(Path(td));before=copy.deepcopy(fact)
            result=g.normalize_native_facts([fact],mapping,base)
            self.assertEqual(fact,before)
            self.assertEqual(result[0]['observed_lexical_path'],str(lexical))
            self.assertEqual(result[0]['path'],str(target.resolve()))
            self.assertNotIn('..',Path(result[0]['path']).parts)
            runtime=object.__new__(g.RuntimeFiles)
            runtime.base=base;runtime.used={}
            witness=base/'runtime_witness.bin';witness.write_bytes(b'frozen fixture')
            runtime.rows={str(witness):dict(original_path='/old/runtime_witness.bin',
                path=str(witness),sha256=g.sha(witness),bytes=witness.stat().st_size)}
            runtime.generated=[];runtime.host_native=result
            runtime.reverify()
            target.write_bytes(b'x'*fact['bytes'])
            with self.assertRaisesRegex(RuntimeError,'native dependency changed'):runtime.reverify()

    def test_personal_escape_and_leave_then_reenter_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base,target,_,fact,mapping=self.fixture(root)
            outside=root/'external.so';outside.write_bytes(target.read_bytes())
            for path in [base/'../external.so',base/'../personal/packages/nvidia/cublas/lib/libcublas.so.12']:
                with self.subTest(path=path),self.assertRaisesRegex(RuntimeError,'outside personal base'):
                    g.normalize_native_facts([fact|{'path':str(path)}],mapping,base)

    def test_personal_map_sha_size_and_missing_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base,target,_,fact,mapping=self.fixture(Path(td))
            for bad in [fact|{'sha256':'0'*64},fact|{'bytes':fact['bytes']+1}]:
                with self.subTest(bad=bad),self.assertRaisesRegex(RuntimeError,'pin differs'):
                    g.normalize_native_facts([bad],mapping,base)
            with self.assertRaisesRegex(RuntimeError,'absent from active'):
                g.normalize_native_facts([fact],{},base)
            target.write_bytes(b'short')
            with self.assertRaisesRegex(RuntimeError,'size changed'):
                g.normalize_native_facts([fact],mapping,base)

    def test_cancelled_component_is_checked_for_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            base,_,_,fact,mapping=self.fixture(Path(td));link=base/'packages/torch/lib'
            original=Path.is_symlink
            # Portable proof that a link in a cancelled component is checked,
            # even on Windows where creating a symlink may require a privilege.
            with mock.patch.object(Path,'is_symlink',lambda p:p==link or original(p)):
                with self.assertRaisesRegex(RuntimeError,'Symlink'):
                    g.normalize_native_facts([fact],mapping,base)

    def test_real_personal_symlink_and_symlink_parent_semantics_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base,target,_,fact,mapping=self.fixture(root)
            outside=root/'external';outside.mkdir();link=base/'link'
            try:link.symlink_to(outside,target_is_directory=True)
            except OSError as exc:self.skipTest('Platform symlink privilege unavailable: '+str(exc))
            for path in [link/'library.so',link/'../personal/packages/nvidia/cublas/lib/libcublas.so.12']:
                with self.subTest(path=path),self.assertRaises(RuntimeError):
                    g.normalize_native_facts([fact|{'path':str(path)}],mapping,base)

    def test_existing_host_path_normalizes_and_conflicting_alias_pin_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base,_,_,_,mapping=self.fixture(root)
            host=root/'host';(host/'lib').mkdir(parents=True)
            target=host/'host.so';target.write_bytes(b'host bytes')
            fact=dict(path=str(host/'lib/../host.so'),sha256=g.sha(target),bytes=target.stat().st_size)
            result=g.normalize_native_facts([fact],mapping,base)
            self.assertEqual(result[0]['path'],str(target.resolve()))
            self.assertEqual(result[0]['observed_lexical_path'],fact['path'])
            with self.assertRaisesRegex(RuntimeError,'Conflicting native aliases'):
                g.normalize_native_facts([fact,fact|{'path':str(target),'sha256':'0'*64}],mapping,base)

    def test_host_spelling_cannot_reenter_personal_namespace(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base,target,_,fact,mapping=self.fixture(root);(root/'outside').mkdir()
            path=root/'outside/../personal/packages/nvidia/cublas/lib/libcublas.so.12'
            with self.assertRaisesRegex(RuntimeError,'Host native alias enters'):
                g.normalize_native_facts([fact|{'path':str(path)}],mapping,base)


if __name__=='__main__':unittest.main(verbosity=2)
