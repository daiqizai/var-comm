"""Tiny real-file native observation checks; no model, tensor, CUDA or PHY calls."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import leo_whole_gate_v1 as g
import leo_whole_input_map_v1 as m


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.base=self.root/'personal';self.base.mkdir()
        self.host=self.root/'host';self.host.mkdir();self.lib=self.host/'libfixture.so.1.2'
        self.lib.write_bytes(b'actual tiny native fixture')
        self.row=dict(path=str(self.lib),sha256=g.sha(self.lib),bytes=self.lib.stat().st_size)
        self.projection_sha='a'*64
        self.doc=dict(schema='LEO_NEW_HOST_NATIVE_ADDENDUM_V1',status='CPU_IMPORT_OBSERVED_NOT_NUMERICAL',
            model_calls=0,tensor_calls=0,original_runtime_projection_sha256=self.projection_sha,files=[self.row])
        self.runtime=object.__new__(g.RuntimeFiles);self.runtime.base=self.base;self.runtime.used={}
        witness=self.base/'runtime_witness';witness.write_bytes(b'original runtime witness')
        self.runtime.rows={str(witness):dict(original_path='/old/runtime_witness',path=str(witness),
            sha256=g.sha(witness),bytes=witness.stat().st_size)}
        self.runtime.generated=[];self.original_host=[];self.runtime.host_native=self.original_host
        self.number=0

    def pin(self,doc=None):
        self.number+=1;p=self.base/f'addendum{self.number}.json';g.write(p,self.doc if doc is None else doc)
        return dict(path=str(p),sha256=g.sha(p))

    def attach(self,doc=None):
        g.attach_host_native_addendum(self.runtime,self.pin(doc),self.projection_sha,self.base)

    def test_real_files_accept_then_reverify_without_mutating_original_projection(self):
        self.attach();self.assertEqual(self.runtime.host_native,[self.row]);self.assertEqual(self.original_host,[])
        self.assertIsNot(self.runtime.host_native,self.original_host);self.assertIsNot(self.runtime.host_native[0],self.row)
        self.runtime.reverify()
        self.lib.write_bytes(b'x'*self.row['bytes'])
        with self.assertRaisesRegex(RuntimeError,'native dependency changed'):self.runtime.reverify()

    def test_stale_projection_wrong_schema_status_or_nonzero_calls_stop(self):
        changes=[dict(original_runtime_projection_sha256='b'*64),dict(schema='wrong'),dict(status='PASS'),
                 dict(model_calls=1),dict(tensor_calls=1),dict(model_calls=False)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(RuntimeError):self.attach(self.doc|change)
        self.assertIs(self.runtime.host_native,self.original_host)

    def test_tampered_receipt_and_receipt_outside_personal_base_stop(self):
        pin=self.pin();Path(pin['path']).write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'JSON pin changed'):
            g.attach_host_native_addendum(self.runtime,pin,self.projection_sha,self.base)
        p=self.host/'addendum.json';g.write(p,self.doc)
        with self.assertRaisesRegex(RuntimeError,'Personal absolute path required'):
            g.attach_host_native_addendum(self.runtime,dict(path=str(p),sha256=g.sha(p)),self.projection_sha,self.base)

    def test_wrong_file_sha_size_and_existing_host_conflict_stop(self):
        for change in [dict(sha256='0'*64),dict(bytes=self.row['bytes']+1)]:
            with self.subTest(change=change),self.assertRaisesRegex(RuntimeError,'file bytes changed'):
                self.attach(self.doc|dict(files=[self.row|change]))
        self.runtime.host_native=[self.row|dict(sha256='0'*64)]
        with self.assertRaisesRegex(RuntimeError,'conflicts with original facts'):self.attach()
        self.runtime.host_native=[dict(self.row)];before=copy.deepcopy(self.runtime.host_native)
        self.attach();self.assertEqual(self.runtime.host_native,before)

    def test_personal_native_and_noncanonical_parent_segments_stop(self):
        p=self.base/'personal.so';p.write_bytes(self.lib.read_bytes())
        with self.assertRaisesRegex(RuntimeError,'outside the personal base'):
            self.attach(self.doc|dict(files=[self.row|dict(path=str(p))]))
        (self.host/'nested').mkdir()
        with self.assertRaisesRegex(RuntimeError,'Canonical host native file'):
            self.attach(self.doc|dict(files=[self.row|dict(path=str(self.host/'nested/../libfixture.so.1.2'))]))
        # Portable link refusal, including Windows without symlink privileges.
        original=Path.is_symlink
        with mock.patch.object(Path,'is_symlink',lambda p:p==self.lib or original(p)):
            with self.assertRaisesRegex(RuntimeError,'Canonical host native file'):self.attach()

    def test_duplicate_empty_oversize_and_nonlibrary_files_stop(self):
        for rows in [[],[self.row]*2,[self.row]*33]:
            with self.subTest(count=len(rows)),self.assertRaises(RuntimeError):self.attach(self.doc|dict(files=rows))
        p=self.host/'script.py';p.write_bytes(self.lib.read_bytes())
        with self.assertRaisesRegex(RuntimeError,'shared libraries'):
            self.attach(self.doc|dict(files=[self.row|dict(path=str(p))]))
        p=self.host/'directory.so';p.mkdir()
        with self.assertRaisesRegex(RuntimeError,'file bytes changed'):
            self.attach(self.doc|dict(files=[self.row|dict(path=str(p))]))

    def test_later_invalid_row_cannot_partially_attach(self):
        p=self.host/'second.so';p.write_bytes(b'second')
        bad=dict(path=str(p),sha256='0'*64,bytes=p.stat().st_size)
        with self.assertRaises(RuntimeError):self.attach(self.doc|dict(files=[self.row,bad]))
        self.assertIs(self.runtime.host_native,self.original_host);self.assertEqual(self.original_host,[])

    def test_builder_optional_arguments_are_paired_and_pin_is_preserved(self):
        self.assertIsNone(m.optional_host_native_pin(SimpleNamespace()))
        pin=self.pin()
        for args in [dict(host_native_addendum=pin['path']),dict(host_native_addendum_sha256=pin['sha256']),
                     dict(host_native_addendum=pin['path'],host_native_addendum_sha256='invalid')]:
            with self.subTest(args=args),self.assertRaises(RuntimeError):m.optional_host_native_pin(SimpleNamespace(**args))
        original_inside=g.inside
        with mock.patch.object(g,'inside',side_effect=lambda p:original_inside(p,self.base)):
            self.assertEqual(m.optional_host_native_pin(SimpleNamespace(host_native_addendum=pin['path'],
                             host_native_addendum_sha256=pin['sha256'])),pin)

    def test_published_base_or_verified_descendant_only(self):
        with mock.patch.object(m.subprocess,'run') as run:
            m.verify_published_head(self.base,m.HEAD);run.assert_not_called()
            descendant='b'*40;run.return_value=SimpleNamespace(returncode=0)
            m.verify_published_head(self.base,descendant)
            self.assertEqual(run.call_args.args[0][-4:],['merge-base','--is-ancestor',m.HEAD,descendant])
            for code in [1,128]:
                run.return_value=SimpleNamespace(returncode=code)
                with self.subTest(code=code),self.assertRaisesRegex(RuntimeError,'not a verified descendant'):
                    m.verify_published_head(self.base,descendant)


if __name__=='__main__':unittest.main(verbosity=2)
