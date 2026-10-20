"""Synthetic qualification only; no production model/codec/PHY invocation."""
import ast
import tempfile
import types
import unittest
from pathlib import Path
import contract
import native
from unittest.mock import patch


class ContractTests(unittest.TestCase):
    def test_grid(self):
        rows=contract.grid()
        self.assertEqual(len(rows),288)
        self.assertEqual(len({r['key'] for r in rows}),288)
        self.assertEqual({r['source_index'] for r in rows},set(contract.INDICES))
        for snr in contract.SNRS:
            for idx in contract.INDICES:
                a=[r for r in rows if r['snr_db']==snr and r['source_index']==idx]
                self.assertEqual([r['phase'] for r in a],['warmup']*3+['measure']*3)
                self.assertEqual(len({r['counter'] for r in a}),1)
        self.assertEqual(sum(r['phase']=='measure' for r in rows)*5,720)

    def test_resource_adapter_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fake.py';p.write_text('def f(x):\n    assert x == 2\n    return x*x\n')
            m=types.SimpleNamespace(__file__=str(p))
            f=native.replace_function(m,'f',[('x == 2','x == 6')])
            self.assertEqual(f(6),36)
            with self.assertRaises(AssertionError):f(2)
            with self.assertRaises(AssertionError):native.replace_function(m,'f',[('x == 9','x == 6')])

    def test_packet_accounting(self):
        c=native.Calls()
        self.assertEqual(c.call({'kind':'header'}, {}, lambda:4),4)
        self.assertEqual(c.decode('test','key','body',{},lambda:7),7)
        self.assertEqual(c.counts,dict(header=1,body=1))

    def test_new_owner_no_old_launchers(self):
        s=(Path(__file__).parent/'owner.py').read_text()
        self.assertNotIn('timing_owner_v6',s)
        self.assertNotIn('training',s.replace('training=0',''))
        self.assertIn('p.wait(',s)
        self.assertNotIn('.poll(',s)

    def test_no_inference_in_import(self):
        for p in Path(__file__).parent.glob('*.py'):
            tree=ast.parse(p.read_text())
            for n in tree.body:
                if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
                    self.fail(f'Top-level call in {p.name}')

    def test_rx_entry_has_no_source_arg(self):
        tree=ast.parse((Path(__file__).parent/'worker.py').read_text())
        n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_rx')
        self.assertEqual([a.arg for a in n.args.args],['actual','r','method','ctx','meter'])
        text=ast.unparse(n)
        self.assertNotIn('pixels',text)
        self.assertNotIn('archive',text)
        self.assertIn('None, trans, modes',text)

    def test_bpg_failure_preserved_success_cleaned(self):
        import bpg
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);tmp=root/'tmp';tmp.mkdir()
            with patch.object(contract,'T',root/'evidence'):
                with self.assertRaisesRegex(RuntimeError,'synthetic'):
                    with bpg.workspace(tmp) as name:
                        (Path(name)/'input.bin').write_bytes(b'synthetic')
                        raise RuntimeError('synthetic')
                evidence=list((root/'evidence/BPG/failed_codec_evidence').glob('*/input.bin'))
                self.assertEqual(len(evidence),1)
                self.assertEqual(evidence[0].read_bytes(),b'synthetic')
                with bpg.workspace(tmp) as name:
                    (Path(name)/'success.bin').write_bytes(b'ok')
                self.assertFalse(Path(name).exists())


if __name__=='__main__':
    unittest.main()
