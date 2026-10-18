import ast,pathlib,unittest
import numpy as np
import link
class Wire(unittest.TestCase):
    def test_public96(self):
        p=link.templates();self.assertEqual(len(p),96);self.assertEqual(len({r['profile_id'] for r in p}),96)
        self.assertEqual(sum(r['mode']=='null' for r in p),12)
        self.assertTrue(all(r['n']==956*r['q'] and r['source_capacity_bits']==r['k']-29 for r in p))
    def test_all_nonnull_payload_roundtrips(self):
        for p in link.templates():
            if p['mode']=='null':continue
            payload=bytes((i*13)%256 for i in range(p['source_capacity_bits']//8))
            a=link.parse(link.pack(payload,p),p);self.assertEqual(a['payload'],payload);self.assertEqual(a['status'],'PAYLOAD_PARSED')
    def test_all_null_paid(self):
        for p in link.templates()[:12]:
            bits=link.pack(b'',p);self.assertEqual(len(bits),p['k']);self.assertEqual(link.parse(bits,p)['status'],'NULL_PROFILE')
    def test_wrong_crc_gray(self):
        p=link.templates()[12];b=link.pack(b'ab',p);b[-1]^=1
        self.assertEqual(link.parse(b,p)['status'],'CRC_REJECT');self.assertIsNone(link.parse(b,p)['payload'])
    def test_wrong_but_accepted_payload_not_corrected(self):
        p=link.templates()[12];a=link.parse(link.pack(b'changed',p),p);self.assertEqual(a['payload'],b'changed')
    def test_nonzero_padding_rejected(self):
        p=link.templates()[12];b=link.pack(b'a',p);b[21]=1;b=link.ep.append_crc(b[:-16]);self.assertEqual(link.parse(b,p)['status'],'BODY_INVALID')
    def test_empty_nonnull_rejected(self):
        with self.assertRaises(AssertionError):link.pack(b'',link.templates()[12])
    def test_nonempty_null_rejected(self):
        with self.assertRaises(AssertionError):link.pack(b'a',link.templates()[0])
    def test_capacity_includes29(self):
        p=link.templates()[12]
        with self.assertRaises(AssertionError):link.pack(bytes(p['k']//8),p)
    def test_no_fit_and_fallback(self):
        a={e:100*e for e in range(1,8)};self.assertEqual(link.fallback(a,7,99),0);self.assertEqual(link.fallback(a,7,350),3)
    def test_actual_prefix_not_target(self):
        a={e:100*e for e in range(1,8)};self.assertEqual(link.fallback(a,2,700),2)
    def test_receiver_signature_no_tx(self):
        tree=ast.parse(pathlib.Path(link.__file__).read_text());r=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='receive')
        self.assertEqual([a.arg for a in r.args.args],['self','observed','snr','counter','ledger','key'])
    def test_no_old_runtime_factory(self):
        t=ast.parse(pathlib.Path(link.__file__).read_text());calls={ast.unparse(n.func) for n in ast.walk(t) if isinstance(n,ast.Call)}
        self.assertNotIn('ep.Runtime',calls);self.assertNotIn('ep.create_runtime',calls)
if __name__=='__main__':unittest.main(verbosity=2)
