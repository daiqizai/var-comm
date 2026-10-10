"""Synthetic CPU qualification only; no model, channel or scientific data."""
import concurrent.futures
import tempfile
from pathlib import Path
import unittest
from t2_ledger import Ledger
from t2_pilot import state_key, validate_request

def add_one(args):
    path,i=args;l=Ledger(path,'a'*64,16)
    try:return l.call({'event_id':str(i),'kind':'body'},{'seed':i},lambda:{'decoded':i})
    finally:l.close()

class Qualification(unittest.TestCase):
    def test_completed_callback_never_repeated(self):
        with tempfile.TemporaryDirectory()as d:
            l=Ledger(Path(d)/'l.sqlite','a'*64,1);event={'event_id':'a'}
            self.assertEqual(l.call(event,{'x':1},lambda:{'v':2}),{'v':2})
            self.assertEqual(l.call(event,{'x':1},lambda:self.fail('callback repeated')),{'v':2})
            self.assertEqual(l.snapshot(),dict(total=1,unresolved=0,cap=1));l.close()
    def test_reservation_survives_failure(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'l.sqlite';l=Ledger(p,'a'*64,2)
            with self.assertRaises(ValueError):l.call({'event_id':'a'},{},lambda:(_ for _ in ()).throw(ValueError('simulated')))
            l.close();l=Ledger(p,'a'*64,2)
            with self.assertRaisesRegex(RuntimeError,'Unresolved'):l.call({'event_id':'a'},{},lambda:self.fail())
            self.assertEqual(l.snapshot()['unresolved'],1);l.close()
    def test_request_and_cap_immutable(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'l.sqlite';l=Ledger(p,'a'*64,1);l.close()
            for h,c in [('b'*64,1),('a'*64,2)]:
                with self.assertRaisesRegex(RuntimeError,'changed'):Ledger(p,h,c)
    def test_cap_cannot_overshoot(self):
        with tempfile.TemporaryDirectory()as d:
            l=Ledger(Path(d)/'l.sqlite','a'*64,1);l.call({'event_id':'a'},{},lambda:{})
            with self.assertRaisesRegex(RuntimeError,'cap exhausted'):l.call({'event_id':'b'},{},lambda:self.fail())
            l.close()
    def test_changed_request_or_event_cannot_reuse(self):
        with tempfile.TemporaryDirectory()as d:
            l=Ledger(Path(d)/'l.sqlite','a'*64,3);l.call({'event_id':'a','kind':'header'},{'wave':'x'},lambda:{})
            for e,r in [({'event_id':'a','kind':'body'},{'wave':'x'}),({'event_id':'a','kind':'header'},{'wave':'y'})]:
                with self.assertRaises(RuntimeError):l.call(e,r,lambda:self.fail())
            l.close()
    def test_parallel_reservations_global_cap(self):
        with tempfile.TemporaryDirectory()as d:
            p=str(Path(d)/'l.sqlite');l=Ledger(p,'a'*64,16);l.close()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4)as ex:
                values=list(ex.map(add_one,[(p,i)for i in range(16)]))
            self.assertEqual(sorted(x['decoded']for x in values),list(range(16)))
            l=Ledger(p,'a'*64,16);self.assertEqual(l.snapshot()['total'],16);l.close()
    def test_cache_key_uses_all_received_values(self):
        a=dict(receiver_state=dict(kind='tokens',m=1,K=0,prefix=[[1]],partial_values=[],order='raster'))
        b=dict(receiver_state=dict(kind='tokens',m=1,K=0,prefix=[[2]],partial_values=[],order='raster'))
        self.assertNotEqual(state_key(a),state_key(b))
        self.assertEqual(state_key(a),state_key(dict(a,source_id='ignored_by_RX_state_cache')))
    def test_scope_rejects_removed_full_protection(self):
        r=dict(schema='WCL_T2_WHOLE_PILOT_V1',snrs=[10,19],noise_seed=4101,source_count=100,
            profiles=[dict(wire_key=str(i),K=0,N=1024,allocation_modes=['full_budget']if i<7 else['nominal'])for i in range(106)],
            records=[dict(source_index=i)for i in range(100)],packet_cap=42400,frame_count=21200,workers=8)
        validate_request(r);r['profiles'][0]['allocation_modes']=['nominal']
        with self.assertRaisesRegex(RuntimeError,'seven'):validate_request(r)

if __name__=='__main__':unittest.main()
