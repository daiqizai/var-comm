import contextlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from h_coarse_driver import (generator,public_counter,worker_layout_indices,wilson,energy_summary,
    summarize_rows,measure_cell,audit_worker_ledger,digest,csv_write)
from h64_catalog import PROTOCOL,all_buckets
from h64_phy import pack_body,array_sha,crc_ok
from test_h64_core import FakeBackend,RecordingLedger


class CoarseTests(unittest.TestCase):
    def buckets(self):
        rows=all_buckets()
        for index,b in enumerate(rows):
            b.update(admission='ADMITTED',layout={'layout_id':'TEST_ONLY' if index==0 else f'TEST{index}'})
        return rows
    def test_two_worker_cartesian_partition_and_public_counter_uniqueness(self):
        self.assertEqual(worker_layout_indices(0),[0,1,2,3])
        self.assertEqual(worker_layout_indices(1),[4,5,6,7])
        values=[public_counter(i,s,j) for i in range(8) for s in (13,16,19) for j in range(2048)]
        self.assertEqual(len(values),49152);self.assertEqual(len(set(values)),49152)
        with self.assertRaises(RuntimeError):worker_layout_indices(2)
        with self.assertRaises(RuntimeError):public_counter(0,13,2048)
    def test_seed_domains_are_repeatable_and_independent(self):
        a=generator(PROTOCOL,'layout',13,7,'payload').integers(0,2,1000)
        np.testing.assert_array_equal(a,generator(PROTOCOL,'layout',13,7,'payload').integers(0,2,1000))
        for changed in [('layout',13,7,'noise'),('other',13,7,'payload'),('layout',19,7,'payload'),('layout',13,8,'payload')]:
            self.assertFalse(np.array_equal(a,generator(PROTOCOL,*changed).integers(0,2,1000)))
    def test_cell_actual_packet_count_and_exact_replay(self):
        b=self.buckets()[0];backend=FakeBackend();ledger=RecordingLedger();boundaries=[]
        rows,summary=measure_cell(backend,ledger,b,0,19,lambda:boundaries.append(1),blocks=3)
        self.assertEqual(len(ledger.events),3);self.assertEqual(backend.calls,3)
        self.assertEqual(len(boundaries),3)
        self.assertEqual(summary['trials'],3)
        self.assertEqual(summary['correct']+summary['rejected']+summary['undetected'],3)
        replay,replayed=measure_cell(backend,ledger,b,0,19,lambda:None,blocks=3)
        self.assertEqual(rows,replay);self.assertEqual(summary,replayed);self.assertEqual(backend.calls,3)
    def test_wrong_accepted_and_invalid_parser_are_still_U_not_C_or_R(self):
        class WrongAcceptedLedger(RecordingLedger):
            def decode_once(self,phase,event_id,kind,key,request,decode):
                value=super().decode_once(phase,event_id,kind,key,request,decode)
                value['crc_accepted']=True;value['parser_accepted']=False
                value['decoded_bits'][17]^=1
                return value
        rows,summary=measure_cell(FakeBackend(),WrongAcceptedLedger(),self.buckets()[0],0,19,lambda:None,blocks=2)
        self.assertEqual(summary['undetected'],2);self.assertEqual(summary['parser_invalid'],2)
        self.assertEqual(summary['correct'],0);self.assertEqual(summary['rejected'],0)
    def test_stop_before_call_and_failed_callback_is_not_retried(self):
        ledger=RecordingLedger();backend=FakeBackend()
        def stop():raise RuntimeError('STOP')
        with self.assertRaisesRegex(RuntimeError,'STOP'):
            measure_cell(backend,ledger,self.buckets()[0],0,13,stop,blocks=2)
        self.assertEqual(backend.calls,0);self.assertEqual(ledger.events,[])
        backend.raise_decode=True
        with self.assertRaisesRegex(RuntimeError,'decoder failure'):
            measure_cell(backend,ledger,self.buckets()[0],0,13,lambda:None,blocks=2)
        self.assertEqual(len(ledger.events),1);self.assertEqual(ledger.events[0]['status'],'RESERVED')
    def test_wilson_energy_and_csv_output(self):
        self.assertLess(wilson(0,2048)[1],.01)
        self.assertGreater(wilson(0,2048)[1],0)
        self.assertLess(wilson(2048,2048)[0],1)
        energy=energy_summary([1,2,3]);self.assertEqual(energy['mean'],2)
        self.assertEqual(energy['minimum'],1);self.assertEqual(energy['maximum'],3)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'x.csv';csv_write(p,[{'i':0,'correct':1},{'i':1,'correct':0}])
            self.assertNotIn(b'\r',p.read_bytes())
    def test_ledger_audit_accepts_only_exact_completed_worker_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger.sqlite';buckets=self.buckets();payload={'decoded_bits':[0,1],'crc_accepted':False}
            with contextlib.closing(sqlite3.connect(path)) as c:
                c.execute('CREATE TABLE events(event_id TEXT,phase TEXT,kind TEXT,status TEXT,result TEXT,result_sha TEXT)')
                for s in (13,16,19):
                    c.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(f'HCOARSE:TEST_ONLY:{s}:0','coarse','body','COMPLETE',json.dumps(payload),digest(payload)))
                # Peer live reservation does not invalidate a finished worker.
                c.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',('HCOARSE:TEST4:13:0','coarse','body','RESERVED',None,None))
                c.commit()
            result=audit_worker_ledger(path,buckets,[0],blocks=1)
            self.assertEqual(result['event_count'],3)
            with contextlib.closing(sqlite3.connect(path)) as c:
                c.execute("UPDATE events SET status='RESERVED' WHERE event_id='HCOARSE:TEST_ONLY:13:0'")
                c.commit()
            with self.assertRaisesRegex(RuntimeError,'unresolved'):
                audit_worker_ledger(path,buckets,[0],blocks=1)


if __name__=='__main__':unittest.main()
