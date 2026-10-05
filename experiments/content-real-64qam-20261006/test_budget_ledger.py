"""Synthetic callbacks only; these tests do not run any physical decoder."""
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from budget_ledger import BudgetLedger, BudgetError, canonical


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'H.sqlite'
        self.identities={i:dict(pid=i,start_ticks=100+i,uid=1002,argv=['python','worker',str(i)]) for i in (1,2)}
    def reader(self,pid):
        if pid not in self.identities:raise ProcessLookupError(pid)
        return self.identities[pid]
    def ledger(self,pid=1,limits=None):
        result=BudgetLedger(self.path,'registered','H',limits or dict(qualify=2,development=3),self.identities[pid],self.reader)
        result.register_configuration('p',dict(k=100,n=200,q=4))
        return result
    def test_precharge_and_exact_reuse(self):
        ledger=self.ledger();calls=[]
        def decode():
            self.assertEqual(ledger.snapshot()['charged'],1)
            calls.append(1);return dict(bits=[0,1],crc_accept=False)
        value=ledger.decode_once('qualify','e','header','p',{'seed':1},decode)
        self.assertEqual(value,ledger.decode_once('qualify','e','header','p',{'seed':1},decode))
        self.assertEqual(len(calls),1)
        with self.assertRaises(BudgetError):ledger.decode_once('qualify','e','header','p',{'seed':2},decode)
    def test_phase_cannot_borrow_development(self):
        ledger=self.ledger()
        for i in range(2):ledger.decode_once('qualify',str(i),'body','p',{},lambda:{})
        with self.assertRaisesRegex(BudgetError,'phase budget'):ledger.decode_once('qualify','extra','body','p',{},lambda:{})
        self.assertEqual(ledger.snapshot()['phase_remaining']['development'],3)
        ledger.decode_once('development','dev','body','p',{},lambda:{})
    def test_failure_charged_and_blocks_new_work(self):
        ledger=self.ledger()
        def fail():raise RuntimeError('decoder error')
        with self.assertRaises(RuntimeError):ledger.decode_once('qualify','failed','body','p',{},fail)
        self.assertEqual(ledger.snapshot()['charged'],1)
        with self.assertRaisesRegex(BudgetError,'Failed charged'):ledger.decode_once('development','next','body','p',{},lambda:{})
        with self.assertRaises(BudgetError):ledger.assert_quiescent()
    def test_branch_or_registration_cannot_change(self):
        self.ledger()
        with self.assertRaises(BudgetError):BudgetLedger(self.path,'registered','C',dict(qualify=2,development=3),self.identities[1],self.reader)
        with self.assertRaises(BudgetError):BudgetLedger(self.path,'changed','H',dict(qualify=2,development=3),self.identities[1],self.reader)
    def test_orphan_and_PID_reuse_do_not_allow_restart(self):
        ledger=self.ledger()
        with ledger.transaction() as c:
            c.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',('old','qualify','body','p','sha','RESERVED',canonical(self.identities[2]),0,None,None,None,None))
            c.execute("UPDATE counters SET charged=1 WHERE phase='qualify'")
        self.identities[2]=dict(self.identities[2],start_ticks=999)
        with self.assertRaisesRegex(BudgetError,'identity changed'):ledger.decode_once('qualify','new','body','p',{},lambda:{})
        self.identities.pop(2)
        with self.assertRaisesRegex(BudgetError,'Orphaned'):ledger.decode_once('qualify','new','body','p',{},lambda:{})
        self.assertEqual(ledger.snapshot()['charged'],1)
    def test_two_live_workers_atomic_cap(self):
        limits=dict(qualify=1,development=2)
        a=self.ledger(1,limits);b=self.ledger(2,limits)
        entered=threading.Event();release=threading.Event();errors=[]
        def callback():entered.set();self.assertTrue(release.wait(5));return {'done':1}
        def run():
            try:a.decode_once('qualify','a','body','p',{},callback)
            except BaseException as e:errors.append(e)
        thread=threading.Thread(target=run);thread.start();self.assertTrue(entered.wait(5))
        try:
            with self.assertRaisesRegex(BudgetError,'phase budget'):b.decode_once('qualify','b','body','p',{},lambda:{})
            b.decode_once('development','b','header','p',{},lambda:{})
        finally:release.set();thread.join(5)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertEqual(a.assert_quiescent()['charged'],2)
    def test_result_tampering_rejected(self):
        ledger=self.ledger();ledger.decode_once('qualify','a','body','p',{},lambda:{'bits':[1]})
        with ledger.transaction() as c:c.execute("UPDATE events SET result='{}' WHERE event_id='a'")
        with self.assertRaisesRegex(BudgetError,'modified'):ledger.decode_once('qualify','a','body','p',{},lambda:{})
    def test_engineering_reserve_not_available_to_normal_science(self):
        ledger=self.ledger(limits=dict(qualify=1,development=2,engineering_reserve=10))
        with self.assertRaisesRegex(BudgetError,'registered recovery'):
            ledger.decode_once('engineering_reserve','a','body','p',{},lambda:{})
        self.assertEqual(ledger.snapshot()['charged'],0)
    def test_phase_counter_corruption_cannot_shift_reserved_budget(self):
        ledger=self.ledger();ledger.decode_once('qualify','a','body','p',{},lambda:{})
        with ledger.transaction() as c:
            c.execute("UPDATE counters SET charged=0 WHERE phase='qualify'")
            c.execute("UPDATE counters SET charged=1 WHERE phase='development'")
        with self.assertRaisesRegex(BudgetError,'Counter/event'):ledger.snapshot()


if __name__=='__main__':unittest.main()
