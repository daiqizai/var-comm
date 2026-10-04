"""Synthetic CPU contract tests, never a real LDPC/physical qualification."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import phy_lookup as p
from uep_common import read,identity


def spec():
    return dict(phy_key='body_fixture',kind='body',source_bits=24,crc_bits=16,tail_bits=0,information_bits=40,
        transmitted_bits=128,q=2,modulation='QPSK',layout={})


def events(count,correct=None,rejected=None,undetected=None):
    good=np.ones(count,dtype=bool) if correct is None else np.asarray(correct,dtype=bool)
    reject=~good if rejected is None else np.asarray(rejected,dtype=bool)
    hidden=np.zeros(count,dtype=bool) if undetected is None else np.asarray(undetected,dtype=bool)
    return dict(correct=good,rejected=reject,undetected=hidden,actual_E=np.full(count,128.))


class ProbabilityTests(unittest.TestCase):
    def test_shards_cover_each_config_snr_once_header_only_on_zero(self):
        catalog={f'body_{i}':dict(kind='body') for i in range(17)}
        catalog['header']=dict(kind='header');all_points=[(key,snr) for key in sorted(catalog) for snr in p.SNRS]
        parts=[p.shard_points(catalog,all_points,index,8) for index in range(8)]
        union=[point for part in parts for point in part]
        self.assertEqual(set(union),set(all_points));self.assertEqual(len(union),len(set(union)))
        self.assertEqual({snr for key,snr in parts[0] if key=='header'},set(p.SNRS))
        self.assertTrue(all(key!='header' for part in parts[1:] for key,_ in part))
        for index in range(17):
            owners=[i for i,part in enumerate(parts) if any(key==f'body_{index}' for key,_ in part)]
            self.assertEqual(len(owners),1)

    def test_refine_shard_is_subset_and_does_not_reassign_after_narrowing(self):
        catalog={f'body_{i}':dict(kind='body') for i in range(7)};catalog['header']=dict(kind='header')
        all_points=[(key,snr) for key in sorted(catalog) for snr in p.SNRS];refine=[('body_4',7),('header',13)]
        for index in range(3):
            self.assertEqual(p.shard_points(catalog,refine,index,3),[x for x in p.shard_points(catalog,all_points,index,3) if x in refine])

    def test_wilson_zero_errors_has_nonzero_true_failure_upper_bound(self):
        lo,hi=p.wilson(256,256)
        self.assertLess(lo,1);self.assertAlmostEqual(hi,1);self.assertGreater(1-lo,.01)
        zero=p.wilson(0,256);self.assertEqual(zero[0],0);self.assertGreater(zero[1],0)

    def test_undetected_errors_are_not_counted_correct(self):
        meta=events(3,[True,False,False],[False,True,False],[False,False,True])
        row=p.batch_receipt(meta,0,3)
        self.assertEqual((row['n_correct'],row['n_reject'],row['n_undetected']),(1,1,1))

    def test_nonpartition_or_non_boolean_events_are_rejected(self):
        with self.assertRaises(RuntimeError):p.batch_receipt(events(2,[True,False],[True,True]),0,2)
        bad=events(2);bad['correct']=np.ones(2,dtype=np.uint8)
        with self.assertRaises(RuntimeError):p.batch_receipt(bad,0,2)

    def test_public_counter_and_payload_stable_across_batch_boundaries(self):
        s=spec();full,c=p.random_payloads(s,7,0,256)
        for batch in (32,64,128):
            chunks=[p.random_payloads(s,7,start,batch) for start in range(0,256,batch)]
            np.testing.assert_array_equal(full,np.concatenate([x[0] for x in chunks]))
            self.assertEqual(c,sum([x[1] for x in chunks],[]))
        self.assertEqual(len(set(c)),256)
        self.assertNotEqual(c,p.counters_for(s['phy_key'],13,0,256))
        self.assertNotEqual(c,p.counters_for('other_physical_layout',7,0,256))

    def test_only_qualified_main_budget_catalog(self):
        group=spec();group.pop('kind');group.pop('q')
        profile=dict(N=1024,encoder_qualified=True,groups=[group],header_phy_key='header_fixture')
        catalog=p.physical_catalog([profile]);self.assertEqual(set(catalog),{'body_fixture','header_fixture'})
        profile['N']=2048
        with self.assertRaises(RuntimeError):p.physical_catalog([profile])


class HeaderTests(unittest.TestCase):
    def test_header_transmit_awgn_decode_and_false_accept_are_real_calls(self):
        class HeaderFixture:
            def __init__(self):self.calls=0;self.ids=[];self.seen=[]
            def transmit(self,pid):self.pid=pid;self.ids.append(pid);return np.ones((68,2),dtype=np.float32)
            def receive(self,y,snr,codebook=None):
                self.seen.append(y.copy());kind=self.calls%3;self.calls+=1
                return dict(header_ok=kind!=1,profile_id=self.pid if kind==0 else None if kind==1 else (self.pid+1)%4096)
        header=HeaderFixture();result=p.simulate_header(header,dict(phy_key='header_fixture'),7,0,3)
        self.assertEqual(header.calls,3);self.assertTrue(all(0<=pid<4096 for pid in header.ids))
        self.assertTrue(all(not np.array_equal(y,np.ones((68,2))) for y in header.seen))
        self.assertEqual(p.batch_receipt(result,0,3)['n_undetected'],1)


class ResumeTests(unittest.TestCase):
    def test_three_batch_sizes_have_same_deterministic_crc_counts(self):
        counts=[]
        def simulate(_,__,s,snr,start,count):
            index=np.arange(start,start+count)
            return events(count,index%7>1,index%7==0,index%7==1)
        for batch in (32,64,128):
            with tempfile.TemporaryDirectory() as directory:
                runner=p.LookupRunner({'body_fixture':spec()},directory,'fixture',None,None,batch,simulate=simulate,synthetic=True)
                counts.append(runner.run_point('body_fixture',7)['counts'])
        self.assertTrue(all(x==counts[0] for x in counts))

    def test_safe_stop_commits_batch_and_resume_uses_next_counter(self):
        calls=[]
        def simulate(_,__,s,snr,start,count):calls.append((start,count));return events(count)
        with tempfile.TemporaryDirectory() as directory:
            runner=p.LookupRunner({'body_fixture':spec()},directory,'fixture',None,None,128,
                stop=lambda:len(calls)>=1,simulate=simulate,synthetic=True)
            with self.assertRaises(p.StopRequested):runner.run_point('body_fixture',7)
            runner.stop=lambda:False;cp=runner.run_point('body_fixture',7)
            self.assertEqual(calls,[(0,128),(128,128)]);self.assertEqual(cp['counts']['n_blocks'],256)
            again=runner.run_point('body_fixture',7);self.assertEqual(cp,again);self.assertEqual(len(calls),2)
            bundle=runner.export('SYNTHETIC_FIXTURE');self.assertTrue(bundle['synthetic'])
            self.assertEqual(bundle['rows'][0]['p_correct'],1)
            self.assertLess(bundle['rows'][0]['p_correct_ci'][0],1)

    def test_refine_stops_at_max_blocks_without_errors(self):
        def simulate(_,__,s,snr,start,count):return events(count)
        with tempfile.TemporaryDirectory() as directory:
            runner=p.LookupRunner({'body_fixture':spec()},directory,'fixture',None,None,128,simulate=simulate,synthetic=True)
            runner.run_point('body_fixture',7);cp=runner.run_point('body_fixture',7,refine=True)
            self.assertEqual(cp['counts']['n_blocks'],20000);self.assertEqual(cp['counts']['n_correct'],20000)
            self.assertEqual(cp['batches'][-1]['n_blocks'],32)
            path=Path(directory)/'points'/identity(dict(phy_key='body_fixture',snr_db=7))/'coarse.json'
            self.assertEqual(read(path)['counts']['n_blocks'],256)

    def test_refine_counts_reject_plus_undetected_errors(self):
        def simulate(_,__,s,snr,start,count):
            idx=np.arange(start,start+count);hidden=idx%4==0;rejected=idx%4==1
            return events(count,~(hidden|rejected),rejected,hidden)
        with tempfile.TemporaryDirectory() as directory:
            runner=p.LookupRunner({'body_fixture':spec()},directory,'fixture',None,None,64,simulate=simulate,synthetic=True)
            cp=runner.run_point('body_fixture',7,refine=True)
            self.assertEqual(cp['counts']['n_blocks'],256)
            self.assertEqual(cp['counts']['n_reject']+cp['counts']['n_undetected'],128)

    def test_checkpoint_count_or_counter_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=p.LookupRunner({'body_fixture':spec()},directory,'fixture',None,None,64,
                simulate=lambda _,__,s,snr,start,count:events(count),synthetic=True)
            cp=runner.run_point('body_fixture',7);bad=copy.deepcopy(cp);bad['batches'][1]['start']=0
            bad['payload_sha256']=identity({k:v for k,v in bad.items() if k!='payload_sha256'})
            with self.assertRaises(RuntimeError):p.verify_checkpoint(bad,'fixture',cp['point'])

    def test_test_simulator_cannot_enter_scientific_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):p.LookupRunner({'body_fixture':spec()},directory,'fixture',None,None,64,
                simulate=lambda *_:None,synthetic=False)

    def test_simulation_exception_is_not_swallowed(self):
        def failed(*_):raise RuntimeError('actual decoder failed')
        with tempfile.TemporaryDirectory() as directory:
            runner=p.LookupRunner({'body_fixture':spec()},directory,'fixture',None,None,64,simulate=failed,synthetic=True)
            with self.assertRaisesRegex(RuntimeError,'actual decoder failed'):runner.run_point('body_fixture',7)


if __name__=='__main__':unittest.main()
