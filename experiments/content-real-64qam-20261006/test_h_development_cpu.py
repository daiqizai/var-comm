"""Synthetic CPU engineering tests. No real channel decoder or image is used."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

HERE=Path(__file__).absolute().parent
for path in (HERE,HERE.parent/'phy_codec',HERE.parent/'payload',HERE.parent/'full_calibration',HERE.parent/'runner'):
    sys.path.insert(0,str(path))
import h_development_cpu as core
from test_h_full_payload_cpu import fixture as calibration_fixture
from test_h_payload_cpu import FakeBackend,FakeHeader,FakeLedger
from h64_catalog import SIZES,token_count,digest
import h64_phy as phy
from budget_ledger import BudgetLedger,BudgetError


def fixture():
    old=calibration_fixture()
    protocol=dict(old['protocol'],status='FROZEN_BEFORE_DATA',main_snrs_db=[13,19],
        population={'development_noise_seeds':[6201,6202,6203]},
        development={'maximum_packet_calls':13200},calibration={'no_development_selection':True},
        matched_attribution=dict(m=7,K=0,nominal_rate='1/2',arms=list(core.ARMS)))
    oldrows=core.calibrated.schedule(old['selected'],old['partial_reference'],old['shortlist'],old['catalogue'],old['source_ids'])
    finalized=dict(status='H_FULL1000_CALIBRATION_POLICIES_FINALIZED',schema='H_FULL1000_SELECTION_V1',
        source_count=1000,measured_frames=42000,whole_policy_reselected=False,new_candidate_search=False,
        new_packet_decodes=0,new_visual_inference=0,development_used=False,holdout_used=False,
        source_ids=old['source_ids'],whole_candidates=copy.deepcopy(old['selected']['selected_candidates']),
        selected_partial=[dict(candidate=copy.deepcopy(oldrows[s]['candidate']),full_slot=s,
            selection_scope='ACTUAL_FULL1000_THREE_FROZEN_PARTIAL_CANDIDATES') for s in (8,11)],
        fixed_source_controls=core.selection_rules.fixed_controls(old['catalogue'],protocol,
            {(r['candidate']['policy_key'],r['candidate']['snr_db']) for r in oldrows}))
    pop=dict(stage='m1_development',calibration_or_development='m1_development',
        source_ids=[f'dev_{i:03d}' for i in range(100)],preprocessing_ids=['a'*64]*100,
        data_bindings=[dict(index=i,rgb_sha256='b'*64,source_npz_sha256='c'*64) for i in range(100)])
    return dict(protocol=protocol,catalogue=old['catalogue'],finalized=finalized,selected=old['selected'],
        partial_reference=old['partial_reference'],shortlist=old['shortlist'],development_registration=pop)


def seal(a):
    c=dict(status='H_DEVELOPMENT_CPU_ENGINEERING_SEALED',execution_registration_sha256='d'*64,
        engineering_choices_canonical_sha256=digest(core.ENGINEERING_CHOICES))
    for k,v in a.items():c[k+'_canonical_sha256']=digest(v)
    for k,p in dict(core=core.__file__,initial_core=core.initial.__file__,calibrated_core=core.calibrated.__file__,
                   fixed_control_rules=core.selection_rules.__file__,phy=phy.__file__).items():
        c[k+'_source_sha256']=core.file_sha(p)
    c['public_schedule_canonical_sha256']=digest(core.schedule(a['finalized'],a['selected'],a['partial_reference'],a['shortlist'],a['catalogue'],a['protocol']))
    return c


class DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=fixture();cls.ctx=core.prepare(**cls.args,contract=seal(cls.args))

    def setUp(self):
        self.backend,self.header,self.ledger=FakeBackend(),FakeHeader(),FakeLedger()
        self.scales=[(np.arange(s*s,dtype=np.int64)+13*i)%4096 for i,s in enumerate(SIZES)]
        self.streams={m:np.zeros(12*token_count(m)+2,dtype=np.uint8) for m in (6,7,8,9)}

    def frame(self,slot=0,**changes):
        a=dict(ctx=self.ctx,development_slot=slot,source_id='dev_099',source_index=99,noise_seed=6201,
               scales=self.scales,arithmetic_bits=self.streams,backend=self.backend,header=self.header,ledger=self.ledger)
        a.update(changes);return core.run_frame(**a)

    def test_exact18_schedule_no_ranking_and_MAIN_budget_reserved(self):
        rows=self.ctx['schedule']
        self.assertEqual([r['development_slot'] for r in rows],list(range(18)))
        self.assertEqual([r['role'] for r in rows],['H_WHOLE_SYSTEM']*8+['H_RAW_PARTIAL_SYSTEM']*2+['H_FIXED_M7_ATTRIBUTION']*8)
        self.assertEqual([r['candidate'] for r in rows[:8]],sorted(self.args['selected']['selected_candidates'],key=lambda x:x['slot']))
        self.assertEqual(18*100*3*2,core.H_MAX_CALLS)
        self.assertEqual(4*100*3*2,core.MAIN_RESERVED_CALLS)
        self.assertEqual(core.H_MAX_CALLS+core.MAIN_RESERVED_CALLS,13200)
        changed=copy.deepcopy(self.args);changed['finalized']['all_candidate_summaries']=[{'score':1e100}]
        self.assertEqual(core.prepare(**changed,contract=seal(changed))['schedule'],rows)

    def test_cannot_reselect_whole_or_invent_partial_or_fixed_control(self):
        for field in ('whole_candidates','selected_partial','fixed_source_controls'):
            a=copy.deepcopy(self.args)
            if field=='selected_partial':a['finalized'][field][0]['candidate']['K']-=1
            else:a['finalized'][field][0]['target_m']=6
            with self.assertRaises(ValueError):core.prepare(**a,contract=seal(a))

    def test_population_and_original_metadata_rejected_on_role_duplicates_order_or_sha(self):
        for change in ('role','duplicate','binding','hash'):
            a=copy.deepcopy(self.args);p=a['development_registration']
            if change=='role':p['stage']='m1_calibration'
            elif change=='duplicate':p['source_ids'][1]=p['source_ids'][0]
            elif change=='binding':p['data_bindings'][1]['index']=0
            else:p['preprocessing_ids'][0]='bad'
            with self.assertRaises(ValueError):core.prepare(**a,contract=seal(a))
        with self.assertRaisesRegex(ValueError,'source order'):self.frame(source_id='dev_000')
        self.assertFalse(self.ledger.events)

    def test_seal_and_future_main_slots_cannot_decode(self):
        with self.assertRaisesRegex(ValueError,'seal'):core.prepare(**self.args,contract={'status':'DRAFT'})
        c=seal(self.args);c['phy_source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source changed'):core.prepare(**self.args,contract=c)
        for slot in core.MAIN_SLOTS:
            with self.assertRaises(ValueError):self.frame(slot)
            with self.assertRaises(ValueError):core.FrameLedger(self.ledger,slot,0,6201)
        self.ledger.limits=dict(self.ledger.limits,development=13201)
        with self.assertRaises(ValueError):self.frame()
        self.assertFalse(self.ledger.events)

    def test_common_noise_fresh_stage_and_public_counter_bijection(self):
        noise=core.standard_noise('dev_099',13,6201)
        raw=json.dumps([core.PROTOCOL,'development','dev_099',13,6201],separators=(',',':')).encode()
        seed=int.from_bytes(hashlib.sha256(raw).digest()[:16],'little')
        np.testing.assert_array_equal(noise,np.random.Generator(np.random.PCG64(seed)).standard_normal((1024,2)))
        self.assertFalse(np.array_equal(noise,core.calibrated.standard_noise('dev_099',13,6101)))
        frames=[self.frame(s) for s in (0,8,10)]
        self.assertEqual(len({r['noise_sha256'] for r in frames}),1)
        self.assertEqual(len({r['event_id'] for r in frames}),3)
        self.assertEqual(len({r['public_frame_counter'] for r in frames}),3)
        self.assertEqual({core.frame_counter(s,i,n) for s in range(22) for i in range(100) for n in core.SEEDS},set(range(6600)))
        with self.assertRaises(ValueError):self.frame(noise_seed=6101)

    def test_event_wrapper_rejects_other_frame_phase_and_kind_without_call(self):
        bounded=core.FrameLedger(self.ledger,0,0,6201)
        for phase,event,kind in [('initial_true200',bounded.event+':body','body'),
            ('development',bounded.event+':header','body'),('development',core.frame_event(1,0,6201)+':body','body')]:
            with self.assertRaises(ValueError):bounded.decode_once(phase,event,kind,'key',{},lambda:{})
        self.assertFalse(self.ledger.events)

    def test_whole_uses_exact_old_fallback_and_fixedm7_never_falls_back(self):
        c=self.ctx['schedule'][1];self.streams[7]=np.zeros(20,dtype=np.uint8)
        new=core.select_payload(self.ctx,c,self.scales,self.streams)
        old=core.initial.select_payload(c['candidate'],self.args['catalogue'],self.scales,self.streams)
        self.assertEqual(new['profile'],old['profile']);self.assertEqual(new['attempts'],old['attempts'])
        np.testing.assert_array_equal(new['payload'],old['payload'])
        fixed=self.frame(11);self.assertEqual(fixed['tx']['actual_m'],7)
        self.assertFalse(fixed['tx']['fell_back']);self.assertEqual(len(fixed['tx']['attempts']),1)
        self.assertEqual(fixed['tx']['mode'],'arithmetic')
        self.streams[7]=phy.raw_payload(self.scales,7)
        tie=core.select_payload(self.ctx,self.ctx['schedule'][11],self.scales,self.streams)
        self.assertEqual(tie['profile']['mode'],'raw')

    def test_partial_exact_raster_budget_and_public_ID_paid(self):
        f=self.frame(9)
        self.assertEqual(f['tx']['K'],140);self.assertEqual(f['received_raw_tokens'],np.concatenate(self.scales)[:395].tolist())
        self.assertEqual(self.header.sent_ids,[f['tx']['profile_id']])
        self.assertEqual([x['phase'] for x in self.ledger.events],['development']*2)
        self.assertEqual(f['logical_packet_events'],2);self.assertFalse(f['tx']['fell_back'])

    def test_wrong_known_ID_determines_actual_RX_not_TX_source(self):
        p=next(p for p in self.args['catalogue']['profiles'] if (p['q'],p['m'],p['K'],p['mode'],p['nominal_rate'])==(4,6,0,'raw','1/2'))
        bits=phy.raw_payload(self.scales,6);bits[0]^=1
        self.header.force_id=p['profile_id'];self.backend.decoded_override=phy.pack_body(bits,p)
        f=self.frame(8)
        self.assertEqual(f['received_raw_tokens'],phy.raw_tokens(bits,p).tolist())
        self.assertEqual(self.backend.decode_calls,[(p['k'],p['n'],p['q'])]);self.assertFalse(f['gray'])
        self.assertTrue(f['evaluation_only']['accepted_wire_mismatch'])

    def test_wrong_accepted_arithmetic_kept_for_independent_RX(self):
        p=next(p for p in self.args['catalogue']['profiles'] if (p['q'],p['m'],p['K'],p['mode'],p['nominal_rate'])==(6,6,0,'arithmetic','1/2'))
        self.header.force_id=p['profile_id'];self.backend.decoded_override=phy.pack_body(np.zeros(20,dtype=np.uint8),p)
        f=self.frame(8)
        self.assertEqual(f['rx_source_status'],'ARITHMETIC_CANONICAL_RX_REQUIRED');self.assertIsNone(f['gray'])
        self.assertFalse(f['arithmetic_source_decode_run']);self.assertFalse(f['policy_selection'])

    def test_reject_one_call_failed_body_charged_unavailable_not_replaced(self):
        self.header.reject=True;f=self.frame(0)
        self.assertEqual(f['logical_packet_events'],1);self.assertTrue(f['gray'])
        self.header.reject=False;self.backend.raise_decode=True
        with self.assertRaisesRegex(RuntimeError,'scripted CPU'):self.frame(1)
        self.assertEqual(self.ledger.events[-1]['status'],'FAILED')
        e=copy.deepcopy(self.ctx['schedule'][10]);e['status']='NOT_FEASIBLE'
        with self.assertRaises(ValueError):core.select_payload(self.ctx,e,self.scales,self.streams)

    def test_real_atomic_ledger_wrapper_preserves_precharge_failure_and_no_retry(self):
        identity=dict(pid=100,start_ticks=500,uid=1002,argv=['test-only'])
        with tempfile.TemporaryDirectory() as directory:
            ledger=BudgetLedger(Path(directory)/'test.sqlite','test-registration','H',{'development':13200},
                                identity,lambda pid:identity)
            bounded=core.FrameLedger(ledger,0,0,6201);bounded.register_configuration('test',{'synthetic':True})
            observed=[]
            def fail():
                observed.append(ledger.snapshot()['charged'])
                raise RuntimeError('synthetic decoder failure')
            with self.assertRaises(RuntimeError):bounded.decode_once('development',bounded.event+':header','header','test',{},fail)
            self.assertEqual(observed,[1]);self.assertEqual(ledger.snapshot()['charged'],1)
            with self.assertRaisesRegex(BudgetError,'Unresolved charged'):
                bounded.decode_once('development',bounded.event+':header','header','test',{},lambda:{})
            other=core.FrameLedger(ledger,1,0,6201)
            with self.assertRaisesRegex(BudgetError,'Failed charged'):
                other.decode_once('development',other.event+':header','header','test',{},lambda:{})
            self.assertEqual(ledger.snapshot()['charged'],1)


if __name__=='__main__':unittest.main()
