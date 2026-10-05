"""CPU engineering tests using scripted test doubles, never a real decoder."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).absolute().parent
for p in (HERE, HERE.parent/'phy_codec', HERE.parent/'payload'):
    sys.path.insert(0, str(p))
import h_full_payload_cpu as core
from h64_catalog import catalogue, all_buckets, digest, token_count, SIZES
from test_h_payload_cpu import FakeBackend, FakeHeader, FakeLedger
import h64_phy as phy


def fixture():
    cat = catalogue(all_buckets(FakeBackend()))
    source_ids = [f'full_{i:04d}' for i in range(1000)]
    old, winners = [], []
    for snr in core.SNRS:
        for arm in core.ARMS:
            for m in (7, 8):
                identity = dict(arm=arm, target_m=m, nominal_rate='1/2')
                row = dict(**identity, q=4 if arm.startswith('H16') else 6, K=0,
                           candidate_id='WHOLE:'+digest(identity), policy_key=digest(identity),
                           snr_db=snr, slot=len(old), selection_rank=m-6)
                old.append(row)
                if m == 7:
                    winners.append(dict(row, selection_rank=1))
    parts = []
    for snr, m, ks, rate in ((13, 7, [81, 80, 79], '1/2'), (19, 8, [140, 139, 138], '5/6')):
        for k in ks:
            p = next(p for p in cat['profiles'] if p['q'] == 6 and p['m'] == m and p['K'] == k
                     and p['nominal_rate'] == rate and p['mode'] == 'raw')
            parts.append(dict(p, arm=core.PARTIAL_ARM, candidate_id='PARTIAL:'+p['profile_key'],
                              target_m=m, policy_key=p['profile_key'], token_count=token_count(m, k),
                              snr_db=snr, selection_rank=len(parts)%3+1, slot=len(parts)))
    shortlist = dict(status='H_EXPECTED_PSNR_SHORTLIST_FROZEN', ready_for_real_calibration=True,
                     source_ids=source_ids[::5], whole_candidates=old, partial_candidates=parts)
    selected = dict(status='H_INITIAL_TRUE200_SELECTION_COMPLETE', source_count=200, selected_count=8,
                    candidate_count=16, measured_frames=9600, new_candidate_search=False, development_used=False,
                    holdout_used=False, full1000_calibration_complete=False, source_ids=source_ids[::5],
                    selected_candidates=winners)
    partial = dict(status='FROZEN_PRESCREEN_REFERENCE_ONLY_NOT_RESELECTED', candidate_count=6,
                   partial_reselected=False, actual_partial_calibration_complete=False, partial_candidates=parts)
    protocol = dict(schema='H_CODEC_PROTOCOL_V1', N=1024, header_symbols=68, body_symbols=956, Es=2, channel='AWGN')
    args = dict(protocol=protocol, catalogue=cat, selected=selected, partial_reference=partial,
                shortlist=shortlist, source_ids=source_ids)
    return args


def seal(args):
    contract = dict(status='H_FULL_PAYLOAD_CPU_ENGINEERING_SEALED', execution_registration_sha256='a'*64,
                    core_source_sha256=core.file_sha(core.__file__),
                    initial_core_source_sha256=core.file_sha(core.initial.__file__),
                    engineering_choices_canonical_sha256=digest(core.ENGINEERING_CHOICES))
    for key, value in args.items():
        contract[key+'_canonical_sha256'] = digest(value)
    contract['public_schedule_canonical_sha256'] = digest(core.schedule(
        args['selected'], args['partial_reference'], args['shortlist'], args['catalogue'], args['source_ids']))
    return contract


class FullLedger(FakeLedger):
    limits = dict(core.PHASE_CAPS, initial_true200=20000, development=13200)


class FullCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = fixture()
        cls.context = core.prepare(**cls.args, contract=seal(cls.args))

    def setUp(self):
        self.backend, self.header, self.ledger = FakeBackend(), FakeHeader(), FullLedger()
        self.scales = [(np.arange(s*s, dtype=np.int64)+43*i)%4096 for i,s in enumerate(SIZES)]
        self.streams = {m: np.zeros(12*token_count(m)+2,dtype=np.uint8) for m in (6,7,8,9)}

    def run_frame(self, slot=0, **changes):
        args = dict(ctx=self.context, full_slot=slot, source_id='full_0999', source_index=999,
                    noise_seed=6101, scales=self.scales, arithmetic_bits=self.streams,
                    backend=self.backend, header=self.header, ledger=self.ledger)
        args.update(changes)
        return core.run_frame(**args)

    def test_schedule_preserves_all_winners_and_partial_references(self):
        s = self.context['schedule']
        self.assertEqual([r['full_slot'] for r in s], list(range(14)))
        self.assertEqual([r['phase'] for r in s], ['whole_calibration']*8+['partial_calibration']*6)
        self.assertEqual([r['candidate'] for r in s[:8]], sorted(self.args['selected']['selected_candidates'],key=lambda r:r['slot']))
        self.assertEqual([r['candidate'] for r in s[8:]], self.args['shortlist']['partial_candidates'])

    def test_missing_extra_or_altered_winners_and_partial_are_rejected(self):
        for section, field in [('selected','selected_candidates'),('partial_reference','partial_candidates')]:
            for op in ('missing','duplicate','changed'):
                a = copy.deepcopy(self.args)
                rows = a[section][field]
                if op == 'missing': rows.pop()
                elif op == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
                else: rows[0]['nominal_rate']='3/4'
                with self.assertRaises(ValueError):
                    core.schedule(a['selected'],a['partial_reference'],a['shortlist'],a['catalogue'],a['source_ids'])

    def test_new_core_requires_exact_seal_and_no_development(self):
        contract = seal(self.args)
        with self.assertRaisesRegex(ValueError,'seal'):
            core.prepare(**self.args,contract=dict(contract,status='DRAFT'))
        with self.assertRaisesRegex(ValueError,'source differs'):
            core.prepare(**self.args,contract=dict(contract,initial_core_source_sha256='0'*64))
        a=copy.deepcopy(self.args);a['selected']['development_used']=True
        with self.assertRaises(ValueError): core.prepare(**a,contract=seal(a))

    def test_shared_noise_all_methods_but_independent_initial_stage(self):
        whole=self.run_frame(2);part=self.run_frame(8)
        self.assertEqual(whole['noise_sha256'],part['noise_sha256'])
        self.assertNotEqual(whole['public_frame_counter'],part['public_frame_counter'])
        self.assertEqual(whole['scrambling_session'],part['scrambling_session'])
        noise=core.standard_noise('full_0999',13,6101)
        text=json.dumps([core.PROTOCOL,'full_calibration','full_0999',13,6101],separators=(',',':')).encode()
        seed=int.from_bytes(hashlib.sha256(text).digest()[:16],'little')
        np.testing.assert_array_equal(noise,np.random.Generator(np.random.PCG64(seed)).standard_normal((1024,2)))
        self.assertFalse(np.array_equal(noise,core.initial.standard_noise('full_0999',13,6101)))
        counters={core.frame_counter(s,i,n) for s in range(14) for i in range(1000) for n in core.SEEDS}
        self.assertEqual(counters,set(range(42000)))

    def test_whole_fallback_exactly_matches_frozen_initial_implementation(self):
        c=self.context['rows_by_slot'][1]
        self.streams[7]=np.zeros(1000,dtype=np.uint8)
        got=core.select_payload(self.context,c,self.scales,self.streams)
        old=core.initial.select_payload(c['candidate'],self.args['catalogue'],self.scales,self.streams)
        self.assertEqual(got['profile'],old['profile'])
        self.assertEqual(got['attempts'],old['attempts'])
        np.testing.assert_array_equal(got['payload'],old['payload'])

    def test_partial_is_exact_canonical_raster_no_fallback_and_charged_separately(self):
        frame=self.run_frame(11)
        self.assertEqual(frame['tx']['K'],140)
        self.assertEqual(len(frame['received_raw_tokens']),395)
        self.assertEqual(frame['received_raw_tokens'],np.concatenate(self.scales)[:395].tolist())
        self.assertFalse(frame['tx']['fell_back'])
        self.assertEqual([x['phase'] for x in self.ledger.events],['partial_calibration']*2)
        self.assertEqual(frame['total_symbols'],1024)

    def test_wrong_known_header_uses_received_layout_not_selected_partial(self):
        p=next(p for p in self.args['catalogue']['profiles'] if p['q']==4 and p['m']==6
               and p['K']==0 and p['mode']=='raw' and p['nominal_rate']=='1/2')
        bits=phy.raw_payload(self.scales,6);bits[0]^=1
        self.header.force_id=p['profile_id'];self.backend.decoded_override=phy.pack_body(bits,p)
        frame=self.run_frame(8)
        self.assertEqual(self.backend.decode_calls,[(p['k'],p['n'],p['q'])])
        self.assertEqual(frame['received_raw_tokens'],phy.raw_tokens(bits,p).tolist())
        self.assertTrue(frame['evaluation_only']['accepted_wire_mismatch'])
        self.assertFalse(frame['gray'])

    def test_wrong_accepted_arithmetic_remains_pending_even_for_partial_tx(self):
        p=next(p for p in self.args['catalogue']['profiles'] if p['q']==6 and p['m']==6
               and p['K']==0 and p['mode']=='arithmetic' and p['nominal_rate']=='1/2')
        self.header.force_id=p['profile_id'];self.backend.decoded_override=phy.pack_body(np.zeros(20,dtype=np.uint8),p)
        frame=self.run_frame(8)
        self.assertEqual(frame['rx_source_status'],'ARITHMETIC_CANONICAL_RX_REQUIRED')
        self.assertIsNone(frame['gray']);self.assertFalse(frame['neural_metrics_run'])

    def test_header_reject_charges_one_and_failed_body_remains_failed(self):
        self.header.reject=True
        frame=self.run_frame(8)
        self.assertTrue(frame['gray']);self.assertEqual(len(self.ledger.events),1)
        self.header.reject=False;self.backend.raise_decode=True
        with self.assertRaisesRegex(RuntimeError,'scripted CPU'):self.run_frame(9)
        self.assertEqual(self.ledger.events[-1]['status'],'FAILED')

    def test_budget_reallocation_and_wrong_source_rejected_before_decode(self):
        self.ledger.limits=dict(FullLedger.limits,whole_calibration=49000)
        with self.assertRaisesRegex(ValueError,'separate'):self.run_frame()
        self.assertFalse(self.ledger.events)
        self.ledger.limits=FullLedger.limits
        with self.assertRaisesRegex(ValueError,'source order'):self.run_frame(source_id='full_0000')
        self.assertFalse(self.ledger.events)


if __name__=='__main__': unittest.main()
