"""CPU fixtures only. Real downloaded plan metadata is mandatory, not a skip."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import main_raw64_plan_only as p

HERE=Path(__file__).absolute().parent
RESEARCH=HERE.parents[1]
OLD=RESEARCH/'content_real_64qam_20261006'
SUMMARY=OLD/'current/main_plan_only/MAIN/plan_only_v1/result/summary'

def assets():
    if os.environ.get('MAIN64_TEST_ASSETS'):return p.read(os.environ['MAIN64_TEST_ASSETS'])
    return {k:str(v.absolute()) for k,v in dict(
        action_core=OLD/'main1000_prep/main_action_space.py',
        original_planner=RESEARCH/'prior_aware_uep_20261004/runtime/profiles.py',
        original_backend=RESEARCH/'prior_aware_uep_20261004/runtime/ldpc_backend.py',
        original_common=RESEARCH/'prior_aware_uep_20261004/runtime/uep_common.py',
        old_plan=HERE.parent/'assets/backend_plan_receipt.json',
        old_profiles=SUMMARY/'candidate_profiles.json',old_aliases=SUMMARY/'wire_aliases.json',
        old_qualification=RESEARCH/'paper_consolidation_20261005/reference/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json').items()}

class Tensor:
    def __init__(self,a):self.a=a
    def cpu(self):return self
    def numpy(self):return self.a

class Encoder:
    calls=0
    def __init__(self,k,n,**kw):
        type(self).calls+=1
        self.k=k;self.n=n;self.k_filler=4;self.k_ldpc=k+4;self.n_cb=n+100
        self.n_cb_comp=n+50;self._bg='bg2';self.z=2;self.n_ldpc=n+200
        self.out_int=Tensor(np.arange(n,dtype=np.int64))
    def forward(self,*a):raise AssertionError('Never execute a packet')

class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths=assets();cls.old=p.read(cls.paths['old_plan'])
        cls.original=p.load(cls.paths['original_planner'],'profiles')
        cls.core=p.load(cls.paths['action_core'],'original_main_finite_action_core')
        cls.ident=p.new_identity(cls.old['backend_identity'],p.sha(p.__file__))
    def config(self):
        c=dict(self.paths,schema='MAIN_RAW64_PLAN_ONLY_CONFIG_V1',scope='CPU_RESOURCE_METADATA_ONLY',
               candidate_K_rule='ORIGINAL_FINITE_QUARTILES_AND_BACKEND_MAX',new_packet_decodes=0,GPU_jobs=0)
        c['source_bindings']={self.paths[k]:p.sha(self.paths[k]) for k in p.FROZEN_SOURCES}
        c['source_bindings'][str(Path(p.__file__).absolute())]=p.sha(p.__file__)
        c['input_bindings']={self.paths[k]:p.sha(self.paths[k]) for k in ('old_plan','old_profiles','old_aliases','old_qualification')}
        return c
    def fakeplans(self):
        rows=[]
        for q in p.query_superset(('64QAM',)):
            r={k:q[k] for k in ('k','n','num_bits_per_symbol','query_id')}
            r.update(disposition='PLANNED',layout=p.encoder_layout(Encoder,r['k'],r['n'],6,self.ident));rows.append(r)
        return rows
    def test_query_full_source_domain_and_exact_counts(self):
        q=p.query_superset();self.assertEqual(len(q),3707)
        self.assertEqual({w:sum(r['num_bits_per_symbol']==w for r in q) for w in (2,4,6)},{2:613,4:1235,6:1859})
        self.assertEqual(max(r['source_token_count'] for r in q if r['num_bits_per_symbol']==6),397)
        self.assertIn((28,5736,6),{(r['k'],r['n'],r['num_bits_per_symbol']) for r in q})
        self.assertFalse(any(r['k']==12*424+16 and r['num_bits_per_symbol']==6 for r in q))
    def test_actual_original1848_reused_without_backend_initialization(self):
        self.assertEqual(p.sha(self.paths['old_plan']),p.OLD_PLAN_SHA)
        reused,missing=p.split_queries(p.query_superset(),self.old)
        self.assertEqual((len(reused),len(missing)),(1848,1859));self.assertEqual({r['num_bits_per_symbol'] for r in missing},{6})
        broken=copy.deepcopy(self.old);broken['plans'].pop()
        with self.assertRaisesRegex(ValueError,'full1848'):p.split_queries(p.query_superset(),broken)
    def test_m1_k28_raw_crc_exact_and_no_forward(self):
        layout=p.encoder_layout(Encoder,28,84,6,self.ident)
        self.assertEqual(layout['k'],12+16);self.assertEqual(layout['repetition_bits'],0)
        self.assertEqual(layout['scrambling'],self.old['backend_identity']['scrambling'])
        with self.assertRaisesRegex(ValueError,'dimensions'):p.encoder_layout(Encoder,29,84,6,self.ident)
    def test_backend_repetition_bad_permutation_and_unexpected_errors(self):
        class Rep(Encoder):
            def __init__(self,*a,**k):super().__init__(*a,**k);self.n_cb_comp=self.n-1
        class Perm(Encoder):
            def __init__(self,*a,**k):super().__init__(*a,**k);self.out_int=Tensor(np.zeros(self.n,dtype=np.int64))
        with self.assertRaises(p.UnsupportedConfiguration):p.encoder_layout(Rep,28,84,6,self.ident)
        with self.assertRaisesRegex(ValueError,'permutation'):p.encoder_layout(Perm,28,84,6,self.ident)
        def runtime(*a,**k):raise RuntimeError('software bug')
        with self.assertRaisesRegex(RuntimeError,'software bug'):p.encoder_layout(runtime,28,84,6,self.ident)
    def test_nonmonotone_finite_max_does_not_become_full_integer_grid(self):
        ident=self.ident
        class Backend:
            identity=ident;qualified=False
            def plan(s,k,n,num_bits_per_symbol):
                if (k-16)//12 in (52,53,55):raise self.original.UnsupportedConfiguration('synthetic hole')
                return p.encoder_layout(Encoder,k,n,6,ident)
        planner=self.core.MainPlanner(self.original.Planner(Backend()))
        mcs=p.MCS('64QAM','1/2')
        self.assertEqual(planner.max_K(4,mcs,'nominal'),24) # T54 below canonical T55 hole
        full=p.enumerate64(self.core,self.original,self.fakeplans(),self.ident)
        g=next(r for r in full['generation'] if r['m']==8 and r['nominal_rate']=='5/6' and r['allocation_mode']=='nominal')
        self.assertEqual(g['max_legal_K'],142);self.assertEqual(g['K_candidates'],[0,42,84,126,142])
        self.assertEqual({r['m'] for r in full['generation']},set(range(1,10)))
        self.assertTrue(any(r['m']==1 and r['K']==0 for r in full['profiles']))
    def test_append_exact_legacy_IDs_aliases_and_full_paid_dictionary(self):
        oldp=p.read(self.paths['old_profiles']);olda=p.read(self.paths['old_aliases'])
        enum=p.enumerate64(self.core,self.original,self.fakeplans(),self.ident)
        result=p.append_catalogue(oldp,olda,enum['profiles'])
        self.assertEqual(result['profiles'][:270],oldp);self.assertEqual(result['aliases'][:329],olda)
        self.assertEqual([r['profile_id'] for r in result['profiles']],list(range(len(result['profiles']))))
        self.assertTrue(all(r['profile_id']>=270 for r in result['aliases'][329:]));self.assertFalse(result['original_RX_results_automatically_reusable'])
        broken=copy.deepcopy(enum['profiles'][0]);broken['wire_key']=oldp[0]['wire_key']
        with self.assertRaises(ValueError):p.append_catalogue(oldp,olda,[broken])
        too_many=[dict(enum['profiles'][0],wire_key=f'wire{i}',candidate_id=f'id{i}') for i in range(3827)]
        with self.assertRaisesRegex(ValueError,'12bit'):p.append_catalogue(oldp,olda,too_many)
    def test_missing_duplicate_changed_layout_and_fake_disposition_rejected(self):
        rows=self.fakeplans();row=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):p.index_plans([row,row])
        row['layout']['k_filler']+=1
        with self.assertRaisesRegex(ValueError,'hash'):p.index_plans([row])
        with self.assertRaisesRegex(ValueError,'Missing actual'):p.enumerate64(self.core,self.original,rows[1:],self.ident)
        row=copy.deepcopy(rows[0]);row['disposition']='SOFTWARE_FAILURE'
        with self.assertRaisesRegex(ValueError,'silent'):p.index_plans([row])
    def test_response_identity_and_exact_missing_set(self):
        _,missing=p.split_queries(p.query_superset(),self.old)
        nr=dict(status='MAIN_RAW64_MISSING_RESOURCE_PLANS_COMPLETE',plans=self.fakeplans(),
            backend64_identity=self.ident,old_backend_identity=self.old['backend_identity'],encoder_forward_calls=0,decoder_calls=0,GPU_used=False)
        p.validate_response(nr,self.old,missing,p.sha(p.__file__))
        nr['plans'].pop()
        with self.assertRaisesRegex(ValueError,'exact missing'):p.validate_response(nr,self.old,missing,p.sha(p.__file__))
        nr['old_backend_identity']={}
        with self.assertRaisesRegex(ValueError,'identity'):p.validate_response(nr,self.old,missing,p.sha(p.__file__))
    def test_context_real_all_source_and_input_pins_no_science(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=Path(d)/'config.json';p.save(cfg,self.config());p.context(cfg)
            result=p.run('queries',cfg,Path(d)/'out')
            self.assertEqual(result['reused_count'],1848);self.assertEqual(len(result['missing_queries']),1859)
            self.assertFalse(result['ledger_created']);self.assertFalse(result['qualified_new_PHY'])
            with self.assertRaisesRegex(ValueError,'no retry'):p.run('queries',cfg,Path(d)/'out')
    def test_context_rejects_altered_frozen_source_even_if_caller_rebinds(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.config();fake=Path(d)/'action.py';fake.write_text('changed')
            c['action_core']=str(fake);c['source_bindings'][str(fake)]=p.sha(fake);cfg=Path(d)/'cfg.json';p.save(cfg,c)
            with self.assertRaisesRegex(ValueError,'Frozen original'):p.context(cfg)
    def test_assemble_real_old_metadata_and_synthetic_new_is_not_qualification(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.config();nr=dict(status='MAIN_RAW64_MISSING_RESOURCE_PLANS_COMPLETE',plans=self.fakeplans(),
                backend64_identity=self.ident,old_backend_identity=self.old['backend_identity'],encoder_forward_calls=0,decoder_calls=0,GPU_used=False)
            new=Path(d)/'synthetic.json';p.save(new,nr);c['new_plans']=str(new);c['input_bindings'][str(new)]=p.sha(new)
            cfg=Path(d)/'cfg.json';p.save(cfg,c);r=p.run('assemble',cfg,Path(d)/'out')
            self.assertEqual(r['legacy_profile_count'],270);self.assertFalse(r['quality_ranking'])
            self.assertFalse(r['qualified_new_PHY']);self.assertFalse(r['automatic_successor'])
    def test_plan_scope_requires_hidden_CUDA_before_backend_import(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=Path(d)/'cfg.json';p.save(cfg,self.config())
            with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                with self.assertRaisesRegex(ValueError,'CPU only'):p.run('plan',cfg,Path(d)/'out')
            self.assertEqual(p.read(Path(d)/'out/failure.json')['status'],'FAILED_PRESERVE_NO_RETRY')

if __name__=='__main__':unittest.main()
