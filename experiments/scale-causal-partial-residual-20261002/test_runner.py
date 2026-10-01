"""CPU-only runner boundary tests with explicit fake data/model outputs.

These exercise bookkeeping and policy control flow, not image quality. Native
PHY and real-weight tests are separate test_partial.py/qualification.py gates.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
import partial_phy as phy


class FakeLatent:
    def __init__(self,value=0):self.array=np.full((1,32,16,16),value,np.float32)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.array


class FakeTokens:
    def __init__(self,seed=1):
        rng=np.random.default_rng(seed)
        self.scales=[rng.integers(0,4096,p*p,dtype=np.int64) for p in phy.SIZES]
    def numpy(self):return np.concatenate(self.scales)


def event(action,partial_used=False):
    e={name:False for name in ['header_ok','header_crc_ok','header_fields_legal','prefix_crc_ok',
        'partial_fields_legal','partial_usable','body_crc_ok','raw_candidate_used_after_crc_failure','source_complete']}
    e.update(header_ok=True,header_crc_ok=True,header_fields_legal=True,action=action,
        decoded_m=action.m,decoded_q=action.q,decoded_order=action.order,
        prefix_crc_ok=True,partial_crc_ok=True if action.q else None,
        partial_fields_legal=True,partial_usable=partial_used,body_crc_ok=True,
        trusted_prefix_scales=action.m,hard_candidate_prefix_scales=action.m,
        source_error='',partial_discard_reason='',prefix=FakeTokens().scales[:action.m])
    return e


def load_runner(fake_common):
    fake_rx=types.ModuleType('partial_receiver')
    fake_rx.prefix_logits=lambda *a,**kw:np.zeros((25,4096),np.float32)
    fake_rx.order_positions=lambda prefix,order,q,logits=None,truth_next=None:np.arange(q)
    fake_rx.received_cache_key=lambda e:str(e.get('cache_key','same_received_event'))
    fake_rx.complete_partial=lambda *a,**kw:None
    study=types.ModuleType('var_comm.study');study.seeded_noise=lambda name,seed,shape:np.zeros(shape)
    try:import torch
    except ModuleNotFoundError:torch=types.ModuleType('torch')
    spec=importlib.util.spec_from_file_location('_m1_runner_cpu_boundary_test',HERE/'m1_runner.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,common=fake_common,partial_receiver=fake_rx,torch=torch,
                    **{'var_comm.study':study}):
        spec.loader.exec_module(module)
    return module


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.c=types.ModuleType('common');self.c.RUN='CPU_SYNTHETIC';self.c.OUT=root/'out';self.c.RESULT=root/'results'
        self.c.SNRS=[1,4,7,13,19];self.c.CAL_SEEDS=[4101,4102,4103];self.c.DEV_SEEDS=[2001,2002,2003]
        self.c.split_tokens=lambda t:list(np.split(np.asarray(t),np.cumsum([p*p for p in phy.SIZES])[:-1]))
        self.c.assets=types.SimpleNamespace(mismatch_permutation=lambda:list(range(99,-1,-1)))
        self.c.check=lambda:None;self.c.assert_frozen=lambda loaded:None;self.c.status=lambda *a,**kw:None
        self.c.quality=lambda loaded,record,latent,clean,reference,mis,images:(dict(psnr_db=30.,lpips_alex=.2,dino_cosine=.8,
            dino_mismatched=.3,latent_valid=latent is not None,decoder_applied=latent is not None,latent_sq_err_final=0.),
            np.zeros((3,256,256),np.float32) if images else None)
        self.runner=load_runner(self.c)
        self.loaded=dict(vae='fake',var='fake',device='cpu')
    def tearDown(self):self.temp.cleanup()

    def test_rank_reference_psnr_cap_and_K0_admissible(self):
        group=[dict(action_id='whole4',m=4,q=0,failure_fraction=0.,psnr_db=30.,lpips_alex=.3),
            dict(action_id='whole5',m=5,q=0,failure_fraction=0.,psnr_db=30.,lpips_alex=.2),
            dict(action_id='partial',m=4,q=6,failure_fraction=.1,psnr_db=29.8,lpips_alex=.1),
            dict(action_id='bad_psnr',m=4,q=12,failure_fraction=.05,psnr_db=29.74,lpips_alex=.01)]
        ranked,reference=self.runner.rank(group)
        self.assertEqual(reference['action_id'],'whole5')
        self.assertEqual(ranked[0]['action_id'],'partial')
        self.assertNotIn('bad_psnr',[x['action_id'] for x in ranked])
        ranked,_=self.runner.rank([x for x in group if x['q']==0])
        self.assertEqual(ranked[0]['q'],0)

    def test_q0_scheme_alias_and_raster_random_no_sender_VAR(self):
        canonical=phy.Action(1024,'16QAM',8,0,'entropy')
        self.assertEqual(canonical.order,'whole')
        self.assertEqual(self.runner.action_from(self.runner.action_dict(canonical)),canonical)
        scales=FakeTokens().scales
        with patch.object(self.runner.rx,'prefix_logits',side_effect=AssertionError('Unused VAR should not run')):
            self.assertIsNone(self.runner.tx_positions({},scales,canonical,{}))
            for order in ['raster','random']:
                np.testing.assert_array_equal(self.runner.tx_positions({},scales,phy.Action(512,'QPSK',4,6,order),{}),np.arange(6))

    def test_entropy_oracle_sender_logits_memo_is_per_prefix(self):
        scales=FakeTokens().scales;memo={};loaded=dict(vae='fake',var='fake',device='cpu')
        with patch.object(self.runner.rx,'prefix_logits',return_value=np.zeros((25,4096),np.float32)) as logits:
            for order in ['entropy','oracle']:
                self.runner.tx_positions(loaded,scales,phy.Action(512,'QPSK',4,6,order),memo)
        self.assertEqual(logits.call_count,1)

    def test_header_q0_wrong_accept_and_different_m_diagnostics_are_safe(self):
        tokens=FakeTokens();d=dict(records=[dict(image_id='s',preprocessing_id='p')],T=[tokens],F=[None],reference=[None])
        a=phy.Action(1024,'16QAM',4,0,'whole')
        got=event(phy.Action(1024,'16QAM',7,6,'entropy'),partial_used=True)
        result=dict(fhat=FakeLatent(),diagnostics=dict(partial_used=True,known_tokens_fixed=True,
            known_positions=[90,91,92,93,94,95],known_values=[1]*6))
        with patch.object(self.runner.phy,'transmit',return_value=(np.ones((1024,2)),{'E':2048})),\
             patch.object(self.runner.phy,'receive',return_value=got),\
             patch.object(self.runner.rx,'complete_partial',return_value=result),\
             patch.object(self.runner,'tx_positions',side_effect=lambda loaded,scales,a,memo:None if a.q==0 else (_ for _ in ()).throw(AssertionError('Wrong header must not index TX partial'))):
            row,_=self.runner.score_one(self.loaded,d,0,a,7,2001,dict(waves={},logits={}),{}, {})
        self.assertFalse(row['header_exact']);self.assertTrue(row['partial_used'])
        self.assertEqual(row['order_agreement'],'');self.assertEqual(row['received_partial_exact'],'')

    def test_protocol_body_failure_and_quality_cache_within_same_source(self):
        a=phy.Action(512,'QPSK',4,0,'whole');tokens=FakeTokens()
        d=dict(records=[dict(image_id='s',preprocessing_id='p')],T=[tokens],F=[None],reference=[None])
        got=event(a);got['body_crc_ok']=False;got['prefix_crc_ok']=False
        result=dict(fhat=FakeLatent(),diagnostics=dict(partial_used=False,known_tokens_fixed=True))
        tm=dict(waves={},logits={});rm={};mm={}
        with patch.object(self.runner.phy,'transmit',return_value=(np.ones((512,2)),{'E':1024})),\
             patch.object(self.runner.phy,'receive',return_value=got),\
             patch.object(self.runner.rx,'complete_partial',return_value=result) as complete,\
             patch.object(self.c,'quality',wraps=self.c.quality) as quality:
            first,_=self.runner.score_one(self.loaded,d,0,a,7,2001,tm,rm,mm)
            second,_=self.runner.score_one(self.loaded,d,0,a,7,2002,tm,rm,mm)
        self.assertEqual(complete.call_count,1);self.assertEqual(quality.call_count,1)
        self.assertEqual(self.runner.summarize([first,second])[0]['failure_fraction'],1.)
        self.assertTrue(first['latent_valid'])

    def test_screen_forces_whole_and_resets_all_caches_each_source(self):
        artifacts={};tables={};memo_instances={}
        records=[dict(image_id=f's{i}',preprocessing_id=f'p{i}') for i in range(1000)]
        self.c.data=lambda role,loaded:dict(records=records)
        self.c.registration=lambda loaded,data,stage,inputs:'CPU_SYNTHETIC_REGISTRATION'
        self.c.csv_rows=lambda path,rows:tables.__setitem__(Path(path).name,rows)
        self.c.seal=lambda path,obj:artifacts.__setitem__(Path(path).name,obj)
        self.c.write=self.c.seal;self.c.sha=lambda path:'CPU_SYNTHETIC_SHA'
        grid=[phy.Action(N,f,m,q,order) for N in (512,1024) for f in phy.PHY_FAMILIES
              for m,q,order in [(4,0,'whole'),(5,0,'whole'),(4,6,'entropy'),(4,12,'entropy')]]
        def fake_score(loaded,d,i,a,snr,seed,tm,rm,mm):
            if i not in memo_instances:
                self.assertEqual(tm,dict(waves={},logits={}))
                self.assertEqual(rm,{});self.assertEqual(mm,{})
                memo_instances[i]=(tm,rm,mm)
                tm['waves']['source_marker']=i;rm['source_marker']=i;mm['source_marker']=i
            else:
                self.assertIs(memo_instances[i][0],tm);self.assertIs(memo_instances[i][1],rm)
                self.assertIs(memo_instances[i][2],mm)
            return dict(N=a.N,phy_family=a.phy,snr_db=snr,action_id=self.runner.action_id(a),m=a.m,q=a.q,order=a.order,
                psnr_db=30.,lpips_alex=.05 if a.q==12 else .1 if a.q else .2 if a.m==5 else .3,
                dino_cosine=.8,E=2*a.N,header_ok=True,body_crc_ok=True),None
        with patch.object(self.runner,'legal_grid',return_value=grid),patch.object(self.runner,'score_one',side_effect=fake_score):
            self.runner.calibration({},True)
        self.assertEqual(len(memo_instances),200)
        self.assertEqual(len({id(x[0]) for x in memo_instances.values()}),200)
        short=artifacts['m1_shortlist.json'];self.assertFalse(short['development_read'])
        self.assertEqual(short['noise_seeds'],[4101]);self.assertEqual(short['sources'],200)
        self.assertEqual(len(short['cells']),100)
        for cell in short['cells']:
            selected=[next(a for a in grid if self.runner.action_id(a)==aid) for aid in cell['action_ids']]
            self.assertTrue(any(a.q==0 for a in selected))
            if cell['method']=='entropy':
                self.assertEqual(len(selected),3)
                self.assertTrue(any(a.q==0 and a.m==5 for a in selected))
        self.assertEqual(artifacts['m1_screen_complete.json']['rows'],200*len(grid)*5)


if __name__=='__main__':unittest.main()
