"""Synthetic population/paid-capacity/noise/counter tests; no scientific inputs."""
from pathlib import Path
import sys, unittest, copy, ast
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan import *
from c_predclass_frame import choose
import numpy as np


def profiles():
    out = []
    for q, rate in ((2,'1/2'),(2,'2/3'),(4,'2/3'),(4,'3/4'),(6,'5/6')):
        for prefix in range(1,257):
            out.append(dict(profile_id=len(out),q=q,nominal_rate=rate,family=FAMILY,
                prefix_count=prefix,m=prefix,K=0,k=1039))
        out.append(dict(profile_id=len(out),q=q,nominal_rate=rate,family=NULL,prefix_count=0,m=0,K=0,k=1039))
    return out


def source(index):
    return dict(source_id='synthetic_%d'%index,source_index=index,predicted_class=999,
        streams=dict(sha256='%064x'%index),lengths=[dict(prefix=i,bits=i*5)for i in range(1,257)])


class Plan(unittest.TestCase):
    def setUp(self):
        self.p=profiles();self.s=[source(i)for i in (9,2,499)]
    def test_full_six_choices_three_original_noises(self):
        cs=build_choices(self.s,self.p,3);es=events(cs)
        self.assertEqual(len(cs),18);self.assertEqual(len(es),54)
        self.assertEqual({v['noise_seed']for v in es},{4101,4102,4103})
        self.assertEqual({v['source_index']for v in es},{2,9,499})
        self.assertEqual(sum(v['snr_db']==4 for v in es),18)
    def test_class_paid_before_fitting(self):
        cs=build_choices(self.s,self.p,3)
        self.assertTrue(all(c['prefix']==200 and c['arithmetic_capacity_bits']==1000 and c['container_bits']==1010 for c in cs))
    def test_nonmonotonic_lengths_longest_actual_fit(self):
        self.s[0]['lengths'][-1]['bits']=998
        cs=build_choices(self.s,self.p,3)
        self.assertTrue(all(c['prefix']==256 for c in cs if c['source_index']==9))
    def test_paper32_capacity_diagnostic_not_quality_choice(self):
        cs=build_choices(self.s,self.p,3)
        self.assertTrue(all(c['paper32_prefix']==192 and c['prefix']==200 for c in cs))
    def test_null_keeps_complete_denominator(self):
        for s in self.s:
            for x in s['lengths']:x['bits']=2000
        cs=build_choices(self.s,self.p,3)
        self.assertEqual(len(events(cs)),54)
        self.assertTrue(all(c['explicit_null']and c['prefix']==0 and c['paid_class_bits']==0 and c['container_bits']==2 for c in cs))
    def test_noise_and_candidate_identity_unique_waveform_reuse(self):
        cs=build_choices(self.s,self.p,3);es=events(cs)
        self.assertEqual(len({e['key']for e in es}),54)
        self.assertEqual(len({c['waveform_key']for c in cs}),15)
        for c in cs:
            self.assertEqual(c['counter'],COUNTER_BASE+4096*c['source_index']+c['profile_id'])
    def test_source_order_deterministic_original_indices(self):
        self.assertEqual(build_choices(self.s,self.p,3),build_choices(list(reversed(self.s)),self.p,3))
    def test_duplicate_or_missing_source_rejected(self):
        with self.assertRaises(AssertionError):build_choices([self.s[0]]*3,self.p,3)
        with self.assertRaises(AssertionError):build_choices(self.s,self.p,2)
    def test_missing_prefix_or_invalid_class_rejected(self):
        s=copy.deepcopy(self.s);s[0]['lengths'].pop()
        with self.assertRaises(AssertionError):build_choices(s,self.p,3)
        s=copy.deepcopy(self.s);s[0]['predicted_class']=1000
        with self.assertRaises(ValueError):build_choices(s,self.p,3)
    def test_profile_range_and_catalogue_completeness(self):
        p=copy.deepcopy(self.p);p[0]['profile_id']=4096
        with self.assertRaises(AssertionError):build_choices(self.s,p,3)
        with self.assertRaises(AssertionError):build_choices(self.s,self.p[:-1],3)
    def test_plan_matches_qualified_real_choose_function(self):
        # Exercise the already-qualified packer with synthetic bit vectors only.
        for s in self.s:
            streams={'p%03d'%v['prefix']:np.zeros(v['bits'],np.uint8) for v in s['lengths']}
            for c in build_choices([s],self.p,1):
                p,bits,checks=choose(streams,self.p,c['q'],c['rate'],s['predicted_class'])
                self.assertEqual(p['profile_id'],c['profile_id'])
                self.assertEqual(len(bits),c['container_bits'])
                self.assertEqual(len(checks),c['length_checks'])
    def test_length_field_ceiling_and_paid_class(self):
        p=copy.deepcopy(self.p)
        for v in p:v['k']=20000
        s=copy.deepcopy(self.s)
        for item in s:
            for row in item['lengths']:row['bits']=8181 if row['prefix']==7 else 8182
        cs=build_choices(s,p,3)
        self.assertTrue(all(c['prefix']==7 and c['container_bits']==8191 for c in cs))
    def test_no_model_or_phy_runtime_import(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('ep_phy',sys.modules)
    def test_receiver_call_has_only_observation_public_state_and_ledger(self):
        tree=ast.parse(Path(__file__).with_name('worker.py').read_text())
        calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='receive']
        self.assertEqual(len(calls),1)
        names={n.id for n in ast.walk(calls[0])if isinstance(n,ast.Name)}
        self.assertTrue(names.issubset({'rt','observed','j','ledger'}))
        self.assertNotIn('s',names);self.assertNotIn('bits',names);self.assertNotIn('c',names)


if __name__=='__main__':unittest.main(verbosity=2)
