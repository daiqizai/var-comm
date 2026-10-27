"""Synthetic formal population and fixed-policy regression, never real science."""
from pathlib import Path
import sys,unittest,copy,ast
import numpy as np
from plan import *
from c_predclass_frame import choose
import calibration_plan_reference as calibration

POLICY={'4':'p256_q2_r2-3','10':'p256_q4_r2-3','19':'p256_q6_r5-6'}

def profiles():
    rows=[]
    for q,rate in ((2,'1/2'),(2,'2/3'),(4,'2/3'),(4,'3/4'),(6,'5/6')):
        for prefix in range(1,257):
            rows.append(dict(profile_id=len(rows),q=q,nominal_rate=rate,family=FAMILY,
                prefix_count=prefix,m=prefix,K=0,k=1039))
        rows.append(dict(profile_id=len(rows),q=q,nominal_rate=rate,family=NULL,prefix_count=0,m=0,K=0,k=1039))
    return rows

def source(i):
    return dict(source_id='synthetic_%d'%i,source_index=i,predicted_class=999,
        streams={'sha256':'%064x'%i},lengths=[dict(prefix=j,bits=j*5)for j in range(1,257)])

class FormalPlan(unittest.TestCase):
    def setUp(self):self.p=profiles();self.s=[source(i)for i in(9,2,23)]
    def choices(self,s=None,p=None,population='common500',policy=None):
        sources=self.s if s is None else s
        return build_choices(sources,self.p if p is None else p,population,POLICY if policy is None else policy,len(sources))
    def test_common500_full_denominators(self):
        cs=build_choices([source(i)for i in range(500)],self.p,'common500',POLICY)
        es=events(cs,'common500');self.assertEqual(len(es),4500)
        self.assertEqual({r['noise_seed']for r in es},{6201,6202,6203})
        for snr in (4,10,19):self.assertEqual(sum(r['snr_db']==snr for r in es),1500)
    def test_kodak24_full_denominators(self):
        es=events(build_choices([source(i)for i in range(24)],self.p,'kodak24',POLICY),'kodak24')
        self.assertEqual(len(es),216);self.assertEqual({r['noise_seed']for r in es},{2001,2002,2003})
        self.assertEqual({r['source_index']for r in es},set(range(24)))
    def test_calibration_fitting_parity(self):
        before=calibration.build_choices(self.s,self.p,3)
        before={(r['source_id'],r['snr_db']):r for r in before if r['candidate_id']==POLICY[str(r['snr_db'])]}
        for r in self.choices():
            old=before[(r['source_id'],r['snr_db'])]
            for k in set(old)-{'key','counter','waveform_key'}:self.assertEqual(r[k],old[k],k)
    def test_preserved_indices_and_order(self):
        self.assertEqual(self.choices(),self.choices(list(reversed(self.s))))
        self.assertEqual({v['source_index']for v in self.choices()},{2,9,23})
    def test_fixed_policy_rejects_unknown_or_wrong_snr(self):
        for k,value in [('4','p256_q4_r2-3'),('10','p256_q4_r9-10'),('19','changed')]:
            p=dict(POLICY);p[k]=value
            with self.assertRaises(AssertionError):self.choices(policy=p)
        with self.assertRaises(AssertionError):self.choices(policy={'4':POLICY['4']})
    def test_population_separation_and_counter(self):
        a=self.choices();b=self.choices(population='kodak24')
        self.assertFalse({v['key']for v in a}&{v['key']for v in b})
        for pop,cs in [('common500',a),('kodak24',b)]:
            for r in cs:self.assertEqual(r['counter'],POPULATIONS[pop]['counter_base']+4096*r['source_index']+r['profile_id'])
        with self.assertRaises(AssertionError):events(a,'kodak24')
    def test_paid_class_null_and_exact_denominator(self):
        self.assertTrue(all(r['prefix']==200 and r['arithmetic_capacity_bits']==1000 for r in self.choices()))
        ss=copy.deepcopy(self.s)
        for s in ss:
            for row in s['lengths']:row['bits']=2000
        cs=self.choices(ss);self.assertEqual(len(events(cs,'common500')),27)
        self.assertTrue(all(r['explicit_null']and r['container_bits']==2 for r in cs))
    def test_nonmonotonic_longest_and_actual_packer(self):
        ss=copy.deepcopy(self.s);ss[0]['lengths'][-1]['bits']=998
        for s in ss:
            streams={'p%03d'%v['prefix']:np.zeros(v['bits'],np.uint8)for v in s['lengths']}
            for c in self.choices([s]):
                p,bits,checks=choose(streams,self.p,c['q'],c['rate'],s['predicted_class'])
                self.assertEqual(p['profile_id'],c['profile_id']);self.assertEqual(len(bits),c['container_bits'])
                self.assertEqual(len(checks),c['length_checks'])
        self.assertTrue(all(r['prefix']==256 for r in self.choices(ss)if r['source_index']==9))
    def test_invalid_source_or_catalogue_rejected(self):
        for ss in ([self.s[0]]*3,[source(500)]):
            with self.assertRaises(AssertionError):self.choices(ss)
        with self.assertRaises(AssertionError):self.choices([source(24)],population='kodak24')
        with self.assertRaises(AssertionError):self.choices(p=self.p[:-1])
    def test_original_waveform_noise_and_rx_call_math(self):
        old=ast.parse(Path(__file__).with_name('calibration_worker_reference.py').read_text())
        new=ast.parse(Path(__file__).with_name('worker.py').read_text())
        def calls(tree,attr):
            return [n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr==attr]
        for attr in ('transmit','standard_noise'):
            self.assertEqual([ast.dump(n)for n in calls(old,attr)],[ast.dump(n)for n in calls(new,attr)])
        for tree in (old,new):
            rs=calls(tree,'receive');self.assertEqual(len(rs),1)
            self.assertEqual([kw.arg for kw in rs[0].keywords],['phase']);rs[0].keywords=[]
        self.assertEqual(ast.dump(calls(old,'receive')[0]),ast.dump(calls(new,'receive')[0]))
        def observation(tree):
            return [ast.dump(n.value)for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='observed'for t in n.targets)]
        self.assertEqual(observation(old),observation(new))
    def test_no_model_or_phy_imports(self):
        for name in('torch','tensorflow','ep_phy'):self.assertNotIn(name,sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
