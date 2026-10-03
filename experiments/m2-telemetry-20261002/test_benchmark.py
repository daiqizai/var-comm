"""CPU contracts for the telemetry benchmark; no CUDA, weights or images."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import benchmark as b


def result():
    cases=[]
    for index in range(15):
        cases.append(dict(case={'projection':index//3,'seed':4101+index%3},measurement={'wave':'fixed'},
            inferred=dict(tokens=[np.array([index,k]) for k in range(10)],fhat=np.array([index,.25]),
                diagnostics=[{'changes':index,'objective':1.5}],inference='original'),
            rgb=np.array([index,.5],dtype=np.float32),metrics={'psnr_db':20.,'lpips_alex':.2,'dino_cosine':.8}))
    return dict(base=np.array([1.,2.]),base_rgb=np.array([.1,.2]),base_metrics={'psnr_db':21.},cases=cases)


class BenchmarkTests(unittest.TestCase):
    def test_historical_gate_uses_original_parity_and_exact_scope(self):
        candidates=[parent/relative for parent in Path(__file__).resolve().parents
            for relative in ('experiments/unified-metrics-20261002/replay.py',
                             '.research/metrics_20261002/source/replay.py')]
        replay=b.load('_test_original_metric_parity',next(path for path in candidates if path.is_file()))
        output=result();output['base_metrics'].update(lpips_alex=.2,dino_cosine=.8)
        rows=[]
        for i,case in enumerate(output['cases']):
            case['case']=dict(projection='projection'+str(i//3),noise_seed=4101+i%3,lambda_value=.5)
            case['measurement']={'measurement_y_sha256':'frozen observation'}
            for control in ('UNGUIDED','VAR_GUIDED'):
                rows.append(dict(source_index=0,stage='full_calibration_gate_reduced_noisy',
                    projection=case['case']['projection'],noise_seed=case['case']['noise_seed'],control=control,
                    **{'lambda':0. if control=='UNGUIDED' else .5},**case['measurement'],
                    **(output['base_metrics'] if control=='UNGUIDED' else case['metrics'])))
        historical=dict(registration_sha256='registered',rows=rows)
        self.assertEqual(b.compare_history(output,historical,'registered',replay)['rows'],30)
        close=copy.deepcopy(historical);close['rows'][0]['psnr_db']+=1e-6
        self.assertEqual(b.compare_history(output,close,'registered',replay)['status'],'ORIGINAL_GATE_PARITY_PASS')
        changes=[lambda x:x.update(registration_sha256='other'),lambda x:x['rows'].pop(),
            lambda x:x['rows'][0].update(psnr_db=25.),
            lambda x:x['rows'][0].update(measurement_y_sha256='different'),
            lambda x:x['rows'][0].update(source_index=1),
            lambda x:x['rows'][0].update(stage='other'),
            lambda x:x['rows'].__setitem__(1,copy.deepcopy(x['rows'][0]))]
        for change in changes:
            other=copy.deepcopy(historical);change(other)
            with self.assertRaises(RuntimeError):b.compare_history(output,other,'registered',replay)

    def test_exact_comparison_checks_every_scientific_field(self):
        torch=SimpleNamespace(equal=np.array_equal);first=result()
        self.assertTrue(b.compare(torch,np,first,copy.deepcopy(first)))
        mutations=[lambda x:x['base'].__setitem__(0,9),lambda x:x['base_metrics'].update(psnr_db=21.00001),
            lambda x:x['cases'][0]['inferred']['tokens'][3].__setitem__(0,99),
            lambda x:x['cases'][1]['inferred']['fhat'].__setitem__(1,.25000001),
            lambda x:x['cases'][2]['inferred']['diagnostics'][0].update(changes=99),
            lambda x:x['cases'][3]['rgb'].__setitem__(1,.50001),
            lambda x:x['cases'][4]['metrics'].update(dino_cosine=.80000001),
            lambda x:x['cases'][5]['measurement'].update(wave='other'),lambda x:x['cases'].pop()]
        for mutation in mutations:
            changed=copy.deepcopy(first);mutation(changed)
            with self.assertRaises(RuntimeError):b.compare(torch,np,first,changed)

    def test_aggregate_interleaved_decision_and_no_selective_discard(self):
        rows=[dict(repeat=i,implementation=name,seconds=seconds,exact=True)
              for i,name,seconds in [(0,'original',11.),(0,'cached',10.),(1,'cached',10.),(1,'original',11.)]]
        self.assertEqual(b.decision(rows)['status'],'QUALIFIED')
        rows[0]['seconds']=10.
        self.assertEqual(b.decision(rows)['status'],'USE_ORIGINAL')
        for changed in (rows[:-1],rows[::-1],[dict(x,exact=False) for x in rows],
                        [dict(x,seconds=float('nan')) for x in rows]):
            with self.assertRaises(RuntimeError):b.decision(changed)

    def test_safety_counter_preserves_return_exception_and_restores_method(self):
        class Safety:
            def check(self):return 'original guard result'
        safety=Safety();original=safety.check
        common=SimpleNamespace(assets=SimpleNamespace(old=SimpleNamespace(b=SimpleNamespace(SAFETY=safety))))
        with b.safety_counter(common) as stats:
            self.assertEqual(safety.check(),'original guard result');self.assertEqual(safety.check(),'original guard result')
        self.assertEqual(stats['calls'],2);self.assertGreaterEqual(stats['seconds'],0.)
        self.assertEqual(safety.check,original)
        def fail():raise RuntimeError('original STOP/resource error')
        safety.check=fail
        with self.assertRaisesRegex(RuntimeError,'original STOP'):
            with b.safety_counter(common) as stats:safety.check()
        self.assertIs(safety.check,fail);self.assertEqual(stats['calls'],1)

    def test_rng_comparison_covers_cpu_cuda_numpy_and_python(self):
        torch=SimpleNamespace(equal=np.array_equal)
        state=dict(torch=np.array([1]),cuda=[np.array([2])],numpy=('MT',np.array([3]),1,0,0.),python=('state',1))
        self.assertTrue(b.same_rng(state,copy.deepcopy(state),torch,np))
        for key,value in [('torch',np.array([4])),('cuda',[np.array([5])]),('numpy',('MT',np.array([9]),1,0,0.)),('python',('state',2))]:
            changed=copy.deepcopy(state);changed[key]=value
            self.assertFalse(b.same_rng(state,changed,torch,np))


if __name__=='__main__':unittest.main()
