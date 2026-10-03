"""CPU fixtures for publication-only audit repair; no scientific output produced."""
import ast
import copy
import sys
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
FROZEN=next(p for p in (HERE.parent/'runtime',HERE.parent/'phase2_recovery_r1',HERE.parent)
            if (p/'phase2_completed.py').is_file())
sys.path.insert(0,str(FROZEN))
import phase2_final_audit as repair
import historical_phase2_alias as alias


def proof():
    grid=dict(psnr_db=27.5,lpips_alex=.17,dino_cosine=.85)
    phase2=dict(grid);phase2['psnr_db']+=1.52587890625e-5
    measured=dict(grid);measured['lpips_alex']+=1e-7
    return dict(measured_native_grid_metrics=measured,
        measured_grid_minus_original_grid={k:measured[k]-grid[k] for k in grid},
        measured_grid_minus_original_phase2={k:measured[k]-phase2[k] for k in grid},
        historical_alias=dict(grid_original_metrics=grid,phase2_original_metrics=phase2,
            historical_grid_minus_phase2={k:grid[k]-phase2[k] for k in grid}))


class PublicationAuditTests(unittest.TestCase):
    def check(self,p):repair.verify_metric_closure(p,alias.METRICS,alias.LIMITS)

    def test_actual_legacy_metric_inventory_and_extra_tolerance(self):
        self.assertEqual(alias.METRICS,('psnr_db','lpips_alex','dino_cosine'))
        self.assertIn('dino_mismatched',alias.LIMITS)
        self.check(proof())

    def test_historical_difference_not_applied_to_new_metric(self):
        p=proof();original=copy.deepcopy(p);self.check(p)
        self.assertEqual(p,original)
        self.assertGreater(abs(p['measured_grid_minus_original_phase2']['psnr_db']),alias.LIMITS['psnr_db'])
        self.assertEqual(p['measured_grid_minus_original_grid']['psnr_db'],0)

    def test_strict_limits_not_relaxed(self):
        p=proof();p['measured_native_grid_metrics']['psnr_db']+=2e-5
        p['measured_grid_minus_original_grid']['psnr_db']+=2e-5
        p['measured_grid_minus_original_phase2']['psnr_db']+=2e-5
        with self.assertRaisesRegex(RuntimeError,'strict tolerance'):self.check(p)

    def test_exact_three_keys_required_in_each_recorded_dictionary(self):
        p=proof()
        for prefix,values in [((),p),(('historical_alias',),p['historical_alias'])]:
            for key in values:
                if key=='historical_alias':continue
                for change in ('missing','extra'):
                    bad=copy.deepcopy(p);target=bad[prefix[0]][key] if prefix else bad[key]
                    if change=='missing':target.pop('psnr_db')
                    else:target['dino_mismatched']=0
                    with self.subTest(prefix=prefix,key=key,change=change),self.assertRaisesRegex(RuntimeError,'keys differ'):self.check(bad)

    def test_all_metrics_must_be_finite(self):
        for badvalue in (float('nan'),float('inf')):
            p=proof();p['measured_native_grid_metrics']['psnr_db']=badvalue
            with self.assertRaisesRegex(RuntimeError,'Nonfinite'):self.check(p)

    def test_forged_native_or_historical_delta_fails_closure(self):
        for parent,key in [('', 'measured_native_grid_metrics'),('', 'measured_grid_minus_original_phase2'),
                           ('historical_alias','historical_grid_minus_phase2'),('historical_alias','grid_original_metrics')]:
            p=proof();container=p[parent] if parent else p;container[key]['dino_cosine']+=1e-7
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'do not close'):self.check(p)

    def test_new_independent_mismatch_is_neither_required_nor_changed(self):
        p=proof();p['new_dino_mismatched']=.31;self.check(p)
        self.assertEqual(p['new_dino_mismatched'],.31)

    def test_final_gate_preserves_all_original_validation_except_fixed_iteration(self):
        original=(FROZEN/'phase2_completed.py').read_text(encoding='utf-8')
        updated=(HERE/'phase2_final_audit.py').read_text(encoding='utf-8')
        def function(text):
            node=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='verify_complete_phase2')
            return ast.get_source_segment(text,node)
        expected=function(original).replace('from historical_phase2_alias import create_adapter, LIMITS',
            'from historical_phase2_alias import create_adapter, LIMITS, METRICS')
        before="                for metric,limit in LIMITS.items():\n                    require(abs(float(parity['measured_grid_minus_original_grid'][metric]))<=limit,\n                        'New pure grid metric exceeded its unchanged strict tolerance')"
        self.assertEqual(expected.count(before),1)
        expected=expected.replace(before,'                verify_metric_closure(parity, METRICS, LIMITS)')
        self.assertEqual(ast.dump(ast.parse(function(updated))),ast.dump(ast.parse(expected)))


if __name__=='__main__':unittest.main()
