"""Scope, original-index, cache and bootstrap tests for the small fixed16 tier."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import hifi16_common as c

class Fixed16(unittest.TestCase):
    def test_exact_grid_and_original_ids(self):
        specs=[v for i in c.SOURCE_INDICES for v in c.frame_specs(i)]
        self.assertEqual(len(specs),64)
        self.assertEqual({s['source_index'] for s in specs},set(c.SOURCE_INDICES))
        self.assertEqual({s['noise_seed'] for s in specs},{2001});self.assertEqual({s['snr_db'] for s in specs},{7,13})
        ids=[v for i in c.SOURCE_INDICES for v in c.expected_ids(i)]
        self.assertEqual(len(ids),128);self.assertEqual(len(set(ids)),128)
        for s in specs:
            for m in c.METHODS:self.assertEqual(c.row_id(s,m),c.paired.row_id(s,m))
    def test_reject_nonselected_index_and_renumbering(self):
        for index in (1,2,3,96):
            with self.assertRaises(RuntimeError):c.frame_specs(index)
    def baseline(self):
        return [dict(source_id=f'original{i}',source_index=i,resnet50_source_prediction=5,true_class_index=5,
            resnet50_source_top1_label=1,preprocessing_id='p',reference_sha256='f'*64) for i in c.SOURCE_INDICES]
    def test_source_bootstrap_keeps_original_indices(self):
        sources,identities=c.selected_sources(self.baseline())
        self.assertEqual([identities[s]['index'] for s in sources],sorted(c.SOURCE_INDICES))
        self.assertEqual(identities['original95']['index'],95)
    def test_reject_16_fake_renumbered_sources(self):
        values=self.baseline()
        for i,row in enumerate(values):row['source_index']=i
        with self.assertRaises(RuntimeError):c.selected_sources(values)
    def test_reject_duplicate_or_missing_source(self):
        values=self.baseline();values[-1]=copy.deepcopy(values[0])
        with self.assertRaises(RuntimeError):c.selected_sources(values)
        with self.assertRaises(RuntimeError):c.selected_sources(self.baseline()[:-1])
    def test_original_bootstrap_draws16_not100(self):
        original=c.HERE.parents[3]/'experiments/unified-metrics-20261002/analysis.py'
        reference=c.HERE/'reference_analysis.py.txt'
        source=(original if original.exists() else reference).read_text(encoding='utf-8')
        tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Bootstrap')
        namespace=dict(np=np,REPLICATES=10000,SEED=20261002)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'original_registered_bootstrap','exec'),namespace)
        bootstrap=namespace['Bootstrap']();result=bootstrap.interval(np.arange(16,dtype=float))
        self.assertEqual(result['n_sources'],16);self.assertEqual(result['mean'],7.5)
        self.assertEqual(bootstrap.indices[16].shape,(10000,16));self.assertNotIn(100,bootstrap.indices)
        self.assertEqual(bootstrap.interval(np.zeros(16))['ci_high'],0.)
    def test_eight_row_source_archive_preserves95(self):
        with tempfile.TemporaryDirectory() as folder:
            rgb=np.zeros((3,256,256),dtype=np.float32);rows=[]
            for spec in c.frame_specs(95):
                for method in c.METHODS:
                    rows.append(dict(**spec,method=method,replay_row_id=c.row_id(spec,method),image_sha256=c.rgb_sha(rgb),reference_sha256=c.rgb_sha(rgb)))
            path=Path(folder)/'source.npz';digest=c.atomic_npz(path,images=rgb[None],source_rgb=rgb,row_ids=np.asarray(c.expected_ids(95)),image_slots=np.zeros(8,dtype=np.int64))
            value=dict(binding='test',source_index=95,rows=rows,frame_bindings={},float_reconstructions=dict(path=str(path),sha256=digest,image_slots=[0]*8))
            value['payload_sha256']=c.identity(value)
            self.assertEqual(c.validate_source(value,'test',95),value)
            with self.assertRaises(RuntimeError):c.validate_source(value,'test',15)
    def test_sampler_guard_rejects_short_diagnostic(self):
        with self.assertRaises(RuntimeError):c.validate_full_sampler(dict(header_accepted=True,NFE=2,t_start=255,diagnostic_probe=True),True)
    def test_full_sampler_and_gray_fallback_remain_admitted(self):
        c.validate_full_sampler(dict(header_accepted=True,diagnostic_probe=False,complete_author_schedule=True,
            model_parameters_unchanged=True,parameter_gradients_accumulated=False,schedule_mode='actual_data_cbr',NFE=255,t_start=255),True)
        c.validate_full_sampler(dict(header_accepted=False,NFE=0,fallback='fixed_gray_0.5'),False)
    def test_science_files_parse_and_no_full_queue_resume(self):
        for name in c.FILES:ast.parse((c.HERE/name).read_text(encoding='utf-8'))
        text=(c.HERE/'hifi16_eval.py').read_text()
        self.assertIn('external_eval.reconstruct_frame(codec,receiver,source,spec,directory,binding,selected)',text)
        self.assertFalse(any(isinstance(n,ast.keyword) and n.arg=='step_limit' for n in ast.walk(ast.parse(text))))
        scorer=(c.HERE/'hifi16_score.py').read_text()
        self.assertIn('mismatch_source_id=originals[mismatch[index]]',scorer)
        self.assertIn('for i in SOURCE_INDICES for rid in expected_ids(i)',scorer)
        self.assertNotIn('metric_groups=12',scorer)
    def test_original95_mismatch_can_be_outside_selected16(self):
        source=(c.HERE/'hifi16_score.py').read_text()
        tree=ast.parse(source)
        expression=next(n.value for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='mismatch_source_id')
        originals=[dict(image_id=f'original{i}') for i in range(100)]
        records=[originals[i] for i in c.SOURCE_INDICES];mismatch=list(range(100));mismatch[95]=83
        result=eval(compile(ast.Expression(expression),'actual_scorer_mismatch_reference','eval'),
            dict(originals=originals,records=records,mismatch=mismatch,index=95))
        self.assertEqual(result,'original83')

if __name__=='__main__':unittest.main(verbosity=2)
