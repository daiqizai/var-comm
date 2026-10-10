"""Receipt/schema/namespace checks only; never loads scientific arrays/models."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import leo_whole_input_map_v1 as m
import leo_whole_gate_v1 as g


class Tests(unittest.TestCase):
    def test_source_population_exact_count_order_namespace_and_duplicates(self):
        rows=[dict(original=g.OLD+f'outputs/fixture/{i}.json',restored=f'/private/{i}',bytes=1,sha256='a'*64) for i in range(547)]
        rows[-1]['bytes']=98236269-546
        r=dict(status='FIXED_EXISTING32_BYTES_RESTORED_VERIFIED_NOT_SCIENTIFIC_QUALIFICATION',new100_read=False,
            model_calls=0,count=547,total_bytes=98236269,mapping=rows)
        self.assertEqual(len(m.source_rows(r)),547)
        for changed in [r|{'new100_read':True},r|{'count':546}]:
            with self.assertRaises(RuntimeError):m.source_rows(changed)
        bad=copy.deepcopy(r);bad['mapping'][1]['original']=bad['mapping'][0]['original']
        with self.assertRaises(RuntimeError):m.source_rows(bad)
        bad=copy.deepcopy(r);bad['mapping'][0]['original']='/different/population'
        with self.assertRaises(RuntimeError):m.source_rows(bad)

    def test_model_namespace_is_derived_not_guessed_and_code_uses_repo(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/'repo';seed=base/'assets/model_seed_v1'
            names=['VAR_COMM/outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_A_v1/checkpoints/update_0038000_1789635001035591298.pt',
                'VAR_COMM/outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v2_repaired_20260921/decoder_gate.json',
                'VAR_COMM/src/var_comm/next_scale_prior.py',
                'external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth',
                'external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/var_d16.pth']
            r=dict(status='MODEL_BYTES_AND_PUBLISHED_SPARSE_SOURCE_VERIFIED_NOT_RUNTIME_QUALIFICATION',gpu_calls=0,
                files=[dict(path=str(seed/n),bytes=1,sha256='a'*64,verified=True) for n in names])
            with mock.patch.object(m,'RT',base),mock.patch.object(g,'inside',side_effect=Path):
                rows=m.model_rows(r,root);self.assertEqual(len(rows),5)
                code=next(x for x in rows if x['original_path'].endswith('next_scale_prior.py'))
                self.assertEqual(code['path'],str(root/'src/var_comm/next_scale_prior.py'))
                self.assertIn(g.WEIGHTS['vae'],[x['original_path'] for x in rows])
                bad=copy.deepcopy(r);bad['files'][0]['path']=str(base/'outside/file')
                with self.assertRaises(RuntimeError):m.model_rows(bad,root)

    def test_mapping_conflicts_are_never_silently_overwritten(self):
        row=dict(original_path='/old/a',path='/new/a',sha256='a'*64,bytes=1)
        self.assertEqual(m.merge_rows([row],[dict(row)]),[row])
        with self.assertRaises(RuntimeError):m.merge_rows([row],[row|{'path':'/different'}])

    def test_candidate_missing_original_dist_is_disclosed_not_relabelled(self):
        r=dict(status='UPSTREAM_CANDIDATE_VERIFIED_PENDING_NUMERICAL_QUALIFICATION',original_dist_sha_available=False,
            original_models_exact_matches=7,upstream_commit=g.UPSTREAM_COMMIT,dist_sha256=g.UPSTREAM_DIST_SHA,
            audit_sha256=g.UPSTREAM_AUDIT_SHA,model_calls=0,root='/candidate',
            files=[dict(path=f'models/{i}.py',sha256='a'*64,original_sha256='a'*64,exact_original_match=True,bytes=1) for i in range(7)])
        with mock.patch.object(g,'inside',side_effect=Path):
            self.assertEqual(len(m.candidate_rows(r)[0]),7)
            with self.assertRaises(RuntimeError):m.candidate_rows(r|{'original_dist_sha_available':True})
            bad=copy.deepcopy(r);bad['files'][0]['path']='models/../escape.py'
            with self.assertRaises(RuntimeError):m.candidate_rows(bad)


if __name__=='__main__':unittest.main(verbosity=2)
