import ast,pathlib,sys,unittest
import numpy as np
import score

class Inputs(unittest.TestCase):
    def test_native_float_retained(self):
        a=np.full((3,256,256),.1234567,dtype=np.float32)
        self.assertIs(score.require_image(a),a);self.assertNotEqual(float(a[0,0,0]),round(float(a[0,0,0])*255)/255)
    def test_uint8_rejected(self):
        with self.assertRaises(AssertionError):score.require_image(np.zeros((3,256,256),dtype=np.uint8))
    def test_nonfinite_rejected(self):
        a=np.zeros((3,256,256),dtype=np.float32);a[0,0,0]=np.nan
        with self.assertRaises(AssertionError):score.require_image(a)
    def test_out_of_range_rejected(self):
        with self.assertRaises(AssertionError):score.require_image(np.full((3,256,256),1.001,dtype=np.float32))
    def test_wrong_geometry_rejected(self):
        with self.assertRaises(AssertionError):score.require_image(np.zeros((3,224,224),dtype=np.float32))
    def test_budget(self):
        self.assertEqual(score.CAPS,{'model_load':2,'reference_dino':100,'image_scores':800,'lpips_pair':800,'lpips_backbone_forward':1600,'reconstruction_dino':800})
    def test_no_old_model_constructor(self):
        tree=ast.parse(pathlib.Path(score.__file__).read_text())
        calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse({'old.FourMetrics','old.convnext_projection','torch.hub.load','load_transformer','load_visual_tokenizer'} & calls)
    def test_no_bootstrap_or_channel_import(self):
        tree=ast.parse(pathlib.Path(score.__file__).read_text());names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names += [a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom):names.append(n.module or '')
        self.assertFalse(any('compression' in n or 'ep_phy' in n for n in names))

class FrozenMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(score.OLD));import h800_ep_pilot_metrics_v1 as old
        cls.old=old
        cls.code=score.R.parent.parent/'var_comm_runtime_20261010/assets/pilot_metric_code_seed_v1'
    def test_psnr_original_projection(self):
        f,proof=self.old.psnr_projection(self.code/'score_bpg_cached_holdout_r2.py')
        x=np.zeros((3,256,256),dtype=np.float32);y=np.full_like(x,.5)
        self.assertAlmostEqual(f(x,y),6.020599913279624,places=12)
        self.assertEqual(proof['original_source_sha256'],self.old.SOURCE_SHA['score_bpg_cached_holdout_r2.py'])
    def test_lpips_original_ast_no_load(self):
        _,_,proof=self.old.lpips_projection(self.code/'quality.py',{})
        self.assertTrue(proof['deleted_only_after_LPIPS_prefix']);self.assertFalse(proof['unused_DINO_S_constructed'])
        self.assertEqual(proof['pair_expression_ast'],'88a6cdf7eaad9d35c1093102620cfe449ebbae6137d644ecde7e60f24870f84f')
if __name__=='__main__':unittest.main(verbosity=2)
