import unittest
from adapter import choose,temperatures,generation_function
from types import SimpleNamespace

class Logits:
    def argmax(self,**kw):
        assert kw==dict(dim=-1,keepdim=True)
        return 'EXPLICIT_ARGMAX'

class Tests(unittest.TestCase):
    def test_received_scales_keep_original_sampling(self):
        for arm in ['original','greedy','low_tau025']:
            self.assertEqual(choose(Logits(),arm,2,4,lambda x,**kw:'ORIGINAL',top_p=.97),'ORIGINAL')
    def test_greedy_never_samples_missing(self):
        def forbidden(*a,**k):raise AssertionError('sampled')
        self.assertEqual(choose(Logits(),'greedy',4,4,forbidden),'EXPLICIT_ARGMAX')
    def test_tau_only_missing(self):
        for e in range(1,8):
            for arm in ['original','greedy','low_tau025']:
                self.assertEqual(temperatures(arm,e)[:e],[.5]*e)
                self.assertEqual(temperatures(arm,e)[e:],[.25 if arm=='low_tau025' else .5]*(7-e))
    def test_private_ast(self):
        s='''def decompress_cfg(gt_leak):
 si=0
 a=sample_with_top_k_top_p_also_inplace_modifying_logits_(1)
 b=sample_with_top_k_top_p_also_inplace_modifying_logits_(2)
 return a,b
'''
        r=SimpleNamespace(sample_with_top_k_top_p_also_inplace_modifying_logits_=lambda x,**kw:x)
        f,h=generation_function(r,'original',s)
        self.assertEqual(f(2),(1,2));self.assertEqual(len(h),64)
    def test_unknown_settings(self):
        with self.assertRaises(AssertionError):temperatures('tuned',4)

if __name__=='__main__':unittest.main(verbosity=2)
