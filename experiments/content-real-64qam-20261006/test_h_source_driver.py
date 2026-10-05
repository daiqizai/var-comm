import unittest
import numpy as np
import h_source_driver as s

class Tensor:
    def __init__(self,a):self.a=np.asarray(a)
    def __getitem__(self,i):return Tensor(self.a[i])
    def argmax(self,i):return Tensor(self.a.argmax(i))
    def cpu(self):return self
    def numpy(self):return self.a.copy()
    def clamp(self,a,b):return Tensor(self.a.clip(a,b))

class FakePrior:
    instances=[]
    def __init__(self,*args):self.values=[];self.fhat=None;self.closed=False;self.__class__.instances.append(self)
    def logits(self,k):
        # A visible dependency on the complete accepted history, no source data.
        a=np.zeros((1,s.SIZES[k]**2,4096),dtype=np.float32)
        a[:,:,sum(int(x.sum()) for x in self.values)%4096]=1
        return Tensor(a)
    def advance(self,chosen,k):self.values.append(chosen.copy());self.fhat=chosen
    def close(self):self.closed=True

class FakeNative:
    class receiver:_Prior=FakePrior
    loaded=dict(vae=None,var=None,device=None,decoder=lambda x:Tensor(np.zeros((1,3,256,256),np.float32)))

class Tests(unittest.TestCase):
    def test_partial_received_information_first_changes_finer_history(self):
        s.render_received(FakeNative(),np.array([3,7]),1,1)
        p=FakePrior.instances[-1]
        np.testing.assert_array_equal(p.values[0],[3])
        np.testing.assert_array_equal(p.values[1],[7,3,3,3])
        np.testing.assert_array_equal(p.values[2],np.full(9,19))
        self.assertTrue(p.closed)
    def test_receiver_rejects_future_source_tokens(self):
        with self.assertRaises(ValueError):s.render_received(FakeNative(),np.arange(680),7,1)
    def test_actual_wrong_but_in_range_token_not_corrected(self):
        s.render_received(FakeNative(),np.array([4095]),1)
        self.assertEqual(FakePrior.instances[-1].values[0][0],4095)
    def test_pixel_psnr_is_sourcewise_and_reports_mse(self):
        r=s.pixel_scores(np.zeros((3,2,2),np.float32),np.ones((3,2,2),np.float32)*.5)
        self.assertEqual(r['mse'],.25);self.assertAlmostEqual(r['psnr_db'],6.020599913)
    def test_token_split_rejects_invalid_and_copies(self):
        x=np.arange(680,dtype=np.int64);z=s.split_tokens(x);z[0][0]=4
        self.assertEqual(x[0],0);self.assertEqual([len(a) for a in z],[v*v for v in s.SIZES])
        with self.assertRaises(ValueError):s.split_tokens(np.ones(680)*.5)

if __name__=='__main__':unittest.main()
