"""CPU engineering fixtures, not a real-weight/GPU qualification.

NumPy fixtures execute the exact helper interface and reject scalar reads in
its position loop. Native PyTorch CPU fixtures run too when Torch is installed.
No original experiment module, image, model, source binding, or GPU is loaded.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import acceleration as speed

try:
    import torch
except ImportError:
    torch = None


class Tensor:
    """Small NumPy implementation of the tensor operations used by the sweep."""
    def __init__(self, value):self.value=np.asarray(value)
    @property
    def shape(self):return self.value.shape
    @property
    def ndim(self):return self.value.ndim
    @property
    def T(self):return Tensor(self.value.T)
    def detach(self):return self
    def squeeze(self, dim):return Tensor(self.value.squeeze(dim))
    def reshape(self,*shape):return Tensor(self.value.reshape(*shape))
    def argmax(self):return Tensor(np.asarray(self.value.argmax(),dtype=np.int64))
    def index_select(self,dim,index):return Tensor(np.take(self.value,index.value,axis=dim))
    def copy_(self,value):self.value[...] = value.value;return self
    def cpu(self):return self
    def tolist(self):return self.value.tolist()
    def __getitem__(self,key):return Tensor(self.value[key])
    def __matmul__(self,other):return Tensor(self.value @ other.value)
    def __add__(self,other):return Tensor(self.value + (other.value if isinstance(other,Tensor) else other))
    def __sub__(self,other):return Tensor(self.value - (other.value if isinstance(other,Tensor) else other))
    def __mul__(self,other):return Tensor(self.value * (other.value if isinstance(other,Tensor) else other))
    __rmul__=__mul__
    def __truediv__(self,other):return Tensor(self.value / other)
    def __ne__(self,other):return Tensor(self.value != other.value)
    def __int__(self):raise AssertionError('Position loop performed a device scalar read')
    def __float__(self):raise AssertionError('Position loop performed a device scalar read')
    def __bool__(self):raise AssertionError('Position loop performed a device scalar branch')


class Backend:
    @staticmethod
    def stack(tensors):return Tensor(np.stack([v.value for v in tensors]))
    @staticmethod
    def where(mask,a,b):return Tensor(np.where(mask.value,a.value,b.value))


def operator(E,G,backend='numpy'):
    if backend=='numpy':
        gram=np.swapaxes(G,1,2) @ G
        quadratic=np.einsum('vc,pcd,vd->pv',E,gram,E)
        arrays=[Tensor(x) for x in (E,G,gram,quadratic)]
    else:
        gram=G.transpose(1,2) @ G
        quadratic=torch.einsum('vc,pcd,vd->pv',E,gram,E)
        arrays=[E,G,gram,quadratic]
    weight,columns,gram,quadratic=arrays
    return SimpleNamespace(q=SimpleNamespace(embedding=SimpleNamespace(weight=weight)),
        columns=columns,gram=gram,quadratic=quadratic)


def numpy_reference(E,G,gram,quadratic,tokens,error,variance,prior=None,static=None,lam=0.):
    tokens=tokens.copy();error=error.copy();gain=0.;changes=0
    for p in range(tokens.shape[1]):
        old=int(tokens[0,p]);b=G[p].T @ error + gram[p] @ E[old]
        scores=(E @ b - .5*quadratic[p])/float(variance)
        lp=prior[0,p] if prior is not None else static
        if lp is not None:scores=scores+float(lam)*lp
        new=int(scores.argmax());gain+=float(scores[new]-scores[old])
        if new!=old:
            error-=G[p] @ (E[new]-E[old]);tokens[0,p]=new;changes+=1
    return tokens,error,gain,changes


class SweepTests(unittest.TestCase):
    def test_fp32_numpy_reference_var_static_likelihood_and_no_scalar_reads(self):
        rng=np.random.default_rng(20261002)
        for mode in ('LIKELIHOOD','STATIC','VAR'):
            E=rng.normal(size=(31,4)).astype(np.float32)
            G=rng.normal(size=(9,7,4)).astype(np.float32)
            op=operator(E,G);tokens=rng.integers(0,len(E),(1,9),dtype=np.int64)
            error=rng.normal(size=7).astype(np.float32)
            pri=rng.normal(size=(1,9,len(E))).astype(np.float32) if mode=='VAR' else None
            sta=rng.normal(size=len(E)).astype(np.float32) if mode=='STATIC' else None
            expected=numpy_reference(E,G,op.gram.value,op.quadratic.value,tokens,error,.37,pri,sta,1.5)
            weights=E.copy()
            actual=speed.coordinate_sweep(op,Tensor(tokens.copy()),Tensor(error.copy()),.37,
                log_probs=None if pri is None else Tensor(pri),
                static_log_prior=None if sta is None else Tensor(sta),lam=1.5,_torch=Backend)
            np.testing.assert_array_equal(actual['tokens'].value,expected[0])
            np.testing.assert_array_equal(actual['error'].value.view(np.uint32),expected[1].view(np.uint32))
            self.assertEqual(actual['surrogate_objective_increase'],expected[2])
            self.assertEqual(actual['token_changes'],expected[3]);np.testing.assert_array_equal(E,weights)

    def test_sequential_coupling_is_not_a_simultaneous_update(self):
        E=np.array([[0.],[1.]],dtype=np.float32);G=np.ones((2,1,1),dtype=np.float32)
        out=speed.coordinate_sweep(operator(E,G),Tensor(np.zeros((1,2),dtype=np.int64)),
            Tensor(np.ones(1,dtype=np.float32)),1.,_torch=Backend)
        np.testing.assert_array_equal(out['tokens'].value,[[1,0]])
        self.assertEqual(out['token_changes'],1)
        # Scoring both positions against the initial error would choose [1,1].
        self.assertNotEqual(out['tokens'].value.tolist(),[[1,1]])

    def test_unchanged_token_retains_error_bits_and_first_argmax_tie(self):
        E=np.zeros((3,2),dtype=np.float32);G=np.ones((4,4,2),dtype=np.float32)
        error=np.array([0.,-0.,-1.,2.],dtype=np.float32)
        out=speed.coordinate_sweep(operator(E,G),Tensor(np.zeros((1,4),dtype=np.int64)),
            Tensor(error.copy()),1.,_torch=Backend)
        np.testing.assert_array_equal(out['error'].value.view(np.uint32),error.view(np.uint32))
        np.testing.assert_array_equal(out['tokens'].value,np.zeros((1,4),dtype=np.int64))
        self.assertEqual(out['token_changes'],0);self.assertEqual(out['surrogate_objective_increase'],0.)

    def test_input_validation_and_no_truth_interface(self):
        import inspect
        for name in ('clean','truth','target','source_id','source_index'):
            self.assertNotIn(name,inspect.signature(speed.infer).parameters)
            self.assertNotIn(name,inspect.signature(speed.coordinate_sweep).parameters)
        E=np.zeros((3,2),dtype=np.float32);G=np.ones((2,1,2),dtype=np.float32)
        op=operator(E,G);tokens=Tensor(np.zeros((1,2),dtype=np.int64));error=Tensor(np.zeros(1,dtype=np.float32))
        for variance in (0.,-1.,np.nan,np.inf):
            with self.assertRaises(ValueError):speed.coordinate_sweep(op,tokens,error,variance,_torch=Backend)
        with self.assertRaises(ValueError):
            speed.coordinate_sweep(op,tokens,error,1.,log_probs=Tensor(np.zeros((1,2,3))),
                static_log_prior=Tensor(np.zeros(3)),_torch=Backend)

    @unittest.skipIf(torch is None,'Native PyTorch CPU is unavailable in the local bundled runtime')
    def test_native_torch_cpu_fp32_reference(self):
        torch.manual_seed(20261002)
        for mode in ('LIKELIHOOD','STATIC','VAR'):
            E=torch.randn(31,4,dtype=torch.float32);G=torch.randn(9,7,4,dtype=torch.float32)
            op=operator(E,G,'torch');original_tokens=torch.randint(0,31,(1,9));error=torch.randn(7)
            pri=torch.randn(1,9,31) if mode=='VAR' else None
            sta=torch.randn(31) if mode=='STATIC' else None
            expected_tokens=original_tokens.clone();expected_error=error.clone();gain=0.;changes=0
            for p in range(9):
                old=int(expected_tokens[0,p]);b=G[p].T @ expected_error+op.gram[p] @ E[old]
                scores=(E @ b-.5*op.quadratic[p])/.37
                lp=pri[0,p] if pri is not None else sta
                if lp is not None:scores=scores+1.5*lp
                new=int(scores.argmax());gain+=float(scores[new]-scores[old])
                if new!=old:
                    expected_error-=G[p] @ (E[new]-E[old]);expected_tokens[0,p]=new;changes+=1
            actual=speed.coordinate_sweep(op,original_tokens.clone(),error.clone(),.37,
                log_probs=pri,static_log_prior=sta,lam=1.5)
            self.assertTrue(torch.equal(actual['tokens'],expected_tokens))
            self.assertTrue(torch.equal(actual['error'],expected_error))
            self.assertEqual(actual['surrogate_objective_increase'],gain)
            self.assertEqual(actual['token_changes'],changes)


if __name__=='__main__':unittest.main()
