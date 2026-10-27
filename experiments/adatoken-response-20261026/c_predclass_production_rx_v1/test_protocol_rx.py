"""Real arithmetic, synthetic tokens/CDF and fake native model only."""
from pathlib import Path
import sys,os,unittest,contextlib
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from c_arithmetic import load_codec,canonical
from common import asha
from protocol_rx import realize,SourceRejected


class Toy:
    def __init__(self,table):
        self.table=table;self.calls=[];self.previous=None
        self.torch=SimpleNamespace(random=SimpleNamespace(fork_rng=self.fork),manual_seed=self.seed)
    def fork(self,devices):self.calls.append(('fork',devices));return contextlib.nullcontext()
    def seed(self,seed):self.calls.append(('seed',seed))
    def start(self,cl):self.calls.append(('start',cl));self.previous=None
    def logits(self,i):self.calls.append(('logits',i,self.previous));return i
    def cdf(self,logits,fun):self.calls.append(('cdf',logits));return self.table
    def tail(self,logits):self.calls.append(('tail',logits));return np.int64(logits%16384)
    def render(self,tokens):
        self.calls.append(('render',tuple(tokens)))
        return np.full((3,256,256),np.mean(tokens)/16384,np.float32)


def original(n,Encoder,Decoder,cdf,bits,prefix,predicted_class,seed):
    # Reference the preserved qualification sequence; it has no new rejection path.
    n.start(predicted_class);dec=Decoder(bits);enc=Encoder();tokens=[];traces=[]
    for i in range(prefix):
        logits=n.logits(i);table=n.cdf(logits,cdf);traces.append(asha(table))
        v=dec.decode(table);enc.encode(v,table);tokens.append(int(v[0]));n.previous=tokens[-1]
    canonical(dec,enc,bits);received=np.array(tokens,np.int64)
    with n.torch.random.fork_rng(devices=[0]):
        n.torch.manual_seed(seed)
        for i in range(prefix,256):
            logits=n.logits(i);v=n.tail(logits);tokens.append(int(v.item()));n.previous=tokens[-1]
    tokens=np.array(tokens,np.int64);image=n.render(tokens)
    return image,received,tokens,traces


class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Encoder,cls.Decoder,cls.cdf,cls.proof=load_codec(os.environ['SYNTHETIC_CODEC_ROOT'])
        cls.table=np.arange(16385,dtype=np.int64)[None,:]*1024
    def stream(self,count):
        e=self.Encoder()
        for i in range(count):e.encode(np.array([(i*37)%16384],np.int64),self.table)
        return e.finish()
    def result(self,bits,prefix,n=None,Decoder=None):
        n=Toy(self.table)if n is None else n
        return realize(n,self.Encoder,Decoder or self.Decoder,type(self).cdf,bits,prefix,101,12345),n
    def test_success_sequence_prefix1(self):self.check_success(1)
    def test_success_sequence_prefix128(self):self.check_success(128)
    def test_success_sequence_prefix256(self):self.check_success(256)
    def check_success(self,prefix):
        bits=self.stream(prefix);a=Toy(self.table);b=Toy(self.table)
        ref=original(a,self.Encoder,self.Decoder,type(self).cdf,bits,prefix,101,12345)
        out,_=self.result(bits,prefix,b)
        for key,value in zip(('image','received_tokens','completed_tokens','cdf_hashes'),ref):
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(out[key],value)
            else:self.assertEqual(out[key],value)
        self.assertEqual(a.calls,b.calls);self.assertEqual(out['status'],'ACCEPTED_SOURCE')
        self.assertEqual((out['prior_forwards'],out['prior_branch_forwards'],out['decoder_forwards']),(256,512,1))
    def assert_gray(self,out,n,reason):
        self.assertEqual(out['status'],'FIXED_GRAY_SOURCE_REJECT');self.assertEqual(out['reason'],reason)
        np.testing.assert_array_equal(out['image'],np.full((3,256,256),.5,np.float32))
        self.assertEqual(out['decoder_forwards'],0);self.assertEqual(out['completed_tokens'].size,0)
        self.assertFalse(any(c[0]in('tail','render','fork','seed')for c in n.calls))
    def test_extra_terminal_bit_is_gray(self):
        out,n=self.result(np.concatenate([self.stream(1),np.array([0],np.uint8)]),1)
        self.assert_gray(out,n,'NONCANONICAL_ARITHMETIC_LENGTH_OR_TERMINATION')
    def test_exhausted_wire_is_gray(self):
        out,n=self.result(np.zeros(2,np.uint8),64)
        self.assert_gray(out,n,'ARITHMETIC_TERMINAL_LOOKAHEAD_EXCEEDED')
        self.assertLess(out['prior_forwards'],64)
    def test_terminal_position_mismatch_is_gray(self):
        Base=self.Decoder
        class Shift(Base):
            def decode(self,table):
                v=super().decode(table);self.position+=1;return v
        out,n=self.result(self.stream(1),1,Decoder=Shift)
        self.assert_gray(out,n,'ARITHMETIC_TERMINAL_LOOKAHEAD_MISMATCH')
    def test_model_failure_not_misclassified_as_wire(self):
        n=Toy(self.table)
        def fail(i):raise ValueError('Arithmetic terminal lookahead exceeded')
        n.logits=fail
        with self.assertRaisesRegex(ValueError,'lookahead exceeded'):self.result(self.stream(1),1,n)
    def test_invalid_cdf_remains_fatal(self):
        n=Toy(np.zeros((1,16385),np.int64))
        with self.assertRaises((ValueError,AssertionError)):self.result(self.stream(1),1,n)
    def test_unknown_decoder_failure_remains_fatal(self):
        Base=self.Decoder
        class Bad(Base):
            def decode(self,table):raise RuntimeError('unexpected allocation failure')
        with self.assertRaisesRegex(RuntimeError,'allocation'):self.result(self.stream(1),1,Decoder=Bad)
    def test_unknown_canonical_assertion_remains_fatal(self):
        with patch('protocol_rx.canonical',side_effect=AssertionError('unexpected codec inconsistency')):
            with self.assertRaisesRegex(AssertionError,'codec inconsistency'):self.result(self.stream(1),1)
    def test_no_model_import(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
        self.assertTrue(self.proof['inverse_AST_equal'])
    def test_receiver_payload_cannot_name_source_assets(self):
        import ast
        tree=ast.parse(Path(__file__).with_name('worker.py').read_text())
        fields=[v for v in ast.walk(tree)if isinstance(v,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='fields'for t in v.targets)]
        self.assertEqual(len(fields),1)
        self.assertEqual(ast.literal_eval(fields[0].value),{'key','prefix','bits','payload_sha256','received_class_bits','container_sha256'})


if __name__=='__main__':unittest.main(verbosity=2)
