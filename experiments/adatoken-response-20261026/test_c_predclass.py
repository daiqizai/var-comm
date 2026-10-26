import unittest,sys,ast
import numpy as np
from pathlib import Path
from common import definitions,sha
from c_predclass_protocol import *

class PredictedClass(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=Path(sys.argv[1])/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py'
  def require(ok,msg):
   if not ok:raise ValueError(msg)
  ns=dict(np=np,require=require,hashlib=hashlib,FAMILIES=('C_PREDICTED_CLASS',))
  names={'binary','integer_bits','bits_integer','crc16','append_crc','crc_ok','array_sha','pack_body','parse_body'}
  definitions(p,names,ns);cls.phy=ns
 def test_all_classes_roundtrip(self):
  for i in range(1000):self.assertEqual(received_class(class_bits(i)),i)
 def test_reserved_and_malformed_class_reject(self):
  for v in (-1,1000,1023,True,1.0):
   with self.assertRaises(ValueError):class_bits(v)
  for v in range(1000,1024):
   with self.assertRaises(ValueError):received_class(np.array([(v>>i)&1 for i in range(9,-1,-1)],np.uint8))
 def test_exact_paid_capacity_and_full_crc(self):
  for k in(956,1274,2549,2868,4780):
   p=dict(k=k,mode='arithmetic',family='C_PREDICTED_CLASS');a=np.zeros(source_capacity(k),np.uint8);a[-1]=1;payload=container(999,a)
   packed=self.phy['pack_body'](payload,p);self.assertEqual(len(packed),k)
   parsed=self.phy['parse_body'](packed,p);self.assertTrue(parsed['parser_accepted']);c,b=split_container(np.array(parsed['payload'],np.uint8));self.assertEqual(c,999);np.testing.assert_array_equal(a,b)
   with self.assertRaises(ValueError):self.phy['pack_body'](container(999,np.r_[a,np.uint8(0)]),p)
   damaged=packed.copy();damaged[13]^=1;self.assertFalse(self.phy['parse_body'](damaged,p)['crc_accepted'])
 def test_received_valid_changed_class_not_corrected(self):
  p=dict(k=956,mode='arithmetic',family='C_PREDICTED_CLASS');payload=container(4,np.array([0,1],np.uint8));packed=self.phy['pack_body'](payload,p)
  parsed=self.phy['parse_body'](packed,p);self.assertEqual(split_container(np.array(parsed['payload'],np.uint8))[0],4)
 def test_crc_valid_reserved_class_reject(self):
  p=dict(k=956,mode='arithmetic',family='C_PREDICTED_CLASS');packed=self.phy['pack_body'](container(999,np.array([0,1],np.uint8)),p);packed[13:23]=1
  packed=self.phy['append_crc'](packed[:-16]);parsed=self.phy['parse_body'](packed,p);self.assertTrue(parsed['crc_accepted'])
  with self.assertRaises(ValueError):split_container(np.array(parsed['payload'],np.uint8))
 def test_seed_binds_class_prefix_and_received_stream(self):
  a=np.array([0,1,0,1],np.uint8);s=generation_seed(a,32,17);self.assertEqual(s,generation_seed(a.copy(),32,17))
  self.assertEqual(len({s,generation_seed(a,32,18),generation_seed(a,33,17),generation_seed(1-a,32,17)}),4)
 def test_no_model_import_for_protocol(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
 def test_frozen_sampling_configuration(self):self.assertEqual(TAIL,dict(temperature=1.0,top_k=2000,top_p=1.0,cfg_scale=4.0,cfg_interval=-1))
 def test_original_model_loader_and_render_unchanged(self):
  root=Path(__file__).parent
  old=ast.parse((root/'c_native.py').read_text());new=ast.parse((root/'c_predclass_native.py').read_text())
  def node(tree,name):return next(n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
  for name in('build','Native'):
   self.assertEqual(ast.dump(node(old,name),include_attributes=False),ast.dump(node(new,name),include_attributes=False))

if __name__=='__main__':unittest.main(argv=[sys.argv[0]],verbosity=2)
