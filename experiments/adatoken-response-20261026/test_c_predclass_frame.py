import unittest,sys,numpy as np
from common import *
from c_predclass_frame import *
class Frame(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.old=read(sys.argv[1]);cls.cat=catalogue(cls.old)
 def test_paid_profiles_and_unchanged_original(self):
  self.assertEqual(self.cat['profiles'][:360],self.old['profiles']);self.assertLess(len(self.cat['profiles']),4096)
  self.assertEqual(len([p for p in self.cat['profiles']if p['family']==FAMILY]),256*5)
 def test_actual_class_reduces_fitting_capacity(self):
  p=next(p for p in self.cat['profiles']if p['family']==FAMILY and p['q']==2 and p['nominal_rate']=='2/3');cap=p['source_capacity_bits']
  streams={'p%03d'%i:np.zeros(cap-9,np.uint8)for i in range(1,257)};streams['p001']=np.zeros(cap-10,np.uint8)
  selected,bits,checks=choose(streams,self.cat['profiles'],2,'2/3',999);self.assertEqual(selected['m'],1);self.assertEqual(len(bits),cap);self.assertEqual(split_container(bits)[0],999)
 def test_actual_received_class_used(self):
  p=next(p for p in self.cat['profiles']if p['family']==FAMILY);rx=dict(header=dict(header_ok=True),rx_profile=p,body=dict(crc_accepted=True,parser_accepted=True,payload=container(18,np.array([0,1],np.uint8)).tolist()),status='PAYLOAD_PARSED')
  self.assertEqual(source_projection(rx)['received_class'],18)
  rx['body']['payload'][:10]=[1]*10;self.assertEqual(source_projection(rx)['status'],'FIXED_GRAY')
 def test_unknown_received_family_and_failure_gray(self):
  rx=dict(header=dict(header_ok=True),rx_profile=self.old['profiles'][0],body=dict(crc_accepted=True,parser_accepted=True,payload=[0,1]),status='PAYLOAD_PARSED');self.assertEqual(source_projection(rx)['reason'],'UNSUPPORTED_SOURCE_FAMILY')
  rx['header']['header_ok']=False;self.assertEqual(source_projection(rx)['status'],'FIXED_GRAY')
 def test_no_model_import(self):self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main(argv=[sys.argv[0]],verbosity=2)
