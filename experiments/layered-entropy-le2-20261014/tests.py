"""Synthetic metadata/parser/fallback tests; no model or PHY forwards."""
import unittest,ast,tempfile,sys,importlib.util
from common import *
import planning as p,numpy as np
spec=importlib.util.spec_from_file_location('_test_paid_parser',E1/'packet_two_group.py');packet=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet)
class Tests(unittest.TestCase):
 def test_equal_proportion(self):self.assertEqual(p.proportion(100,100),478)
 def test_overhead_proportion(self):self.assertEqual(p.proportion(1000,0),930);self.assertNotEqual(p.proportion(1000,0),956)
 def test_even_clamp(self):
  for b,e in ((1,10000),(10000,1),(123,456)):self.assertEqual(p.proportion(b,e)%2,0);self.assertTrue(2<=p.proportion(b,e)<=954)
 def test_pack_parse(self):
  bits=np.array([1,0,1,1],np.uint8);r=packet.parse_group(packet.pack_group(bits,31+20),51);self.assertTrue(r['accepted']);np.testing.assert_array_equal(r['bits'],bits)
 def test_crc_reject(self):
  a=packet.pack_group(np.array([1,0]),50);a[13]^=1;self.assertFalse(packet.parse_group(a,50)['accepted'])
 def test_empty_profile(self):
  a=packet.pack_group(np.array([],np.uint8),50,True);self.assertTrue(packet.parse_group(a,50,True)['accepted']);self.assertFalse(packet.parse_group(a,50)['accepted'])
 def test_padding_reject(self):
  a=packet.pack_group(np.array([1,0]),50);a[20]=1;a[-16:]=packet.intbits(packet.crc16(a[:-16]),16);self.assertEqual(packet.parse_group(a,50)['status'],'KNOWN_PADDING_REJECT')
 def test_information_short_reject(self):self.assertFalse(packet.parse_group(np.zeros(49,np.uint8),50)['accepted'])
 def test_invalid_length(self):
  a=packet.pack_group(np.array([1,0]),50);a[:13]=packet.intbits(8191,13);a[-16:]=packet.intbits(packet.crc16(a[:-16]),16);self.assertFalse(packet.parse_group(a,50)['accepted'])
 def fixture(self):
  d={};Q={}
  for s in range(50):
   for end in ENDS:Q[s,*end]=.01*(OFFSETS[end[0]]+end[1])
   for j in (4,5,6,7):
    for end in ENDS:
     if OFFSETS[end[0]]+end[1]>OFFSETS[j]:d[s,j,*end]=dict(b_base=str(j*100+s),b_enhancement=str((OFFSETS[end[0]]+end[1])*2),b_single=str((OFFSETS[end[0]]+end[1])*3))
  return d,Q,np.zeros(50)
 def test_actual_per_source_fallback(self):
  st=p.source_states(*self.fixture(),6,(8,0),620,100);self.assertEqual(st[0]['actual_j'],6);self.assertEqual(st[49]['actual_j'],5);self.assertTrue(all(s['status']=='EMPTY_ENHANCEMENT' for s in st))
 def test_null_kept(self):
  st=p.source_states(*self.fixture(),6,(8,0),399,100);self.assertEqual(len(st),50);self.assertTrue(all(s['status']=='NULL' for s in st))
 def test_single_base4_endpoint(self):
  st=p.source_states(*self.fixture(),None,(4,0),450);self.assertEqual(st[0]['actual_m'],4);self.assertEqual(st[-1]['L_B'],449)
 def test_proxy_null(self):
  r=dict(B='b',E='e',A=0.,Bcoef=0.);t={(g,s):dict(p_correct=.5,ci=[.1,.9]) for g in ('b','e','header') for s in SNRS};self.assertEqual(p.proxy(r,t,.123)['J'],.123)
 def test_shared_header_delta(self):
  r=dict(B='b',E=None,A=.5,Bcoef=0.);t={(g,s):dict(p_correct=.5,ci=[.1,.9]) for g in ('b','header') for s in SNRS};self.assertEqual(p.delta_bounds(r,r,t),(0.,0.))
 def test_same_body_quadratic(self):
  r=dict(B='b',E='b',A=.8,Bcoef=-.6);t={(g,s):dict(p_correct=.5,ci=[.1,.9]) for g in ('b','header') for s in SNRS};lo,hi=p.interval(r,t,0);self.assertAlmostEqual(hi,.9*(.8*(2/3)-.6*(2/3)**2));self.assertLessEqual(lo,hi)
 def test_receiver_no_oracle(self):
  node=next(n for n in ast.parse((E/'packets.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='receive');names={a.arg for a in node.args.args};self.assertFalse(names&{'truth','payload','source','CDF','tokens'})
 def test_budget(self):self.assertEqual(96*128+24*384,CAP);self.assertEqual((32+16*4)*128+12*2*384,CAP)
 def test_once_ledger(self):
  with tempfile.TemporaryDirectory(prefix='LE2_SYNTHETIC_') as td:
   l=Ledger(Path(td)/'calls.sqlite');l.reserve('synthetic',[1],1);l.finish('synthetic',[1]);self.assertRaises(AssertionError,l.reserve,'synthetic',[2],1);self.assertEqual(l.summary()[0]['status'],'COMPLETE');l.db.close()
 def test_wilson_zero_errors(self):self.assertLess(wilson(128,128)[0],1);self.assertGreater(wilson(0,128)[1],0)
 def test_scope_no_extra_stages(self):self.assertEqual(SNRS,(7,10));self.assertEqual(len(p.MCS),15)
if __name__=='__main__':unittest.main(verbosity=2)
