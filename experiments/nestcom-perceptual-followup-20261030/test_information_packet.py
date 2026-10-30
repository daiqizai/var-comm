"""Pure engineering tests; these do not certify image quality or wireless PHY."""
import unittest
from information_packet import *
class PacketTests(unittest.TestCase):
 def test_real_paid_fields_and_crc(self):
  c=catalogue();f=primitives()
  for p in [c[0],c[15],c[650],c[651],c[906]]:
   bits=np.array([1,0,1,1,0,0],np.uint8);q,proof=transfer(bits,p,620,999 if p['codec']=='ADA'else None)
   self.assertEqual(q['bits'],bits.tolist());self.assertEqual(proof['wire_bits'],620)
   if p['codec']=='ADA':self.assertEqual(q['received_class'],999)
  protected=f['append_crc'](np.zeros(12,np.uint8));protected[3]^=1;self.assertFalse(f['crc_ok'](protected))
 def test_capacity_boundary(self):
  for p,over in [(catalogue()[1],57),(catalogue()[651],67)]:
   b=np.zeros(620-over,np.uint8);q,v=transfer(b,p,620,1 if over==67 else None);self.assertEqual(v['padding_bits'],0)
   with self.assertRaises(RuntimeError):transfer(np.zeros(621-over,np.uint8),p,620,1 if over==67 else None)
 def test_seed_identity(self):
  self.assertEqual(generation_seed('a/b',2026103001),generation_seed('a/b',2026103001))
  self.assertNotEqual(generation_seed('a/b',2026103001),generation_seed('a/c',2026103001))
if __name__=='__main__':unittest.main()
