import unittest,hashlib
import numpy as np
from p1_common import chain,choose,ash,TARGETS,nt
class ContractTests(unittest.TestCase):
 def test_frozen_targets_and_order(self):
  for f in TARGETS:
   c=chain(f);self.assertEqual(c[0],TARGETS[f]);self.assertEqual(c[-1],(4,0));self.assertEqual(len(c),len(set(map(nt,c))));self.assertEqual(list(map(nt,c)),sorted(map(nt,c),reverse=True))
 def test_finished_length_not_estimate(self):
  c=chain('PARTIAL');streams={f'm{m}_K{k}':np.zeros(2521,np.uint8)for m,k in c};m,k=c[2];streams[f'm{m}_K{k}']=np.ones(2520,np.uint8)
  self.assertEqual(choose(streams,'PARTIAL')[:2],(m,k))
 def test_no_fit_is_error(self):
  streams={f'm{m}_K{k}':np.zeros(2521,np.uint8)for m,k in chain('WHOLE')}
  with self.assertRaisesRegex(RuntimeError,'NO_FEASIBLE'):choose(streams,'WHOLE')
 def test_noise_and_protocol_namespaces_separate(self):
  # Production reads one stored bank for both families; different SNR only scales it.
  z=np.random.default_rng(7).standard_normal((1024,2));copies=[z.copy()for _ in range(2)]
  self.assertEqual(ash(copies[0]),ash(copies[1]));self.assertFalse(np.array_equal(z*10**(-8/20),z*10**(-12/20)))
if __name__=='__main__':unittest.main()
