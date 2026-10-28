import unittest
import numpy as np
from fit_protocol import *
class ProtocolTest(unittest.TestCase):
    def test_catalogue(self):
        c=catalogue();self.assertLess(len(c),4096);self.assertEqual(len(endpoints()),62)
        self.assertEqual(len({OFFSETS[m]+K for m,K in endpoints()}),62)
    def test_boundaries_and_errors(self):
        for K in(0,1,12,25):
            streams=[np.ones(302,np.uint8),np.ones(61,np.uint8)if K//2 else np.empty(0,np.uint8),np.ones(59,np.uint8)if K else np.empty(0,np.uint8)]
            x=pack_compact(117,streams);y=parse_compact(117,K,x)
            self.assertEqual(y['status'],'DIRECTORY_PASS')
            for a,b in zip(streams,y['streams']):self.assertTrue(np.array_equal(a,b))
            bad=x.copy();bad[0]^=1;self.assertEqual(parse_compact(117,K,bad)['status'],'DIRECTORY_CRC_REJECT')
            self.assertEqual(parse_compact(118,K,x)['status'],'DIRECTORY_CRC_REJECT')
            bad=x.copy();bad[-1]^=1;self.assertEqual(parse_compact(117,K,bad)['status'],'DIRECTORY_PASS')
            bad=x.copy();bad[46]^=1;self.assertIsNone(parse_compact(117,K,bad)['streams'][0])
            if K:
                for role in(1,2):
                    if not len(streams[role]):continue
                    bad=x.copy();bad[46+sum(len(z)+16 for z in streams[:role])]^=1
                    parsed=parse_compact(117,K,bad);self.assertIsNone(parsed['streams'][role]);self.assertIsNotNone(parsed['streams'][0])
        x=pack_compact(117,[np.ones(538,np.uint8),np.ones(2,np.uint8),np.ones(2,np.uint8)])
        self.assertEqual(parse_compact(117,2,x)['padding_bits'],0)
        self.assertEqual(modes(True,False,True,1),dict(ALL='CB',SEQ='CB',SEL='CB'))
        self.assertEqual(modes(True,False,True,12),dict(ALL='GRAY',SEQ='C',SEL='CB'))
if __name__=='__main__':unittest.main()
