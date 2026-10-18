import unittest
from extract import interval,exact_event
class Evidence(unittest.TestCase):
    def test_zero_errors_not_zero_upper_bler(self):
        lo,hi=interval(100,100,.05/36);self.assertLess(lo,1);self.assertEqual(hi,1);self.assertGreater(lo,0)
    def test_all_errors(self):
        lo,hi=interval(0,100,.05/36);self.assertEqual(lo,0);self.assertGreater(hi,0)
    def test_success(self):
        p={'transmission':{'profile_id':2,'information_sha256':'abc'},'actual_RX':{'status':'PAYLOAD_PARSED','header':{'header_ok':True,'profile_id':2},'body':{'hard_bits_sha256':'abc','crc_accepted':True,'parser_accepted':True}}}
        self.assertTrue(exact_event(p)['joint_exact_success'])
        p['actual_RX']['body']['hard_bits_sha256']='wrong'
        self.assertTrue(exact_event(p)['accepted_but_information_or_profile_wrong']);self.assertFalse(exact_event(p)['joint_exact_success'])
    def test_header_failure_no_body(self):
        p={'transmission':{'profile_id':2,'information_sha256':'abc'},'actual_RX':{'status':'HEADER_REJECT','header':{'header_ok':False,'profile_id':None},'body':None}}
        self.assertFalse(exact_event(p)['joint_exact_success'])
    def test_wrong_profile_not_corrected(self):
        p={'transmission':{'profile_id':2,'information_sha256':'abc'},'actual_RX':{'status':'PAYLOAD_PARSED','header':{'header_ok':True,'profile_id':3},'body':{'hard_bits_sha256':'abc','crc_accepted':True,'parser_accepted':True}}}
        self.assertFalse(exact_event(p)['joint_exact_success']);self.assertTrue(exact_event(p)['accepted_but_information_or_profile_wrong'])
if __name__=='__main__':unittest.main(verbosity=2)
