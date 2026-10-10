"""Engineering invariants only: no backend, channel or decoder is executed."""
import inspect
import unittest
import numpy as np
import ep_phy as new
import t1_phy as old


class PaidExtensionTests(unittest.TestCase):
    def test_preserves_all_old_profile_ids_and_capacity(self):
        profiles = new.profile_templates()
        self.assertEqual(profiles[:144], old.profile_templates())
        self.assertEqual(len(profiles),360)
        self.assertEqual([p['profile_id'] for p in profiles],list(range(360)))
        for p in profiles[144:]:
            self.assertEqual(p['family'],'EC_VAR_PARTIAL')
            self.assertIn(p['K'],new.ks(p['m']))
            self.assertEqual(p['token_count'],sum(x*x for x in new.SIZES[:p['m']])+p['K'])
            self.assertEqual(p['source_capacity_bits'],p['k']-13-16)
            self.assertEqual(p['header_symbols']+p['body_symbols'],1024)

    def test_whole_packet_bits_and_wire_randomness_unchanged(self):
        a = np.array([1,0,1,0,0,1,1],dtype=np.uint8)
        for p in old.profile_templates():
            np.testing.assert_array_equal(new.pack_body(a,p),old.pack_body(a,p))
        for seed in [4101,6201,9301]:
            np.testing.assert_array_equal(new.standard_noise('same_source',seed),
                                          old.standard_noise('same_source',seed))
        np.testing.assert_array_equal(new.scramble_mask(5736,127,old.PROTOCOL),
                                      old.scramble_mask(5736,127,old.PROTOCOL))
        for f in [new.transmit_body,new.transmit_frame,new.receive_body,new.receive_frame]:
            self.assertEqual(inspect.signature(f).parameters['session'].default,old.PROTOCOL)

    def test_paid_header_controls_interpretation_without_tx_input(self):
        p = new.profile_templates()[359]
        runtime = type('Runtime',(),{'profiles':{str(x['profile_id']):x for x in new.profile_templates()}})()
        actual = new.select_received_profile(runtime,{'header_ok':True,'profile_id':359})
        self.assertEqual(actual,p)
        self.assertIsNone(new.select_received_profile(runtime,{'header_ok':False,'profile_id':None}))
        forbidden = {'tx_profile','tokens','target','payload','profile_id','m','K'}
        self.assertFalse(forbidden.intersection(inspect.signature(new.receive_frame).parameters))

    def test_crc_valid_illegal_length_is_not_silently_truncated(self):
        p = new.profile_templates()[144]
        info = new.pack_body(np.array([0,1],dtype=np.uint8),p)
        for length in [0,1,p['source_capacity_bits']+1]:
            invalid = info.copy()
            invalid[:13] = new.integer_bits(length,13)
            invalid = new.append_crc(invalid[:-16])
            parsed = new.parse_body(invalid,p)
            self.assertEqual(parsed['status'],'DECODE_INVALID')
            self.assertIsNone(parsed['payload'])


if __name__=='__main__':
    unittest.main()
