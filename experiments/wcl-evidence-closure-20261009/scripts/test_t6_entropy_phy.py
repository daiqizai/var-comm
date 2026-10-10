"""Framing and actual received-header dispatch tests; zero real PHY/model calls."""
import inspect,unittest
import numpy as np
import t6_entropy_phy as p

class Entropy2048Tests(unittest.TestCase):
    def test_grid_paid_capacity_m10(self):
        rows=p.profile_templates('EC_VAR_WHOLE')
        self.assertEqual(len(rows),84);self.assertEqual({x['m']for x in rows},set(range(4,11)))
        self.assertEqual(min(x['source_capacity_bits']for x in rows),1951)
        self.assertEqual(max(x['source_capacity_bits']for x in rows),8191)
        self.assertTrue(all(x['body_symbols']==1980 and x['header_symbols']==68 and x['frame_padding_symbols']==0 for x in rows))
        self.assertTrue(all(x['token_count']==680 for x in rows if x['m']==10))

    def test_framing_maximum_and_known_padding(self):
        for profile in p.profile_templates('EC_VAR_WHOLE'):
            bits=np.ones(profile['source_capacity_bits'],np.uint8);info=p.pack_body(bits,profile);parsed=p.parse_body(info,profile)
            self.assertEqual(parsed['payload'],bits.tolist());self.assertEqual(parsed['source_capacity_bits'],min(profile['k']-29,8191))
            corrupt=info.copy();corrupt[-1]^=1;self.assertEqual(p.parse_body(corrupt,profile)['status'],'CRC_REJECT')
            small=p.pack_body(np.array([0,1],np.uint8),profile);small[15]=1;small=p.old.append_crc(small[:-16])
            self.assertEqual(p.parse_body(small,profile)['invalid_reason'],'NONZERO_KNOWN_PADDING')

    def test_length_field_cannot_silently_expand(self):
        profile=max(p.profile_templates('EC_VAR_WHOLE'),key=lambda x:x['k'])
        with self.assertRaises(RuntimeError):p.pack_body(np.zeros(8192,np.uint8),profile)

    def test_rx_api_has_no_transmitter_truth(self):
        args=inspect.signature(p.receive_frame).parameters
        self.assertFalse(set(args)&{'tx_profile','payload','tokens','target','profile_id','source_id'})
        class RT:profiles={'71':{'m':9},'83':{'m':10}}
        self.assertEqual(p.select_received_profile(RT(),dict(header_ok=True,profile_id=83)),{'m':10})

    def test_same_n2048_noise_raw_and_entropy(self):
        import t6_raw_phy as raw
        np.testing.assert_array_equal(raw.standard_noise('source',4101),p.standard_noise('source',4101))
        self.assertEqual(p.standard_noise('source',4101).shape,(2048,2))

if __name__=='__main__':unittest.main()
