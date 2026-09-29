"""Lossless result storage tests; no model or GPU involved."""
import unittest,json
from rx_v3_archive import pack,unpack,original_bytes,MARK
class ArchiveTests(unittest.TestCase):
    def test_exact_roundtrip_including_negative_zero_and_key_order(self):
        bins=[dict(count=0,confidence_sum=0.0,correct_sum=0) for _ in range(15)]
        bins[3]=dict(count=17,confidence_sum=2.1234567891234567,correct_sum=5)
        bins[4]['confidence_sum']=-0.0
        d=dict(z=[dict(bins=bins)],source='图像',a=1.0)
        restored=unpack(json.loads(json.dumps(pack(d),separators=(',',':'))))
        self.assertEqual(original_bytes(restored),original_bytes(d))
        self.assertEqual(len(pack(d)['z'][0]['bins'][MARK]),2)
    def test_zero_and_numeric_types_preserved(self):
        bins=[dict(count=0,confidence_sum=0.0,correct_sum=0) for _ in range(15)]
        bins[1]['confidence_sum']=0
        self.assertEqual(original_bytes(unpack(pack(dict(bins=bins)))),original_bytes(dict(bins=bins)))
    def test_malformed_sparse_duplicates_rejected(self):
        with self.assertRaises(AssertionError):unpack({MARK:[[1,0,0.0,0],[1,0,0.0,0]]})
    def test_reserved_marker_rejected(self):
        with self.assertRaises(AssertionError):pack({MARK:[]})
if __name__=='__main__':unittest.main()
