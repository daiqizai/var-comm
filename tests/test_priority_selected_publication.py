import unittest
from tools.publish_priority_selected_grid import exact_keys, resource_ledger

class PriorityPublicationTests(unittest.TestCase):
    def test_duplicate_or_missing_noise_key_rejected(self):
        rows=[{'snr':1,'seed':2},{'snr':1,'seed':2}]
        with self.assertRaises(RuntimeError):exact_keys(rows,['snr','seed'],[(1,2),(1,3)],'keys')
        exact_keys([{'snr':1,'seed':2},{'snr':1,'seed':3}],['snr','seed'],[(1,2),(1,3)],'keys')

    def test_punctured_body_not_confused_with_source_bits(self):
        r={'family':'hybrid','m_actual':8,'N':4084,'N_header':68,'N_data':2992,'N_continuous':1024}
        a=resource_ledger(r)
        self.assertEqual((a['source_payload_bits'],a['body_mother_bits'],a['body_coded_slots']),(3060,6164,5984))
        self.assertEqual(a['header_mother_bits'],68)
        self.assertEqual(a['header_coded_slots'],136)
        r['N_continuous']=1023
        with self.assertRaises(RuntimeError):resource_ledger(r)

    def test_pure_has_no_digital_payload(self):
        r={'family':'continuous','N':4084,'N_header':0,'N_data':0,'N_continuous':4084}
        a=resource_ledger(r)
        self.assertEqual(a['source_payload_bits'],'not_applicable')
        self.assertEqual(a['body_mother_bits'],0)
        r['N_header']=68
        with self.assertRaises(RuntimeError):resource_ledger(r)
