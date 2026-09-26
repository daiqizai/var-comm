import unittest
from tools.publish_c_initial_milestone import validate_rows, extension_evidence

class CalibrationMilestoneTests(unittest.TestCase):
    def row(self):
        return dict(method='V',source_index='0',image_id='source',snr_db='1',seed='4101',
                    mse='0.01',lpips_alex='0.2',normalized_latent='0.5',utility='0.035',
                    U_image='0.03',header_ok='1',body_crc_ok='0')
    def validate(self, rows):
        validate_rows(rows,['V'],[{'image_id':'source'}],[1],[4101])
    def test_grid_rejects_duplicate_and_missing(self):
        row=self.row()
        self.validate([row])
        for rows in ([],[row,row]):
            with self.assertRaises(ValueError): self.validate(rows)
    def test_failed_header_masks_latent_but_retains_image_loss(self):
        row=self.row();row.update(header_ok='0',utility='0.03')
        self.validate([row])
        row['utility']='0.035'
        with self.assertRaises(ValueError): self.validate([row])
    def test_selection_extension_requires_two_intervals_in_same_arm(self):
        curves=[dict(method='V',utility=x) for x in [1,.99,.99]]
        curves += [dict(method='P',utility=x) for x in [1,1,.99]]
        self.assertFalse(any(v['supports_extension'] for v in extension_evidence(curves,['V','P']).values()))
        curves[2]['utility']=.98
        self.assertTrue(extension_evidence(curves,['V','P'])['V']['supports_extension'])
    def test_source_and_pair_failures_rejected(self):
        row=self.row();row['image_id']='other'
        with self.assertRaises(ValueError): self.validate([row])
        row=self.row();other=dict(row,method='P',body_crc_ok='1')
        with self.assertRaises(ValueError):
            validate_rows([row,other],['V','P'],[{'image_id':'source'}],[1],[4101])
