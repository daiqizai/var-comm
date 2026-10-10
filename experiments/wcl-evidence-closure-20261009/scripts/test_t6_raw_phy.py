"""N2048 actual-metadata and synthetic dispatch contracts; no PHY execution."""
import copy,inspect,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import t2_pilot as h
import t6_raw_phy as p
import t6_calibration_plan as plan

W=Path(__file__).resolve().parents[3]
SNAP=W/'.research/wcl_evidence_closure_20261009/remote_snapshot'
CAT=W/'.research/wcl_evidence_closure_20261009/t6_metadata/actual_layouts/candidate_catalogue.json'


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core=SNAP/'main_action_space.py';wire=SNAP/'main_raw_keep_receiver.py'
        h.load(core,'main_action_space',{str(core):h.sha(core)})
        cls.original=h.load(wire,'_t6_test_original',{str(wire):h.sha(wire)})
        cls.cat=h.read(CAT)

    def test_all521_actual_catalogue_entries_validate_without_backend(self):
        cat=p.PublicCatalogue(self.cat,self.original);cat.verify()
        self.assertEqual(len(cat.header_map()),521)
        self.assertEqual(sum(x['K']==0 for x in cat._profiles),131)
        self.assertEqual(sum(x['m']==10 for x in cat._profiles),3)

    def test_all_actual_raw_endpoints_roundtrip_including_m10(self):
        cat=p.PublicCatalogue(self.cat,self.original);tokens=np.arange(680,dtype=np.int64)
        scales=[tokens[self.original.PREFIX[i]:self.original.PREFIX[i+1]]for i in range(10)]
        for row in cat._profiles:
            bits=self.original.serialize_raw(scales,row);state=self.original.parse_raw(bits,row['m'],row['K'])
            received=np.asarray([v for scale in state['prefix']for v in scale]+state['partial_values'])
            self.assertTrue(np.array_equal(received,tokens[:row['token_count']]))

    def test_catalogue_rejects_N1024_and_body_metadata_changes(self):
        cat=copy.deepcopy(self.cat);cat['profiles'][0]['N']=1024;cat['catalogue_sha256']=h.digest(cat['profiles'])
        with self.assertRaisesRegex(RuntimeError,'N2048'):p.PublicCatalogue(cat,self.original)
        cat=copy.deepcopy(self.cat);cat['profiles'][0]['idle_symbols']+=1;cat['catalogue_sha256']=h.digest(cat['profiles'])
        with self.assertRaisesRegex(RuntimeError,'accounting'):p.PublicCatalogue(cat,self.original)

    def test_keep_uses_failed_crc_actual_hard_bits_not_transmitted_tokens(self):
        cat=p.PublicCatalogue(self.cat,self.original);row=cat.entry(0);g=row['groups'][0]
        bits=np.zeros(g['source_bits'],np.uint8);decoded=np.concatenate((bits,np.ones(16,np.uint8)))
        header=dict(header_ok=True,header_crc_ok=True,header_fields_legal=True,profile_id=0,score=0.)
        body=dict(profile_id=0,phy_key=g['phy_key'],hard_payload=bits.tolist(),decoded_bits=decoded.tolist(),crc_accept=False)
        state=self.original.present_actual(header,body,cat)
        self.assertFalse(state['gray']);self.assertFalse(state['body_crc_accept']);self.assertEqual(state['receiver_state']['m'],row['m'])

    def test_received_header_selects_body_shape_and_profile(self):
        cat=p.PublicCatalogue(self.cat,self.original);rt=Mock();rt.catalogue=cat;rt.original=self.original
        row=cat.entry(12);g=row['groups'][0];header=dict(header_ok=True,header_crc_ok=True,header_fields_legal=True,profile_id=12,score=0.)
        body=dict(profile_id=12,phy_key=g['phy_key'],hard_payload=[0]*g['source_bits'],decoded_bits=[0]*g['information_bits'],crc_accept=False)
        with patch.object(p,'receive_header',return_value=header),patch.object(p,'receive_body',return_value=body)as receive:
            result=p.receive_frame(rt,np.zeros((2048,2),np.float64),10.,5,Mock(),'synthetic','synthetic')
        self.assertEqual(receive.call_args.args[2],row);self.assertEqual(receive.call_args.args[1].shape,(g['symbols'],2))
        self.assertEqual(result['logical_packet_calls'],2);self.assertEqual(result['rx_profile_id'],12)
        self.assertNotIn('source_id',inspect.signature(p.receive_frame).parameters)

    def test_header_rejection_does_not_decode_body(self):
        cat=p.PublicCatalogue(self.cat,self.original);rt=Mock();rt.catalogue=cat;rt.original=self.original
        header=dict(header_ok=False,header_crc_ok=False,header_fields_legal=False,profile_id=None,score=0.)
        with patch.object(p,'receive_header',return_value=header),patch.object(p,'receive_body')as body:
            result=p.receive_frame(rt,np.zeros((2048,2),np.float64),4.,1,Mock(),'synthetic','synthetic')
        body.assert_not_called();self.assertTrue(result['gray']);self.assertEqual(result['logical_packet_calls'],1)

    def test_new2048_noise_and_uint13_capacity_are_explicit(self):
        a=p.standard_noise('same-source',9201);self.assertEqual(a.shape,(2048,2));self.assertEqual(a.dtype,np.float64)
        self.assertTrue(np.array_equal(a,p.standard_noise('same-source',9201)))
        self.assertFalse(np.array_equal(a,p.standard_noise('same-source',9202)))
        query=plan.entropy_candidates();self.assertEqual(len(query),48)
        c=next(x for x in query if x['target_m']==10 and x['q']==6 and x['nominal_rate']=='5/6')
        self.assertEqual(c['k'],9900);self.assertEqual(c['source_capacity'],8191)

    def test_finite_raw_full_union_retains_whole_and_m10_permissions(self):
        pilot=[]
        for snr in plan.SNRS:
            pilot.extend(dict(snr_db=snr,candidate_id=p['candidate_id'],source_count=100,noise_count=1,
                dinov2_vitl14_cosine=str((p['K']>0)*.5+1/(p['profile_id']+1)))for p in self.cat['profiles'])
        full=plan.raw_schedule(self.cat,'full',pilot)
        for snr in plan.SNRS:
            rows=[x for x in full if x['snr_db']==snr]
            self.assertEqual(sum(x['m']==10 for x in rows),3)
            self.assertEqual(sum(x['K']==0 and'full_budget'in x['profile']['allocation_modes']for x in rows),9)
            self.assertTrue(all(x['eligible_PARTIAL']for x in rows))
            self.assertLessEqual(len(rows),22)
        self.assertEqual(len(plan.raw_schedule(self.cat,'pilot')),1563)


if __name__=='__main__':unittest.main()
