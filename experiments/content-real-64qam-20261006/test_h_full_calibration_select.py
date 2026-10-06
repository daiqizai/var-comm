"""Synthetic calibration only. No image model, packet decoder or science run."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
for p in (HERE,HERE.parent,HERE.parent/'payload',HERE.parent/'phy_codec'):sys.path.insert(0,str(p))
import h_full_calibration_select as s
import test_h_full_payload_cpu as fixtures


def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a=fixtures.fixture()
        for group in (a['shortlist']['whole_candidates'],a['shortlist']['partial_candidates'],a['selected']['selected_candidates'],a['partial_reference']['partial_candidates']):
            for c in group:c['mean_attempted_tx_probability_scales']=float(c['target_m'] if c['arm'].endswith('-A') else 0)
        a['protocol'].update(status='FROZEN_BEFORE_DATA',main_snrs_db=[13,19],
            population={'initial_and_full_cal_noise_seeds':[6101,6102,6103],'development_noise_seeds':[6201,6202,6203]},
            calibration={'no_development_selection':True},matched_attribution={'m':7,'K':0,'nominal_rate':'1/2','arms':list(s.ARMS)},
            development={'maximum_packet_calls':13200})
        values=dict(shortlist=a['shortlist'],selected_whole=a['selected'],partial_reference=a['partial_reference'],catalogue=a['catalogue'],
            calibration_registration={'stage':'m1_calibration','calibration_or_development':'m1_calibration','source_ids':a['source_ids']},
            protocol=a['protocol'],interpretation={'schema':'H_PRESCREEN_INTERPRETATION_V1','status':'FROZEN_BEFORE_COARSE_DATA','original_protocol_changed':False})
        cls.artifacts={k:{'path':'/frozen/'+k+'.json','bytes':encoded(v)} for k,v in values.items()}
        cls.schedule=s.physical.schedule(a['selected'],a['partial_reference'],a['shortlist'],a['catalogue'],a['source_ids'])
        cls.rows=[]
        for entry in cls.schedule:
            slot,c=entry['full_slot'],entry['candidate']
            for i,sid in enumerate(a['source_ids']):
                for n in s.SEEDS:
                    # Partial first profile has higher mean PSNR but worse mean MSE.
                    noise=n-6101;score=([10,40,40] if slot in (8,11) else [29,29,29] if slot in (9,12) else [20,20,20])[noise] if slot>=8 else 17+slot
                    state='RAW_SOURCE_DECODED';image='a'*64;view='b'*64
                    rx=dict(status='H_ACTUAL_RX_RECONSTRUCTION_COMPLETE',source_status=state,gray=False,
                        source_decode_complete=True,new_packet_decodes=0,target_image_used_for_reconstruction=False,
                        truth_correction=False,cached_clean_image_used=False,image_sha256=image,receiver_view_sha256=view,
                        received_m=6,received_K=0,received_mode='raw',received_profile_id=0,
                        body_parser_accepted=True,header_accepted=True,arithmetic_canonical_attempted=False)
                    cls.rows.append(dict(**{k:c[k] for k in ('candidate_id','arm','target_m','K','q','nominal_rate','snr_db','slot')},
                        source_index=i,source_id=sid,noise_seed=n,full_slot=slot,phase=entry['phase'],noise_stage=s.physical.NOISE_STAGE,
                        public_frame_counter=s.physical.frame_counter(slot,i,n),psnr_db=score,mse=10**(-score/10),source_status=state,gray=False,
                        image_sha256=image,receiver_view_sha256=view,image_archive='/render/images.npz',image_key='image0',
                        received_m=6,received_K=0,received_mode='raw',received_profile_id=0,rx_summary=rx))
        cls.receipt=dict(status='H_FULL1000_RX_COMPLETE',source_count=1000,source_ids=a['source_ids'],frame_count=42000,
            phase_frame_counts={'whole_calibration':24000,'partial_calibration':18000},images_scored=True,source_decode_complete=True,
            arithmetic_source_decode_complete=True,new_packet_decodes=0,policy_selection=False,development_used=False,holdout_used=False,
            registration_sha256='c'*64,input_bindings={v['path']:s.sha(v['bytes']) for v in cls.artifacts.values()},
            outputs={'/render/images.npz':'d'*64})

    def execute(self,rows=None,artifacts=None,receipt=None):
        metrics=encoded(self.rows if rows is None else rows);done=copy.deepcopy(self.receipt if receipt is None else receipt)
        done['outputs']['/render/frame_metrics.json']=s.sha(metrics)
        return s.finalize_calibration(self.artifacts if artifacts is None else artifacts,metrics,frame_metrics_path='/render/frame_metrics.json',render_completion=done)

    def test_full_grid_partial_uses_mean_psnr_whole_remains_exact_initial_winners(self):
        result=self.execute();old=s.decode(self.artifacts['selected_whole']['bytes'])['selected_candidates']
        self.assertEqual(result['whole_candidates'],old);self.assertFalse(result['whole_policy_reselected'])
        self.assertEqual([r['full_slot'] for r in result['selected_partial']],[8,11])
        self.assertEqual([r['mean_per_source_psnr_db'] for r in result['selected_partial']],[30,30])
        summaries={r['full_slot']:r for r in result['all_candidate_summaries']}
        self.assertGreater(summaries[8]['mean_overall_mse'],summaries[9]['mean_overall_mse'])
        self.assertEqual(len(result['per_source_evidence']),14000)
        self.assertEqual(result['development_plan']['maximum_header_plus_body_calls'],22*100*3*2)
        self.assertFalse(result['development_used']);self.assertFalse(result['H_full_delivery_claimed'])

    def test_gray_frames_are_included_and_cannot_be_silently_relabelled(self):
        rows=list(self.rows)
        for index,row in enumerate(rows):
            if row['full_slot']==8:
                r=copy.deepcopy(row);r.update(source_status='WIRE_REJECT_GRAY',gray=True,psnr_db=10,mse=.1)
                r['rx_summary'].update(source_status=r['source_status'],gray=True,header_accepted=False,body_parser_accepted=False)
                rows[index]=r
        result=self.execute(rows)
        self.assertEqual(result['selected_partial'][0]['full_slot'],9)
        self.assertEqual(result['all_candidate_summaries'][8]['final_receiver_status_counts']['WIRE_REJECT_GRAY'],3000)
        rows[0]=dict(rows[0],gray=True)
        with self.assertRaisesRegex(ValueError,'receiver state'):self.execute(rows)

    def test_original_epsilon_tie_uses_m_K_then_lexical_and_exact_rational_rate(self):
        cs={i:dict(mean_attempted_tx_probability_scales=0,q=6,nominal_rate='1/2',target_m=7,K=81-i,policy_key=str(i)*64) for i in range(3)}
        summaries=[dict(full_slot=i,candidate_id=str(i),mean_per_source_psnr_db=30-(5e-13 if i==2 else 0)) for i in range(3)]
        winner,tied=s.choose_partial(summaries,cs);self.assertEqual(winner['full_slot'],2);self.assertEqual(len(tied),3)
        summaries[2]['mean_per_source_psnr_db']=30-2e-12
        self.assertEqual(s.choose_partial(summaries,cs)[0]['full_slot'],1)
        cs[0]['nominal_rate']='2/3';cs[1]['nominal_rate']='3/4'
        self.assertEqual(s.choose_partial(summaries,cs)[0]['full_slot'],0)
        for c in cs.values():c.update(nominal_rate='1/2',target_m=7,K=80)
        for row in summaries:row['mean_per_source_psnr_db']=0.0
        self.assertEqual(s.choose_partial(summaries,cs)[0]['full_slot'],0)
        cs[0]['policy_key']='f'*64
        self.assertEqual(s.choose_partial(summaries,cs)[0]['full_slot'],1)

    def test_missing_duplicate_and_development_noise_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'frame count'):self.execute(self.rows[:-1])
        rows=list(self.rows);rows[-1]=rows[0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.execute(rows)
        rows=list(self.rows);rows[0]=dict(rows[0],noise_seed=6201)
        with self.assertRaisesRegex(ValueError,'noise identity'):self.execute(rows)

    def test_bound_bytes_actual_complete_state_and_original_population_required(self):
        artifacts=copy.deepcopy(self.artifacts);artifacts['selected_whole']['bytes']+=b' '
        with self.assertRaisesRegex(ValueError,'Input differs'):self.execute(artifacts=artifacts)
        done=copy.deepcopy(self.receipt);done['arithmetic_source_decode_complete']=False
        with self.assertRaisesRegex(ValueError,'actual-RX receipt'):self.execute(receipt=done)
        rows=list(self.rows);rows[0]=dict(rows[0],psnr_db=0)
        with self.assertRaisesRegex(ValueError,'PSNR/MSE'):self.execute(rows)
        rows=list(self.rows);rows[0]=copy.deepcopy(rows[0]);rows[0]['rx_summary']['truth_correction']=True
        with self.assertRaisesRegex(ValueError,'source truth'):self.execute(rows)

    def test_fixed_m_is_protocol_defined_and_unavailable_bucket_never_substituted(self):
        cat=s.decode(self.artifacts['catalogue']['bytes']);p=s.decode(self.artifacts['protocol']['bytes'])
        controls=s.fixed_controls(cat,p)
        self.assertEqual(len(controls),8)
        self.assertTrue(all((r['target_m'],r['K'],r['nominal_rate'],r['raw_source_bits'])==(7,0,'1/2',1860) for r in controls))
        self.assertTrue(all(r['status']=='EXISTING_FIXED_POINT_FEASIBLE_NOT_EVALUATED' for r in controls))
        bucket=next(b for b in cat['buckets'] if b['q']==6 and b['nominal_rate']=='1/2');bucket['admission']='UNAVAILABLE'
        controls=s.fixed_controls(cat,p)
        self.assertEqual(sum(r['status']=='NOT_FEASIBLE' for r in controls),4)
        self.assertTrue(all(r['target_m']==7 and r['nominal_rate']=='1/2' for r in controls))
        self.assertTrue(all(not r['public_profiles'] for r in controls if r['status']=='NOT_FEASIBLE'))


if __name__=='__main__':unittest.main()
