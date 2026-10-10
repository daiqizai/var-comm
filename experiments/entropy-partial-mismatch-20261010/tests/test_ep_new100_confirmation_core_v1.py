"""CPU-only synthetic contracts. No real source pool, pixel or model is opened."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ep_new100_confirmation_core_v1 as c


def desc(name,digest='a'*64):return dict(path='/synthetic/'+name,sha256=digest)


def fixture():
    ids=['n00000000/ILSVRC2012_val_'+str(i+1).zfill(8)+'_n00000000' for i in range(100)]
    calibration=['synthetic_calibration_'+str(i) for i in range(1000)]
    p={name:desc(name,digest) for name,digest in [('raw_plan',c.RAW_PLAN_SHA),('raw_freeze',c.RAW_POLICY_SHA),('whole_policy',c.WHOLE_POLICY_SHA)]}
    rawplan=dict(schema='MAIN_RAW64_UNIFIED500_HOLDOUT_PLAN_V1',source_count=500,schedule=[])
    rawfreeze=dict(status='POLICIES_FROZEN_ON_CALIBRATION1000',holdout_used=False,objective='dinov2_vitl14_cosine',winners=[])
    whole=dict(status='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1',source_count=1000,holdout_used_for_selection=False,
        calibration_source_ids=calibration,policies={'EC_VAR_WHOLE':{}})
    winners=dict(status='FULL1000_CALIBRATION_STRATEGY_FROZEN',final_strategy_frozen=True,holdout_used_for_selection=False,
        calibration_source_ids=calibration,noise_seeds=[4101,4102,4103],snrs={})
    candidate=next(x for x in c.ep.candidates() if x['target_K']==0)
    for s in c.SNRS:
        for family in ['WHOLE','PARTIAL']:
            cid='SYNTHETIC_'+family+'_'+str(s)
            rawplan['schedule'].append(dict(snr_db=s,families=[family],candidate_id=cid,profile_id=0,wire_key='SYNTHETIC',m=4,K=0,modulation='q2'))
            rawfreeze['winners'].append(dict(snr_db=s,family=family,candidate_id=cid))
        whole['policies']['EC_VAR_WHOLE'][str(s)]=dict(candidate_id=candidate['candidate_id'])
        winners['snrs'][str(s)]=dict(winner=candidate['candidate_id'],candidate=copy.deepcopy(candidate),ranking=[candidate['candidate_id']],
            original_final_whole_winner=candidate['candidate_id'],whole_winner_retained=True,source_count=1000,noise_count=3,logical_frames=3000)
    closure=dict(schema='EP_NEW100_FULL_CALIBRATION_CLOSURE_PROOF_V1',complete_grid_verified=True,original_entropy_policy=p['whole_policy'])
    docs={'raw_plan':rawplan,'raw_freeze':rawfreeze,'whole_policy':whole}
    read=lambda d:docs[d['path'].rsplit('/',1)[-1]]
    bundle=c.policies(p,winners,closure,read)
    selection=dict(schema='EP_NEW100_METADATA_SELECTION_V1',status='EP_SOURCE_IDS_FIXED_NO_PIXEL_ACCESS',source_count=100,N=1024,
        SNRs=list(c.SNRS),noise_seeds=list(c.SEEDS),methods=list(c.METHODS),source_reselection_allowed=False,registry=desc('registry'),
        records=[dict(source_index=i,source_id=sid) for i,sid in enumerate(ids)],source_ids=ids)
    gate=dict(schema='EP_NEW100_CONTENT_GATE_V1',status='EP_NEW100_COMPLETE_CONTENT_DEDUP_PASS',selection=desc('selection'),
        registry=selection['registry'],source_count=100,conflicts=[],Encoder_calls_allowed=True,source_reselection_allowed=False)
    return dict(pins=p,docs=docs,read=read,winners=winners,closure=closure,bundle=bundle,selection=selection,gate=gate)


def grid(f):return c.frame_grid(f['selection'],desc('selection'),f['gate'],f['bundle'])


class ConfirmationTests(unittest.TestCase):
    def test_original_budgets_and_600_RX_cap_preserved(self):
        caps=c.scientific_caps();self.assertEqual(caps['packet_decodes'],7200)
        self.assertEqual(caps['source']['encoder'],100);self.assertEqual(caps['source']['source_tx'],100)
        self.assertEqual(caps['source']['source_rx'],0);self.assertEqual(caps['visual']['source_rx'],600)
        self.assertEqual(caps['visual']['prior_scale'],42000)

    def test_receiver_catalogues_and_failure_semantics_remain_distinct(self):
        self.assertEqual([c.method_contract(m)['catalogue_count'] for m in c.METHODS],[433,433,144,360])
        self.assertIn('KEEP',c.method_contract('RAW_PARTIAL')['body_failure'])
        self.assertEqual(c.method_contract('EC_VAR_WHOLE')['source_decoder'],'original_SourceCodec')
        self.assertEqual(c.method_contract('EC_VAR_PARTIAL')['body_failure'],'constant_RGB_0.5')

    def test_whole_winner_may_be_allowed_partial_policy(self):
        f=fixture()
        for s in c.SNRS:
            self.assertEqual(f['bundle']['policies']['EC_VAR_PARTIAL'][str(s)]['target_K'],0)

    def test_partial_winner_cannot_drop_original_whole_from_finalists(self):
        f=fixture();f['winners']['snrs']['4']['ranking']=['other']
        with self.assertRaises(ValueError):c.policies(f['pins'],f['winners'],f['closure'],f['read'])

    def test_original_policy_digest_cannot_be_relabelled(self):
        f=fixture();f['pins']['whole_policy']['sha256']='0'*64
        with self.assertRaises(ValueError):c.policies(f['pins'],f['winners'],f['closure'],f['read'])

    def test_partial_calibration_is_not_frozen(self):
        f=fixture();f['winners']['snrs']['10']['source_count']=100
        with self.assertRaises(ValueError):c.policies(f['pins'],f['winners'],f['closure'],f['read'])

    def test_failed_content_gate_cannot_build_frames(self):
        f=fixture();f['gate']['status']='EP_NEW100_CONTENT_DUPLICATE_STOP'
        with self.assertRaises(ValueError):grid(f)

    def test_content_gate_pins_cannot_switch_population(self):
        f=fixture();f['gate']['registry']=desc('foreign')
        with self.assertRaises(ValueError):grid(f)

    def test_wrong_seeds_or_N2048_cannot_enter(self):
        for field,value in [('noise_seeds',[9201,9202,9203]),('N',2048)]:
            f=fixture();f['selection'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):grid(f)

    def test_complete_grid_uses_same_sources_and_counter_for_all_arms(self):
        f=fixture();rows=grid(f);c.complete_grid(rows,rows)
        self.assertEqual(len(rows),3600)
        self.assertEqual({c.counter(i,s,n) for i in range(100) for s in c.SNRS for n in c.SEEDS},set(range(900)))
        for i in range(0,len(rows),4):
            self.assertEqual(len({x['source_id'] for x in rows[i:i+4]}),1)
            self.assertEqual(len({x['public_frame_counter'] for x in rows[i:i+4]}),1)

    def test_duplicate_or_reordered_rows_stop(self):
        expected=grid(fixture());rows=copy.deepcopy(expected);rows[1]=rows[0]
        with self.assertRaises(ValueError):c.complete_grid(rows,expected)
        rows=copy.deepcopy(expected);rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(ValueError):c.complete_grid(rows,expected)

    def test_missing_failure_workpoint_is_not_dropped(self):
        rows=grid(fixture())
        with self.assertRaises(ValueError):c.complete_grid(rows[:-1],rows)

    def test_method_specific_source_picking_is_rejected(self):
        expected=grid(fixture());rows=copy.deepcopy(expected);rows[1]['source_id']='picked_other_source'
        with self.assertRaises(ValueError):c.complete_grid(rows,expected)

    def test_source_average_and_pairing_retain_signs_and_zeros(self):
        expected=grid(fixture());rows=copy.deepcopy(expected)
        for r in rows:
            partial=r['method']=='RAW_PARTIAL'
            r.update(psnr_db=20.+(1 if partial else 0),lpips_alex=.4-(.1 if partial else 0),
                dinov2_vitl14_cosine=.6,convnext_top1_source_prediction=int(partial))
        means,pairs=c.source_vectors(rows,expected)
        self.assertEqual(len(means),48);self.assertEqual(len(pairs),24)
        self.assertEqual(pairs['RAW_PARTIAL','RAW_WHOLE',4,'psnr_db'],[1.]*100)
        self.assertAlmostEqual(pairs['RAW_PARTIAL','RAW_WHOLE',4,'lpips_alex'][0],-.1)
        self.assertEqual(pairs['RAW_PARTIAL','RAW_WHOLE',4,'convnext_top1_source_prediction'],[1.]*100)
        self.assertEqual(pairs['EC_VAR_PARTIAL','EC_VAR_WHOLE',19,'psnr_db'],[0.]*100)

    def test_metric_nan_or_accuracy_percentage_is_rejected(self):
        expected=grid(fixture());rows=copy.deepcopy(expected)
        for r in rows:r.update(psnr_db=20.,lpips_alex=.4,dinov2_vitl14_cosine=.6,convnext_top1_source_prediction=1)
        rows[0]['lpips_alex']=float('nan')
        with self.assertRaises(ValueError):c.source_vectors(rows,expected)
        rows[0]['lpips_alex']=.4;rows[0]['convnext_top1_source_prediction']=100
        with self.assertRaises(ValueError):c.source_vectors(rows,expected)

    def test_RX_cap_overrun_or_unresolved_cannot_close(self):
        caps=c.scientific_caps()['visual'];values=dict.fromkeys(caps,0);values['source_rx']=601
        closed=dict(caps=caps,reserved=values,completed=values,unresolved=0)
        with self.assertRaises(ValueError):c.check_ledger(closed,caps)
        values['source_rx']=600;closed['unresolved']=1
        with self.assertRaises(ValueError):c.check_ledger(closed,caps)

    def test_catalogue_and_observation_identity_control_reuse(self):
        keys=('tokens_sha256','payload_sha256','waveform_sha256','standard_noise_sha256','observation_sha256',
            'receive_catalogue_sha256','receiver_provider_sha256','PHY_runtime_identity_sha256')
        evidence=dict.fromkeys(keys,'a'*64);evidence.update(source_id='synthetic',snr_db=4,noise_seed=9301,
            public_frame_counter=0,transmitted_profile={'profile_id':72},wire_session='original')
        first=c.reuse_key(evidence);changed=copy.deepcopy(evidence);changed['receive_catalogue_sha256']='b'*64
        self.assertNotEqual(first,c.reuse_key(changed));changed=copy.deepcopy(evidence);changed['observation_sha256']='b'*64
        self.assertNotEqual(first,c.reuse_key(changed))
        evidence['CRC_success']=True
        with self.assertRaises(ValueError):c.reuse_key(evidence)

    def test_preparation_has_no_execution_or_population_selection_claim(self):
        doc=c.preparation_summary()
        self.assertFalse(doc['execution_ready']);self.assertFalse(doc['automatic_successor'])
        for key in ['IDs_selected','pool_hash_ranks_computed','new_pixel_reads','model_calls','channel_calls','bootstrap_calls']:
            self.assertEqual(doc[key],0)

    def test_not_actually_waited_full_calibration_is_rejected(self):
        pins={n:desc(n) for n in ['completion','owner_actual_wait','request']}
        done=dict(schema='H800_EP_ORIGINAL1000_FULL_METRICS_V2',status='PASS_H800_ORIGINAL1000_FULL_FOUR_METRICS_AND_CALIBRATION',
            request_sha256='a'*64,actual_wait={'success':True},actual_children_waited=True,worker_exit_codes=[0])
        values={'completion':done,'owner_actual_wait':{'actual_wait':False,'returncode':0}}
        with self.assertRaises(ValueError):
            c.verify_full_calibration(pins,lambda p:values[p['path'].rsplit('/',1)[-1]],lambda _: {},lambda *a:None)

    def test_full_calibration_uses_complete_existing_rows_and_unchanged_recompute(self):
        f=fixture();winner=f['winners'];winner['new100_admitted']=False
        pins={n:desc(n) for n in ['completion','owner_actual_wait','request']}
        caps=dict(model_constructions=3,reference_preparations=1000,image_scores=36000,
            dinov2_vitl14_reference=1000,dinov2_vitl14_reconstruction=36000,convnext_reference=1000,
            convnext_reconstruction=36000,lpips_pair=36000,lpips_alexnet_backbone_forward=72000)
        complete=dict(model_constructions=3,reference_preparations=1,image_scores=1,
            dinov2_vitl14_reference=1,dinov2_vitl14_reconstruction=1,convnext_reference=1,
            convnext_reconstruction=1,lpips_pair=1,lpips_alexnet_backbone_forward=2)
        request=dict(caps=caps,records=['SYNTHETIC_RECORDS'],finalists={},policy=f['pins']['whole_policy'],logical_events=[None]*9000)
        done=dict(schema='H800_EP_ORIGINAL1000_FULL_METRICS_V2',status='PASS_H800_ORIGINAL1000_FULL_FOUR_METRICS_AND_CALIBRATION',
            request_sha256='a'*64,actual_wait={'success':True},actual_children_waited=True,worker_exit_codes=[0],worker_completion=desc('worker'))
        result=dict(counts=dict(caps=caps,reserved=complete,completed=complete,unresolved=0),
            reuse=dict(new_actual_pairs=1,same_run=8999,closed_pilot=0),actual_reference_preparations=1,
            metric_rows=desc('rows'),full_winners=desc('winner'),logical_frames=9000)
        worker=dict(schema=done['schema'],status=done['status'],request_sha256='a'*64,population='original_calibration_full1000',
            final_strategy_frozen=True,new100_admitted=False,results=result)
        docs=dict(completion=done,owner_actual_wait=dict(actual_wait=True,returncode=0),worker=worker,rows=[None]*9000,winner=winner,
            whole_policy=f['docs']['whole_policy'])
        calls=[]
        def recompute(rows,records,finalists,policy):
            calls.append((len(rows),records,finalists,policy));return copy.deepcopy(winner)
        proof,actual=c.verify_full_calibration(pins,lambda p:docs[p['path'].rsplit('/',1)[-1]],lambda _:request,recompute)
        self.assertEqual(calls[0][0],9000);self.assertTrue(proof['complete_grid_verified']);self.assertEqual(actual,winner)
        result['counts']['unresolved']=1
        with self.assertRaises(ValueError):c.verify_full_calibration(pins,lambda p:docs[p['path'].rsplit('/',1)[-1]],lambda _:request,recompute)


if __name__=='__main__':unittest.main()
