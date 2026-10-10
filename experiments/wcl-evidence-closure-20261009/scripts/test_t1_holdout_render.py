"""Synthetic holdout consumer contracts; zero scientific model/PHY calls."""
import contextlib, copy, tempfile, time, unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import t1_holdout_render as h
import t1_phy as phy
from t1_source_asset_schema import build_record,model_id


def fixture(tmp):
    tokens=np.arange(680,dtype=np.int64);arrays={'tokens':tokens};tx={};sel={}
    candidate=copy.deepcopy(h.candidates()[0]);models={f:model_id(f,'a'*64)for f in h.FAMILIES}
    for family,prefix in zip(h.FAMILIES,('static','var')):
        tx[family]={}
        for m in range(4,8):
            bits=np.resize(np.array([0,1],np.uint8),m*8)
            arrays[f'tx_{prefix}_{m}']=bits
            tx[family][str(m)]=dict(arithmetic_bits=len(bits),bits_key=f'tx_{prefix}_{m}')
        arrays[f'{prefix}_m7_bits']=arrays[f'tx_{prefix}_7'].copy()
        arrays[f'{prefix}_m7_received_tokens']=tokens[:h.OFFSETS[7]].copy()
        choice=h.choose_arithmetic({int(m):e['arithmetic_bits']for m,e in tx[family].items()},candidate,minimum_m=4)
        sel[family]={str(s):dict(candidate_id=candidate['candidate_id'],target_m=7,q=candidate['q'],nominal_rate=candidate['nominal_rate'],**choice)for s in h.SNRS}
    archive=Path(tmp)/'source.npz';np.savez(archive,**arrays)
    sr=build_record(source_index=0,source_id='s0',tokens=tokens,preprocessing_id='pixel',source_assets_checkpoint={'path':'old','sha256':'old'},
        archive=archive,arrays=arrays,static_completion_sha='a'*64,origins={f:'SYNTHETIC'for f in h.FAMILIES},upstream_evidence={f:{}for f in h.FAMILIES})
    sr.update(source_role='holdout',post_hoc_supplement=True,holdout_used_for_selection=False,freeze={'path':'freeze','sha256':'freeze'},
        tx_prefixes=tx,selected_by_snr=sel)
    return sr,arrays,candidate,models


def packet(family,m=7):
    return dict(TX_profile_id=999,actual_RX=dict(status='PAYLOAD_PARSED',header=dict(header_ok=True),
        rx_profile=dict(profile_id=1,family=family,m=m),body=dict(crc_accepted=True,parser_accepted=True,payload=np.resize([0,1],m*8).tolist())))


class Contracts(unittest.TestCase):
    def test_selected_prefix_uses_all_TX_lengths_but_only_selected_RX_stream(self):
        with tempfile.TemporaryDirectory()as tmp:
            sr,a,c,_=fixture(tmp)
            self.assertEqual(set(sr['streams'][h.FAMILIES[0]]),{'7'})
            m,bits,selection=h.selected_stream(sr,a,h.FAMILIES[0],4,c)
            self.assertEqual(m,7);self.assertEqual(len(bits),56);self.assertEqual(selection['actual_m'],7)
            sr['selected_by_snr'][h.FAMILIES[0]]['4']['actual_m']=6
            with self.assertRaisesRegex(ValueError,'selected prefix changed'):h.selected_stream(sr,a,h.FAMILIES[0],4,c)

    def test_selected_bit_mismatch_is_not_channel_failure(self):
        with tempfile.TemporaryDirectory()as tmp:
            sr,a,c,_=fixture(tmp);a['tx_static_7'][0]=1
            with self.assertRaisesRegex(ValueError,'TX stream'):h.selected_stream(sr,a,h.FAMILIES[0],4,c)

    def test_wrong_received_header_uses_actual_family_and_m(self):
        with tempfile.TemporaryDirectory()as tmp:
            sr,a,_,models=fixture(tmp);p=packet(h.FAMILIES[1]);p['TX_family']=h.FAMILIES[0]
            state=h.cached_recovery(sr,a,p,models);self.assertEqual(state['m'],7)
            p['actual_RX']['rx_profile']['m']=6
            self.assertIsNone(h.cached_recovery(sr,a,p,models))

    def test_actual_bit_error_is_cache_miss_and_reject_is_gray(self):
        with tempfile.TemporaryDirectory()as tmp:
            sr,a,_,models=fixture(tmp);p=packet(h.FAMILIES[0]);p['actual_RX']['body']['payload'][0]=1
            self.assertIsNone(h.cached_recovery(sr,a,p,models))
            for reject in('header','crc','parser'):
                p=packet(h.FAMILIES[1])
                if reject=='header':p['actual_RX']['header']['header_ok']=False
                elif reject=='crc':p['actual_RX']['body']['crc_accepted']=False
                else:p['actual_RX']['body']['parser_accepted']=False
                self.assertEqual(h.cached_recovery(sr,a,p,models)['kind'],'gray')

    def test_diagnostic_truth_cannot_mutate_decoded_state(self):
        state=h.cal.receiver_state(np.ones(h.OFFSETS[7],np.int64),7);before=copy.deepcopy(state)
        d=h.diagnostic_token_errors(state,np.zeros(680,np.int64))
        self.assertEqual(state,before);self.assertEqual(d['token_error_count'],h.OFFSETS[7]);self.assertFalse(d['received_prefix_matches_source'])
        self.assertIsNone(h.diagnostic_token_errors(h.cal.gray_state(),np.zeros(680,np.int64))['received_prefix_matches_source'])

    def test_new_holdout_counter_is_unique_and_in_paid_range(self):
        values=[h.counter(i,s,n)for i in range(500)for s in h.SNRS for n in h.SEEDS]
        self.assertEqual(len(set(values)),4500);self.assertGreaterEqual(min(values),0);self.assertLess(max(values),9000)

    def test_scope_prevents_missing_noise_and_budget_expansion(self):
        c=h.candidates()[0];r=dict(schema=h.SCHEMA,families=h.FAMILIES,snrs=h.SNRS,seeds=h.SEEDS,source_count=500,
            frame_count=9000,packet_cap=18000,records=[dict(source_index=i,source_id=str(i))for i in range(500)],source_ids=[str(i)for i in range(500)],
            policies={f:{str(s):copy.deepcopy(c)for s in h.SNRS}for f in h.FAMILIES},workers=8,max_seconds=86400,post_hoc_supplement=True,
            holdout_used_for_selection=False,training_updates=0)
        h.validate_request(r);r['seeds']=[6201,6202]
        with self.assertRaisesRegex(ValueError,'three-noise'):h.validate_request(r)
        r['seeds']=h.SEEDS;r['packet_cap']=18001
        with self.assertRaisesRegex(ValueError,'Finite'):h.validate_request(r)

    def test_sealed_float_image_checks_storage_and_pixel_hash(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'image.npz';image=np.full((3,256,256),.5,np.float32);np.savez(p,image=image)
            v=dict(image_path=str(p),image_key='image',image_slot=0,image_file_sha256=h.sha(p),image_sha256=h.shared.image_sha(image))
            self.assertTrue(np.array_equal(h.checked_image(v),image));v['image_sha256']='bad'
            with self.assertRaisesRegex(ValueError,'identity'):h.checked_image(v)

    def test_eighteen_frame_cpu_resume_never_repeats_ledger_callbacks(self):
        with tempfile.TemporaryDirectory()as tmp:
            sr,a,c,models=fixture(tmp);out=Path(tmp)/'out';out.mkdir();rh='a'*64
            profiles=[dict(c,family=f,m=7,profile_id=i)for i,f in enumerate(h.FAMILIES)]
            r=dict(out=str(out),root=tmp,workers=1,records=[dict(source_index=0,source_id='s0')],freeze=sr['freeze'],
                policies={f:{str(s):c for s in h.SNRS}for f in h.FAMILIES},source_ids=['s0'],source_model_id=models,
                phy_qualification={'path':'q'},registered_profile_catalogue={'profiles':profiles},packet_cap=18000,
                max_seconds=60,deadline_unix=time.time()+60,stop_files=[])
            rt=Mock();rt.catalogue=r['registered_profile_catalogue'];rt.profiles={str(i):p for i,p in enumerate(profiles)};callbacks=[]
            rt.transmit.side_effect=lambda pid,bits,counter:(np.zeros((1024,2),np.float64),dict(k=956,n=1912,
                header_symbols=68,body_symbols=956,frame_padding_symbols=0,arithmetic_bits=len(bits),
                known_information_padding_bits=956-29-len(bits),header_energy=136.,body_energy=1912.,E_frame=2048.,rho=1.))
            def receive(observed,snr,counter,public,ledger,event,phase):
                self.assertEqual(public,rt.profiles)
                family=profiles[int(event.rsplit('profile',1)[1])]['family'];rx=packet(family)['actual_RX']
                def header():callbacks.append('header');return rx['header']
                def body():callbacks.append('body');return rx['body']
                ledger.call({'event_id':event+'.header'},{'counter':counter},header)
                ledger.call({'event_id':event+'.body'},{'counter':counter},body);return rx
            rt.receive.side_effect=receive
            def ledger(*_):return h.Ledger(out/'packet_ledger.sqlite',rh,18000)
            with patch.object(h,'registered',return_value=(r,rh)),patch.object(h.shared,'lock',return_value=contextlib.nullcontext()),\
                patch.object(h,'load_source',return_value=(sr,a)),patch.object(h,'open_worker_ledger',side_effect=ledger),\
                patch.object(phy,'create_runtime',return_value=rt),patch.dict(h.os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                h.cpu_worker('request',0);self.assertEqual(len(callbacks),36)
                h.cpu_worker('request',0);self.assertEqual(len(callbacks),36)
            saved=h.read(out/'cpu_sources/0000.json');self.assertEqual(len(saved['frames']),18)
            self.assertEqual(len({(x['family'],x['snr_db'],x['noise_seed'])for x in saved['frames']}),18)
            self.assertEqual(h.cal.ledger_snapshot(out/'packet_ledger.sqlite',rh,18000)['total'],36)
            image=np.full((3,256,256),.25,np.float32);ap=Path(tmp)/'old_image.npz';np.savez(ap,images=image[None])
            state=h.cal.receiver_state(a['static_m7_received_tokens'],7);key=h.digest(state)
            old=Path(tmp)/'old.json';h.save(old,dict(source_index=0,source_id='s0',states={key:dict(state=state,
                image_path=str(ap),image_key='images',image_slot=0,image_sha256=h.shared.image_sha(image),image_file_sha256=h.sha(ap),
                reuse='EXACT_ORIGINAL_RECEIVED_STATE_IMAGE',new_VAR_render_calls=0)}))
            env=Path(tmp)/'env.json';h.save(env,dict(visual_config={'visual_lock':'unused'},root=tmp))
            r.update(old_reuse=[h.pin(old)],environment_request=h.pin(env),static_completion={'path':'static'})
            native=Mock();codec=Mock()
            with patch.object(h,'registered',return_value=(r,rh)),patch.object(h,'load_source',return_value=(sr,a)),\
                patch.object(h.shared,'lock',return_value=contextlib.nullcontext()),patch.object(h,'guard'),\
                patch.object(h,'build_native',return_value=(native,Mock())),patch.object(h,'SourceCodec',return_value=codec),\
                patch.object(h.shared,'render_state',return_value=image)as render,patch.dict(h.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                plan=h.plan_render('request');self.assertEqual(plan['new_VAR_source_decode_upper_bound'],0)
                self.assertEqual(plan['new_VAR_render_upper_bound'],0)
                h.gpu_worker('request');self.assertEqual(render.call_count,1);codec.decode.assert_not_called()
                h.gpu_worker('request');self.assertEqual(render.call_count,1);codec.decode.assert_not_called()
            gpu=h.read(out/'gpu_sources/0000.json');self.assertEqual(len(gpu['rows']),18)
            self.assertEqual(gpu['new_VAR_render_calls'],0);self.assertTrue(all(x['image_path']==str(ap)for x in gpu['rows']))
            self.assertTrue(all(x['token_error_count']==0 and x['diagnostic_truth_used_for_output']is False for x in gpu['rows']))
            self.assertEqual(set(x['image_key']for x in gpu['rows']),{'images'})

    def test_receiver_decode_resume_and_interrupt_are_durable(self):
        with tempfile.TemporaryDirectory()as tmp:
            sr,a,_,models=fixture(tmp);p=packet(h.FAMILIES[1]);p['actual_RX']['body']['payload'][0]=1
            codec=Mock();codec.decode.side_effect=RuntimeError('model interrupted')
            with self.assertRaisesRegex(RuntimeError,'interrupted'):h.cal.recover_once(Path(tmp)/'decode','a',sr,a,p,codec,models)
            with self.assertRaisesRegex(ValueError,'no automatic repeat'):h.cal.recover_once(Path(tmp)/'decode','a',sr,a,p,codec,models)
            self.assertEqual(codec.decode.call_count,1)


if __name__=='__main__':unittest.main()
