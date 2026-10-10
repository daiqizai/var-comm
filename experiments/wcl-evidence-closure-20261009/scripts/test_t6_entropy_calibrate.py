"""Synthetic EC scheduling, RX ownership, finite-budget and resume checks.

The runtime and neural calls below are fakes. SQLite reservations and source
cache validation use the production implementation; no real science runs.
"""
import argparse,copy,os,tempfile,time,unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import t6_entropy_calibrate as f
from t1_codec_runtime import InvalidSourceStream
from t1_source_asset_schema import model_id,token_sha
from t6_source_codec import build_record
from t6_calibration_plan import entropy_candidates

FAMILY='EC_VAR_WHOLE'

class Checks(unittest.TestCase):
    def test_candidate_admission_must_retain_all_targets(self):
        profiles=[p for p in f.phy.profile_templates(FAMILY)if p['q']==2]
        candidates=[dict(c,actual_ldpc_layout_status='ACTUAL_CONSTRUCTOR_ADMITTED')for c in entropy_candidates()if c['q']==2]
        entries=[dict(source_index=i,source_id='synthetic'+str(i))for i in range(100)]
        schedule=[dict(snr_db=s,candidate_id=c['candidate_id'],candidate=c)for s in f.SNRS for c in candidates]
        r=dict(schema=f.SCHEMA,N=2048,phase='pilot',family=FAMILY,source_count=100,source_ids=[e['source_id']for e in entries],
            records=entries,old_records=entries,noise_seeds=[4101],snrs=f.SNRS,candidates=candidates,schedule=schedule,
            registered_profile_catalogue=dict(profiles=profiles),frame_count=len(schedule)*100,packet_cap=len(schedule)*200,
            gpu_workers=1,workers=2,holdout_used_for_selection=False,confirmation_images_opened=False)
        f.validate(r)
        bad=copy.deepcopy(r);bad['candidates']=bad['candidates'][:-1]
        with self.assertRaisesRegex(RuntimeError,'Candidate set'):f.validate(bad)
        bad=copy.deepcopy(r);bad['schedule'][1]=bad['schedule'][0]
        with self.assertRaisesRegex(RuntimeError,'Unique unchanged'):f.validate(bad)
        bad=copy.deepcopy(r);bad['family']='EC_STATIC_WHOLE'
        with self.assertRaisesRegex(RuntimeError,'Single preregistered'):f.validate(bad)

    def test_unknown_stream_requires_rx_decode_not_tx_truth(self):
        packet=dict(actual_RX=dict(header=dict(header_ok=True),status='PAYLOAD_PARSED',body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0]),
            rx_profile=dict(family=FAMILY,m=10)))
        sr=dict(schema='T1_SOURCE_ASSET_V1',streams={FAMILY:{}},source_role='calibration');models={FAMILY:model_id(FAMILY,'synthetic')}
        class Decoder:
            def __init__(self):self.calls=[]
            def decode(self,family,bits,m):
                self.calls.append((family,bits,m));return dict(received_tokens=np.full(680,17,np.int64))
        codec=Decoder();state,evidence=f.cal.recover(sr,{},packet,codec,models)
        self.assertEqual(codec.calls,[(FAMILY,[1,0],10)]);self.assertEqual(state['m'],10)
        self.assertEqual(state['prefix'][-1],[17]*256);self.assertEqual(evidence['new_VAR_source_decode'],1)
        class Invalid:
            def decode(self,*a):raise InvalidSourceStream('synthetic canonical error')
        state,evidence=f.cal.recover(sr,{},packet,Invalid(),models)
        self.assertEqual(state['kind'],'gray');self.assertEqual(evidence['source_status'],'SOURCE_PARSE_REJECT')
        class Broken:
            def decode(self,*a):raise OSError('synthetic device error')
        with self.assertRaises(OSError):f.cal.recover(sr,{},packet,Broken(),models)

    def test_missing_reachable_prefix_blocks_before_phy(self):
        sr=dict(streams={FAMILY:{'10':dict(payload_bits=100)}})
        with self.assertRaisesRegex(ValueError,'Missing or non-fitting'):
            f.cal.selected_stream(sr,{},FAMILY,dict(target_m=10,source_capacity=50))

    def test_cpu_gpu_close_and_resume_use_actual_identities(self):
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp);tokens=np.zeros(680,np.int64);records=[];old=[];sh='synthetic-static'
            for i in range(2):
                arrays=dict(tokens=tokens,var_m9_bits=np.array([0,1,0,1],np.uint8),var_m9_received_tokens=tokens[:424],
                    var_m10_bits=np.array([0,1,0,1,0,1],np.uint8),var_m10_received_tokens=tokens)
                ap=out/f'asset{i}.npz'
                with ap.open('xb')as z:np.savez(z,**arrays)
                record=build_record(source_index=i,source_id=f'synthetic{i}',tokens=tokens,preprocessing_id='synthetic-pre',
                    source_assets_checkpoint=dict(path='synthetic-only',sha256='synthetic'),archive=ap,arrays=arrays,static_completion_sha=sh,
                    origins={x:'SYNTHETIC_ONLY'for x in ('EC_STATIC_WHOLE',FAMILY)},
                    upstream_evidence={x:{}for x in ('EC_STATIC_WHOLE',FAMILY)})
                cp=out/f'asset{i}.json';f.h.save(cp,record)
                records.append(dict(source_index=i,source_id=f'synthetic{i}',checkpoint=str(cp),checkpoint_sha256=f.h.sha(cp)))
                old.append(dict(source_index=i,source_id=f'synthetic{i}',preprocessing_id='synthetic-pre',class_index=0))
            profiles=[dict(profile_id=m,m=m,family=FAMILY,q=2,nominal_rate='1/2',profile_key='synthetic'+str(m))for m in (9,10)]
            catalog=dict(profiles=profiles);cs=[dict(candidate_id='target'+str(m),target_m=m,q=2,nominal_rate='1/2',source_capacity=4)for m in (9,10)]
            schedule=[dict(snr_db=s,candidate_id=c['candidate_id'],candidate=c)for s in f.SNRS for c in cs]
            r=dict(schema=f.SCHEMA,N=2048,root=tmp,out=tmp,phase='pilot',family=FAMILY,records=records,old_records=old,
                source_count=2,source_ids=[x['source_id']for x in records],snrs=f.SNRS,noise_seeds=[4101],candidates=cs,schedule=schedule,
                frame_count=12,packet_cap=24,workers=1,gpu_workers=1,source_model_id={FAMILY:model_id(FAMILY,sh)},
                static_completion=dict(path='synthetic'),phy_qualification=dict(path='synthetic'),registered_profile_catalogue=catalog,
                pilot_execution_request=None,pilot_completion=None,visual_config=dict(visual_lock='synthetic'),
                deadline_unix=time.time()+90,max_seconds=90,stop_files=[])
            req=out/'execution_request.json';f.h.save(req,r);rh=f.h.sha(req);calls=[];sourcecalls=[];rendercalls=[];scorecalls=[]
            class Runtime:
                catalogue=catalog
                def __init__(self):self.profiles={str(p['profile_id']):p for p in profiles}
                def transmit(self,p,b,c):
                    return np.zeros((2048,2),np.float64),dict(E_frame=4096.,rho=1.,k=1980,n=3960,
                        arithmetic_bits=len(b),known_information_padding_bits=1980-29-len(b))
                def receive(self,obs,snr,counter,profiles,ledger,event):
                    for stage in ('header','body'):
                        def callback(stage=stage):calls.append((event,stage));return dict(synthetic=True)
                        ledger.call(dict(event_id=event+':'+stage),dict(counter=counter),callback)
                    i=counter//3%1000;crc=not(i==0 and snr==4);m=10 if i else 9
                    body=dict(crc_accepted=crc,parser_accepted=crc,payload=([1,1]if i else[0,1,0,1])if crc else None)
                    return dict(header=dict(header_ok=True),body=body,rx_profile=dict(profiles[str(m)]),status='PAYLOAD_PARSED'if crc else'CRC_REJECT')
            class Codec:
                def __init__(self,*a):pass
                def decode(self,family,bits,m):sourcecalls.append((family,bits,m));return dict(received_tokens=np.zeros(680,np.int64))
            class Native:
                def frozen(self):pass
            class Scorer:
                def prepare_source(self,*a):pass
                def __call__(self,record,images):
                    scorecalls.append(record['source_index']);return [{m:.5 for m in f.h.METRICS}for _ in images]
            gray=out/'gray.npz'
            with gray.open('xb')as z:np.savez(z,image=np.full((3,256,256),.5,np.float32))
            gv=dict(state=f.cal.gray_state(),image_path=str(gray),image_key='image',image_slot=0,image_file_sha256=f.h.sha(gray),
                image_sha256=f.h.image_sha(np.full((3,256,256),.5,np.float32)),metrics={m:.5 for m in f.h.METRICS},reuse='SYNTHETIC_OLD_GRAY')
            def render(native,source,state):rendercalls.append(state['m']);return np.full((3,256,256),state['m']/20,np.float32)
            def qualify(*a):f.h.save(out/'qualification_gpu.json',dict(status='PASS',request_sha256=rh))
            def prepare_reference(out,r,rh,scorer,record,pixels,i):
                ref=Path(out)/'metric_references'/f'{i:04d}.json';ap=ref.with_suffix('.pt');ref.parent.mkdir(exist_ok=True)
                if not ap.exists():ap.write_bytes(b'SYNTHETIC_NO_REAL_FEATURES')
                f.h.save(ref,dict(request_sha256=rh,archive_sha256=f.h.sha(ap)))
                scorer.prepare_source(record,pixels.astype(np.float32)/255,i)
            def registration(path):self.assertEqual(str(path),str(req));return r,rh
            with (patch.object(f,'registered',registration),patch.object(f.h,'lock',lambda *a:nullcontext()),
                patch.object(f.phy,'create_runtime',lambda *a:Runtime()),
                patch.object(f.raw,'open_ledger',lambda r,rh:f.raw.Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap'])),
                patch.dict(os.environ,CUDA_VISIBLE_DEVICES='')):
                f.ledger_init(str(req));f.cpu_worker(str(req),0);f.cpu_worker(str(req),0)
                self.assertEqual(len(calls),12)
                with patch.object(f,'source_cache',lambda *a:{f.h.digest(gv['state']):gv}):plan=f.plan_gpu(str(req))
                self.assertEqual(plan['new_VAR_source_decode_cap'],1);self.assertEqual(plan['new_VAR_render_cap'],3)
                with (patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),patch.object(f.h,'gpu_build',lambda *a:(Native(),None,Scorer())),
                    patch.object(f,'SourceCodec10',Codec),patch.object(f.h,'qualify_gpu',qualify),
                    patch.object(f.raw,'prepare_reference_once',prepare_reference),
                    patch.object(f.h,'source_assets',lambda *a:(tokens,np.zeros((3,256,256),np.uint8))),patch.object(f.h,'render_state',render)):
                    f.gpu_worker(str(req));f.gpu_worker(str(req))
                self.assertEqual(len(sourcecalls),1);self.assertEqual(sorted(rendercalls),[9,10]);self.assertEqual(len(scorecalls),2)
                receipt=out/'owner_receipt.json';f.h.save(receipt,dict(request_sha256=rh,actual_children_waited=True,worker_exit_codes=[0]))
                result=f.close(str(req),str(receipt));self.assertEqual(result['frame_count'],12);self.assertEqual(result['packet_ledger']['total'],12)
                self.assertEqual(result['new_VAR_source_decodes'],1);self.assertEqual(result['new_VAR_renders'],2)
                done=f.h.read(out/'completion.json');self.assertIn(str(gray),done['outputs'])
                self.assertEqual(f.close(str(req),str(receipt)),result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    f.h.save(a.out,dict(status='PASS'if result.wasSuccessful()else'FAIL',synthetic_only=True,tests_run=result.testsRun,
        real_image_reads=0,real_model_calls=0,real_channel_calls=0,synthetic_SQLite_callbacks=12,
        source_sha256={n:f.h.sha(Path(__file__).with_name(n))for n in('t6_entropy_calibrate.py','t6_entropy_gpu_parallel.py','t6_entropy_phy.py','test_t6_entropy_calibrate.py')}))
    raise SystemExit(0 if result.wasSuccessful()else 1)
