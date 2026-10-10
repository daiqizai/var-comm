"""Small synthetic recovery/accounting checks; no scientific backend or models."""
import contextlib,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import t6_raw_calibrate as c
import t6_raw_gpu_parallel as parallel
from t2_ledger import Ledger


class RawCalibrationTests(unittest.TestCase):
    def test_reference_feature_resume_does_not_rerun_models(self):
        import pickle
        class Torch:
            @staticmethod
            def save(value,handle):pickle.dump(value,handle)
            @staticmethod
            def load(path,weights_only=False):
                with Path(path).open('rb')as handle:return pickle.load(handle)
        class Scorer:
            torch=Torch();calls=0
            def prepare_source(self,record,target,index):
                self.calls+=1;self.target=target;self.reference=target[None];self.prepared={'feature':np.arange(5,dtype=np.float32)}
                self.record_identity=(record['image_id'],record['preprocessing_id'],index)
        with tempfile.TemporaryDirectory()as td:
            scorer=Scorer();record=dict(image_id='one',preprocessing_id='p');pixels=np.zeros((3,256,256),np.uint8);r={'old_score_identity':{'model':'fixed'}}
            c.prepare_reference_once(td,r,'r',scorer,record,pixels,0);scorer.prepared=None;c.prepare_reference_once(td,r,'r',scorer,record,pixels,0)
            self.assertEqual(scorer.calls,1);np.testing.assert_array_equal(scorer.prepared['feature'],np.arange(5,dtype=np.float32))

    def test_public_counters_keep_pilot_full_identical(self):
        values={c.counter(i,s,n)for i in range(1000)for s in c.SNRS for n in(4101,4102,4103)}
        self.assertEqual(len(values),9000);self.assertEqual(min(values),0);self.assertEqual(max(values),8999)

    def test_source_assignments_disjoint(self):
        for n in(100,1000):
            groups=parallel.assignments({'source_count':n})
            self.assertFalse(set(groups[0])&set(groups[1]));self.assertEqual(sorted(groups[0]+groups[1]),list(range(n)))

    def test_image_proof_rejects_altered_pixels(self):
        with tempfile.TemporaryDirectory()as td:
            p=Path(td)/'x.npz';image=np.full((3,256,256),.5,np.float32);np.savez(p,image=image)
            v=dict(image_path=str(p),image_key='image',image_slot=0,image_file_sha256=c.h.sha(p),image_sha256=c.h.image_sha(image))
            np.testing.assert_array_equal(c.checked_image(v,{}),image)
            v['image_sha256']='wrong'
            with self.assertRaises(RuntimeError):c.checked_image(v,{})

    def test_aggregate_direction_whole_fallback_and_source_mean(self):
        schedule=[dict(snr_db=s,candidate_id=cid,profile=dict(candidate_id=cid,profile_id=j,m=7,K=K))
            for s in c.SNRS for j,(cid,K)in enumerate([('a',0),('b',5)])]
        r=dict(schedule=schedule,source_count=2,noise_seeds=[4101,4102,4103]);rows=[]
        for i in range(2):
            for p in schedule:
                for n in r['noise_seeds']:
                    rows.append(dict(source_index=i,snr_db=p['snr_db'],candidate_id=p['candidate_id'],body_crc_accept=True,header_ok=True,
                        psnr_db=20+i,lpips_alex=.2+i/100,dinov2_vitl14_cosine=.8 if p['candidate_id']=='a'else .7))
        rankings,winners=c.aggregate(r,rows)
        self.assertEqual(len(rankings),6);self.assertEqual(len(winners),6)
        self.assertTrue(all(x['candidate_id']=='a'for x in winners))
        self.assertTrue(all(x['psnr_db']==20.5 for x in rankings))

    def test_missing_noise_cannot_close(self):
        r=dict(schedule=[dict(snr_db=4,candidate_id='a',profile=dict(profile_id=0,m=7,K=0))],source_count=1,noise_seeds=[4101,4102,4103])
        with self.assertRaises(RuntimeError):c.aggregate(r,[])

    def test_actual_cpu_resume_uses_durable_checkpoint(self):
        with tempfile.TemporaryDirectory()as td:
            out=Path(td);rh='f'*64;profile=dict(profile_id=0,candidate_id='one',wire_key='wire')
            r=dict(out=str(out),workers=1,records=[dict(source_index=0,source_id='one',preprocessing_id='pixels')],source_count=1,
                noise_seeds=[4101],schedule=[dict(snr_db=s,profile=profile)for s in c.SNRS],phy_qualification=dict(path='qual'),
                packet_cap=6,pilot_execution_request=None,pilot_completion=None,deadline_unix=time.time()+60,max_seconds=60,stop_files=[])
            class Catalogue:
                digest='catalogue'
                def entry(self,p):return profile
            class Runtime:
                catalogue=Catalogue()
                def transmit(self,p,scales,ctr):return np.ones((2048,2)),dict(counter=ctr)
                def receive(self,y,snr,ctr,meter,event,phase):
                    header=meter.call(dict(event_id=event+'.header'),dict(y=c.h.image_sha(y)),lambda:dict(header_ok=False))
                    return dict(header=header,body=None,receiver_state=dict(kind='gray',m=0,K=0,prefix=[],partial_values=[],order='raster'),
                        gray=True,header_ok=False,body_crc_accept=None,logical_packet_calls=1)
            def opened(*args):return Ledger(out/'packet_ledger.sqlite',rh,6)
            with patch.object(c,'registered',return_value=(r,rh)),patch.object(c,'open_ledger',side_effect=opened),patch.object(c.phy,'create_runtime',return_value=Runtime()),\
                 patch.object(c.h,'source_assets',return_value=(np.zeros(680,np.int64),np.zeros((3,256,256),np.uint8))),\
                 patch.object(c.h,'lock',side_effect=lambda p:contextlib.nullcontext()),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                c.cpu_worker('unused',0);first=(out/'cpu_sources/0000.json').read_bytes();c.cpu_worker('unused',0)
            self.assertEqual(first,(out/'cpu_sources/0000.json').read_bytes())
            meter=opened();self.assertEqual(meter.snapshot(),dict(total=3,unresolved=0,cap=6));meter.close()
            self.assertEqual(len(json.loads(first)['frames']),3)

if __name__=='__main__':unittest.main()
