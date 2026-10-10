"""Synthetic fixed16 scheduling and actual SQLite callback checks; no science."""
import argparse,os,tempfile,time,unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import numpy as np
import t5_entropy_examples as t5
import t1_phy
from t1_entropy_core import candidates,write,sha,read
from t1_source_asset_schema import token_sha

class Checks(unittest.TestCase):
    def test_missing_prefix_is_not_gray(self):
        c=next(x for x in candidates() if x['target_m']==7 and x['q']==2 and x['nominal_rate']=='1/2')
        r=dict(family='EC_VAR_WHOLE',policies={str(s):c for s in t5.SNRS})
        cp=dict(streams={'EC_VAR_WHOLE':{'6':{'payload_bits':950},'7':{'payload_bits':1500}}})
        self.assertTrue(all(v['status']=='BLOCKED_MISSING_PREFIX' for v in t5.selections(r,cp).values()))
        cp['streams']['EC_VAR_WHOLE']['5']={'payload_bits':600}
        self.assertTrue(all(v['actual_m']==5 for v in t5.selections(r,cp).values()))

    def test_fixed16_actual_sqlite_callbacks_never_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);family='EC_VAR_WHOLE';c=next(x for x in candidates() if x['target_m']==7 and x['q']==2 and x['nominal_rate']=='2/3')
            catalogue={'profiles':[dict(family=family,m=7,q=2,nominal_rate='2/3',profile_id=7)]};freeze=out/'family.json';write(freeze,{'synthetic':True})
            records=[]
            for i in t5.INDICES:
                a=out/f'{i:04d}.npz';tokens=np.zeros(680,np.int64)
                with a.open('xb') as f:np.savez(f,tokens=tokens,bits=np.zeros(8,np.uint8))
                cp=out/f'{i:04d}.json';write(cp,dict(archive={'path':str(a),'sha256':sha(a)},source_tokens_sha256=token_sha(tokens),
                    streams={family:{'7':dict(bits_key='bits',payload_bits=8)}}))
                records.append(dict(source_index=i,source_id='synthetic'+str(i),checkpoint=str(cp),checkpoint_sha256=sha(cp)))
            req=out/'request.json';r=dict(schema='T5_ENTROPY_FIXED16_ACTUAL_V1',root=tmp,out=tmp,family=family,family_freeze=t5.pin(freeze),
                source_indices=t5.INDICES,snrs=t5.SNRS,noise_seed=6201,frame_count=48,packet_cap=96,holdout_used=False,source_bindings={},
                policies={str(s):c for s in t5.SNRS},phy_qualification={'path':'SYNTHETIC'},registered_profile_catalogue=catalogue,
                deadline_unix=time.time()+60,max_seconds=60,stop_files=[])
            write(req,r);rh=sha(req);write(out/'sources_complete.json',dict(request_sha256=rh,records=records));calls=[]
            class Runtime:
                profiles={7:catalogue['profiles'][0]}
                def __init__(self):self.catalogue=catalogue
                def transmit(self,p,b,c):return np.zeros((1024,2)),{'synthetic':True}
                def receive(self,obs,snr,counter,profiles,ledger,event,phase):
                    for stage in ('header','body'):
                        def callback(stage=stage):calls.append((event,stage));return {'synthetic_decoded':True}
                        ledger.call({'event_id':event+'/'+stage},{'counter':counter,'phase':phase},callback)
                    return dict(header={'header_ok':True},body={'crc_accepted':False},rx_profile=None,status='CRC_REJECT')
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''),patch.object(t1_phy,'create_runtime',lambda *a:Runtime()),patch.object(t5.shared,'lock',lambda *a:nullcontext()):
                first=t5.cpu(str(req));second=t5.cpu(str(req))
            self.assertEqual(len(calls),96);self.assertEqual(first,second);self.assertEqual(first['unresolved'],0)
            done=read(out/'cpu_complete.json');self.assertEqual(len(done['frames']),48)
            self.assertEqual({x['source_index'] for x in done['frames']},set(t5.INDICES))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    write(a.out,dict(status='PASS' if result.wasSuccessful() else 'FAIL',synthetic_only=True,tests_run=result.testsRun,
        real_model_calls=0,real_channel_calls=0,real_bootstrap_calls=0,synthetic_SQLite_callbacks=96,
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ('t5_entropy_examples.py','t5_entropy_selfcheck.py')}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
