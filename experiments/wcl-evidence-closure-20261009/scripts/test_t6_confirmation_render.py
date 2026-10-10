"""Synthetic scope, resume, source dispatch and zero-metric render contracts."""
import contextlib,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import t6_confirmation_render as c
from t2_ledger import Ledger


class ConfirmationTests(unittest.TestCase):
    def test_all2700_fixed_method_noise_conditions(self):
        import t6_score as score
        keys={(m,s,i,n)for m in c.methods('EC_VAR_WHOLE')for s in c.SNRS for i in range(100)for n in c.SEEDS}
        self.assertEqual(len(keys),2700)
        self.assertEqual(c.methods('EC_VAR_WHOLE'),score.methods('EC_VAR_WHOLE'))
        ids=['fixed_'+str(i)for i in range(100)]
        rows=[dict(method_id=m,snr_db=s,source_index=i,noise_seed=n,source_id=ids[i],point_id=m+'_SNR_'+str(s),N=2048,status='HEADER_REJECT_GRAY')for m,s,i,n in keys]
        self.assertEqual(len(score.frame_grid(rows,ids,'EC_VAR_WHOLE')),2700)
        counters={c.counter(i,s,n)for i in range(100)for s in c.SNRS for n in c.SEEDS}
        self.assertEqual(counters,set(range(900)))

    def test_partial_diagnostics_include_received_partial_values(self):
        state=dict(kind='tokens',m=2,K=2,prefix=[[4],[1,2,3,5]],partial_values=[6,7])
        tokens=np.array([4,1,2,3,5,6,8]+[0]*673,np.int64)
        self.assertEqual(c.token_diagnostics(state,tokens),dict(token_error_count=1,token_error_denominator=7,received_tokens_equal_source=False))

    def test_two_cpu_branches_shared_budget_actual_cache_and_gpu_resume(self):
        with tempfile.TemporaryDirectory()as td:
            out=Path(td);rh='d'*64;profile=dict(profile_id=0,candidate_id='raw',m=7,K=0,groups=[dict(modulation='QPSK')]);candidate=dict(candidate_id='ec',target_m=7,q=2,nominal_rate='1/2')
            methods=c.methods('EC_VAR_WHOLE');schedule=[]
            for s in c.SNRS:
                schedule.extend(dict(method_id=m,snr_db=s,branch='raw',profile=profile)for m in methods[:2])
                schedule.append(dict(method_id=methods[2],snr_db=s,branch='entropy',candidate=candidate))
            record=dict(source_index=0,source_id='fixed_new',preprocessing_id='p');r=dict(out=str(out),root=td,source_count=1,records=[record],
                methods=methods,schedule=schedule,workers=1,gpu_workers=1,family='EC_VAR_WHOLE',source_model_id={'EC_VAR_WHOLE':'model'},
                raw_phy_qualification={'path':'raw'},entropy_phy_qualification={'path':'ec'},static_completion={'path':'static'},
                packet_cap=5400,noise_seeds=c.SEEDS,deadline_unix=time.time()+60,max_seconds=60,stop_files=[],visual_config={'visual_lock':str(out/'visual.lock')})
            request=out/'request.json';c.h.save(request,r)
            class Cat:
                def entry(self,p):return profile
            class Raw:
                catalogue=Cat()
                def transmit(self,p,x,ctr):
                    return np.ones((2048,2)),dict(body_symbols=1980,padding_symbols=0,source_bits=1860,k=1876,n=3960,E_frame=4096.,rho=1.)
                def receive(self,y,snr,ctr,meter,event,phase):
                    hd=meter.call(dict(event_id=event+'.header'),{'y':c.h.image_sha(y)},lambda:dict(header_ok=False,header_crc_ok=False,header_fields_legal=False,profile_id=None))
                    return dict(header=hd,body=None,receiver_state=dict(kind='gray',m=0,K=0,prefix=[],partial_values=[],order=None),body_crc_accept=False,source_status='HEADER_REJECT_GRAY')
            class EC:
                catalogue={'profiles':[dict(profile_id=0,m=7,K=0,q=2,nominal_rate='1/2')]};profiles={'0':catalogue['profiles'][0]}
                def transmit(self,p,x,ctr):return np.ones((2048,2)),dict(k=1980,n=3960,arithmetic_bits=len(x),known_information_padding_bits=1949,E_frame=4096.,rho=1.)
                def receive(self,y,snr,ctr,profiles,meter,event,phase):
                    hd=meter.call(dict(event_id=event+'.header'),{'y':c.h.image_sha(y)},lambda:dict(header_ok=False,header_crc_ok=False,header_fields_legal=False,profile_id=None))
                    return dict(header=hd,body=None,rx_profile=None,status='HEADER_REJECT')
            class Native:
                def frozen(self):pass
            def opened(*_):return Ledger(out/'packet_ledger.sqlite',rh,5400)
            source=(np.zeros(680,np.int64),np.zeros((3,256,256),np.uint8),{},{});selection=dict(attempts=[dict(m=7,bits=2,fits=True)])
            with patch.object(c,'registered',return_value=(r,rh)),patch.object(c,'load_source',return_value=source),\
                patch.object(c.h,'lock',side_effect=lambda p:contextlib.nullcontext()),patch.object(c.rawcal,'open_ledger',side_effect=opened),\
                patch.object(c.rawphy,'create_runtime',return_value=Raw()),patch.object(c.ecphy,'create_runtime',return_value=EC()),\
                patch.object(c.oldrender,'selected_stream',return_value=(7,np.array([0,1],np.uint8),selection)),\
                patch.object(c.oldrender,'build_native',return_value=(Native(),None)),patch.object(c,'SourceCodec10',return_value=None),\
                patch.object(c,'qualify_single'),patch.object(c.h,'render_state',side_effect=lambda *args:np.full((3,256,256),.5,np.float32)),\
                patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                c.cpu_worker(str(request),'raw',0);c.cpu_worker(str(request),'entropy',0);c.cpu_worker(str(request),'raw',0);c.cpu_worker(str(request),'entropy',0)
                meter=opened();self.assertEqual(meter.snapshot(),dict(total=18,unresolved=0,cap=5400));meter.close()
                plan=c.plan_render(str(request));self.assertEqual(plan['new_VAR_source_decode_cap'],0);self.assertEqual(plan['new_VAR_render_cap'],1)
                with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                    c.gpu_worker(str(request));before=(out/'gpu_sources/0000.json').read_bytes();c.gpu_worker(str(request))
                self.assertEqual(before,(out/'gpu_sources/0000.json').read_bytes());done=json.loads(before)
                self.assertEqual(len(done['rows']),27);self.assertEqual(done['new_metric_calls'],0);self.assertEqual(done['new_VAR_render_calls'],0)
                self.assertTrue(all(x['gray']and x['N']==2048 for x in done['rows']))

if __name__=='__main__':unittest.main()
