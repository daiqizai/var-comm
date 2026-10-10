"""CPU-only checks of actual-input reuse, independent budgets and ownership."""
from contextlib import nullcontext
import copy
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_pilot_visual_v1 as h
import ep_source_codec as partial


def received(payload=(1,0,1),q=2):
    return dict(status='PAYLOAD_PARSED',header=dict(header_ok=True,profile_id=144),
        body=dict(crc_accepted=True,parser_accepted=True,payload=list(payload)),
        rx_profile=dict(family='EC_VAR_PARTIAL',m=4,K=6,q=q,profile_id=144))


class VisualTests(unittest.TestCase):
    def test_private_engine_does_not_change_running_PHY_or_source_caps(self):
        prior=dict(h.source.CAPS);runner=h.engine()
        self.assertEqual(h.source.CAPS,prior)
        self.assertEqual(runner.CAPS,h.core.VISUAL_CAPS)
        self.assertEqual(runner.worker_identity.__globals__['__file__'],h.__file__)
        self.assertEqual(runner.adapt_backend.__globals__['CAPS']['prior_scale'],864000)

    def test_frozen_PHY_script_change_blocks_new_visual_owner(self):
        with mock.patch.object(h,'PHY_SCRIPT_SHA','0'*64),self.assertRaisesRegex(RuntimeError,'PHY owner changed'):
            h.engine()

    def test_visual_window_and_model_caps_are_independent_finite(self):
        e=h.execution(time.time()+21600,21600)
        self.assertEqual(e['GPU_workers'],1);self.assertEqual(e['allocator_cap_bytes'],16*(1<<30))
        self.assertEqual(h.CAPS['model_load'],1);self.assertEqual(h.CAPS['source_tx'],0)
        self.assertEqual(h.CAPS['source_rx'],43200);self.assertEqual(h.CAPS['decoder_forward'],43200)
        with self.assertRaises(RuntimeError):h.execution(time.time()+30000,21601)

    def test_cross_device_old_images_are_cache_miss(self):
        old=dict(device_identity_binding={'GPU':0})
        with mock.patch.object(h.g,'checked_json',return_value=old):
            self.assertEqual(h.reusable48(h.visual_identity({'GPU':2})),[])

    def test_CPU_wait_failure_cannot_admit_reconstruction(self):
        root=h.gate.RT/'qualification/h800_ep_pilot_phy_v1_attempt1'
        pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
                  owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
        done=dict(status=h.physical.PASS,actual_wait=dict(success=True),actual_children_waited=True,worker_exit_codes=[0])
        with mock.patch.object(h.gate,'readpin',side_effect=[done,dict(actual_wait=True,returncode=1)]),\
             self.assertRaisesRegex(RuntimeError,'CPU owner wait'):
            h.cpu_closure(pins)

    def test_conflicting_closed48_images_for_same_actual_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);records={}
            for i,value in enumerate((.25,.75)):
                image=np.full((3,256,256),value,np.float32);path=root/f'{i}.npz';np.savez(path,image=image)
                records[str(i)]=dict(image_archive=h.core.descriptor(path),image_sha256=h.g.image_sha(image),
                    physical_frame=dict(path='physical',sha256='b'*64),evidence={})
            records['physical']=dict(actual_RX=received())
            request=dict(reuse48_images=[dict(path=str(i),sha256='a'*64) for i in range(2)],
                         visual_identity={'actual':'same'},physical_frames=[])
            with mock.patch.object(h.gate,'readpin',new=lambda pin:records[pin['path']]),\
                 mock.patch.object(h.g,'inside',side_effect=Path),self.assertRaisesRegex(RuntimeError,'inconsistent old images'):
                h.images(request,None,None,None,None,lambda:None,root/'results')

    def test_exact_actual_input_reuse_preserves_each_frames_failure_and_NEW_counts(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);old_image=np.full((3,256,256),.25,np.float32);archive=out/'old.npz'
            np.savez(archive,image=old_image);archive_pin=h.core.descriptor(archive)
            a=received();b=received(q=6)
            header=received();header.update(status='HEADER_REJECT',body=None,rx_profile=None);header['header']['header_ok']=False
            crc=received();crc['body']['crc_accepted']=False;crc['status']='CRC_REJECT'
            changed=received(payload=(0,1,0));events=[a,b,header,crc,changed,changed]
            ledger=h.g.Ledger(out/'calls',lambda:None,h.CAPS)
            backend=types.SimpleNamespace(g=h.g,source_index=0,traces=[],native=types.SimpleNamespace(torch=types.SimpleNamespace(no_grad=nullcontext)))
            ledger.call('model_load',lambda:backend)
            class Codec:
                def decode(self,family,bits,m,K):
                    for _ in range(m+1):ledger.call('prior_scale',lambda:None)
                    return dict(canonical=True,zero_extension_reads=30,received_tokens=np.zeros(partial.token_count(m,K),np.int64))
            def render(native,tokens,m,K):
                for _ in range(10):ledger.call('prior_scale',lambda:None)
                return ledger.call('decoder_forward',lambda:np.full((3,256,256),.75,np.float32))
            backend.render_function=render
            old=dict(physical_frame={'path':'old-physical','sha256':'c'*64},image_archive=archive_pin,
                     image_sha256=h.g.image_sha(old_image),evidence=dict(render_called=True,source_status='ARITHMETIC_SOURCE_DECODED'))
            saved={}
            def read(pin):
                path=pin['path']
                if path=='old-result':return old
                if path=='old-physical':return dict(actual_RX=a)
                if path.startswith('physical'):
                    return dict(actual_RX=events[int(path[8:])])
                index=int(path[7:]);return dict(frame_index=index,logical_event=dict(source_index=index%100),
                    physical_frame=dict(path='physical'+str(index%6),sha256='d'*64))
            def write(path,value):
                if path.parent.name=='logical' and int(path.stem)<6:saved[int(path.stem)]=value
            r=dict(visual_identity={'actual':'same'},reuse48_images=[dict(path='old-result',sha256='b'*64)],
                   physical_frames=[dict(path=f'logical{i}',sha256='a'*64) for i in range(43200)])
            with mock.patch.object(h.gate,'readpin',new=read),mock.patch.object(h.g,'inside',side_effect=Path),\
                 mock.patch.object(h.g,'write',new=write),\
                 mock.patch.object(h.core,'descriptor',new=lambda p:dict(path=str(p),sha256='f'*64)):
                result=h.images(r,backend,Codec(),partial,ledger,lambda:None,out/'results')
            self.assertEqual(result['reuse']['new_actual_source_inputs'],2) # one fixed gray and one actual new payload
            self.assertEqual(result['reuse']['closed48'],14400)
            self.assertEqual(ledger.completed['source_rx'],1)
            self.assertEqual(ledger.completed['var_render'],1)
            self.assertEqual(ledger.completed['prior_scale'],15)
            self.assertEqual(saved[2]['actual_RX_status'],'HEADER_REJECT')
            self.assertEqual(saved[3]['actual_RX_status'],'CRC_REJECT')
            self.assertNotEqual(saved[2]['actual_header_ok'],saved[3]['actual_header_ok'])
            self.assertEqual(saved[2]['reconstruction'],saved[3]['reconstruction'])
            self.assertNotEqual(saved[1]['actual_received_profile']['q'],saved[0]['actual_received_profile']['q'])
            self.assertEqual(saved[1]['reconstruction'],saved[0]['reconstruction'])


if __name__=='__main__':unittest.main()
