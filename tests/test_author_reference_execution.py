"""Synthetic engineering tests only; never author model quality acceptance."""
import numpy as np
import pytest
from tools.author_reference_execution import execute, receive, resources, frame_counter
from tools.replay_author_reference_timing import release_ready


def test_natural_resources_and_registered_counter():
    assert [sum(resources('swin',r)) for r in (32,64,96)]==[4498,8754,12952]
    assert [sum(resources('adjscc',r)) for r in (2,4,6)]==[4096,8192,12288]
    assert frame_counter(11,13.,2001)==2026091600+11*32+9
    assert frame_counter(11,13.,4101,True)==frame_counter(11,13.,2001)
    with pytest.raises(ValueError):resources('swin',33)


def test_single_noise_and_receiver_has_decoded_metadata_only():
    events=[];sent={'power':9.,'indices':list(range(32))}
    decoded={'power':2.,'indices':list(range(32,64)),'usable':True,'crc_accepted':False,'payload':np.array([0])}
    class Model:
        def transmit(self,*args):events.append('TX');return np.ones((4096,2)),sent
        def receive(self,data,snr,metadata):
            events.append('RX');assert metadata is decoded and metadata is not sent
            assert data.shape==(4096,2)
            return np.zeros((3,256,256),np.float32)
    def encode(*args):return {'symbols':np.ones((402,2)),'payload':np.array([1])}
    def decode(obs,*args):events.append('decode');assert obs.shape==(402,2);return decoded
    def noise(*args):events.append('noise');return np.zeros(args[-1])
    image,info,_=execute(Model(),'swin',32,np.zeros((3,256,256),np.uint8),'source',13.,2001,0,
                         sync=lambda:events.append('sync'),noise_fn=noise,encode=encode,decode=decode)
    assert events==['sync','TX','sync','noise','sync','decode','RX','sync']
    assert info['N']==4498 and info['E']==8996 and info['metadata_exact'] is False
    assert info['metadata_crc_accepted'] is False and info['fallback']==''


def test_invalid_metadata_gray_and_nan_rejection():
    class Model:
        def receive(self,*args):raise AssertionError('illegal metadata must not reach neural RX')
    image,info=receive(Model(),'swin',32,np.ones((4498,2)),1.,0,lambda *args:{'usable':False,'crc_accepted':False})
    assert np.all(image==.5)
    with pytest.raises(ValueError):receive(Model(),'swin',32,np.full((4498,2),np.nan),1.,0,None)


def test_adjscc_has_no_free_power_and_no_header():
    class Model:
        def base_receive(self,obs,snr,counter):
            assert obs.shape==(4096,2) and counter==123
            return np.zeros((3,256,256),np.float32)
    def reject(*args):raise AssertionError('bare ADJSCC has no header')
    _,d=receive(Model(),'adjscc',2,np.zeros((4096,2)),7.,123,reject)
    assert d=={'usable':True,'crc_accepted':None}


def test_release_requires_actual_completion_and_all_process_exits():
    states=[{'status':s} for s in ('REGISTERED_GRIDS_EXECUTED_REFERENCE_AUDIT_PUBLICATION_AND_REMOTE_ACCEPTANCE_PENDING',
              'LOW_BUDGET_MILESTONE_COMPLETE_LATER_STAGES_PENDING','REAL_REFERENCE_METRICS_COMPLETE_PENDING_REVIEW_AND_PUBLICATION',
              'REAL_N4084_REPLAY_COMPLETE_PENDING_REVIEW_AND_PUBLICATION')]
    assert release_ready(states,[True]*3,lambda s:False)
    assert not release_ready(states,[True,False,True],lambda s:False)
    assert not release_ready(states,[True]*3,lambda s:s is states[3])
    states[0]['status']='RUNNING_STAGE'
    assert not release_ready(states,[True]*3,lambda s:False)
