import copy
import numpy as np
import pytest
from tools.legacy_reference_execution import selected_endpoint,validate_waveform
from tools.replay_token_efficiency_n4084 import release_ready


def policies():
    return {'actions':{'1.0':'m8_plus_latent_512_fold_N4084','13.0':'m8_plus_latent_1024'}},{'actions':{f:{'4084':{'Dc':{'1.0':{'quality':8},'13.0':{'quality':9}}}} for f in ('raw','arithmetic')}}


def test_frozen_legacy_endpoint_selection():
    policy,digital=policies()
    assert selected_endpoint('calibration_frozen_resource_lookup',1,policy,digital)=='m8_plus_latent_512_fold_N4084'
    assert selected_endpoint('raw_adaptive_m789_Dc',13,policy,digital)=='raw_N4084_m9_Dc'
    assert selected_endpoint('original_1024_Dc',1,policy,digital)=='m8_plus_latent_1024'
    with pytest.raises(ValueError):selected_endpoint('unregistered',1,policy,digital)
    digital['actions']['raw']['4084']['Dc']['1.0']['quality']=10
    with pytest.raises(ValueError):selected_endpoint('raw_adaptive_m789_Dc',1,policy,digital)
    policy['actions']['1.0']='new_free_action'
    with pytest.raises(ValueError):selected_endpoint('calibration_frozen_resource_lookup',1,policy,digital)


def test_actual_waveform_energy_not_ledger_only():
    wave=np.ones((4084,2),dtype=np.float32)
    assert validate_waveform(wave)==8168
    for bad in (wave[:-1],wave*2,np.full_like(wave,np.nan),np.full_like(wave,np.inf)):
        with pytest.raises(ValueError):validate_waveform(bad)


def test_serial_release_requires_all_completions_and_actual_exits():
    chain={'status':'REGISTERED_GRIDS_EXECUTED_REFERENCE_AUDIT_PUBLICATION_AND_REMOTE_ACCEPTANCE_PENDING','live':False}
    guard={'live':False};ref={'status':'REAL_REFERENCE_METRICS_COMPLETE_PENDING_REVIEW_AND_PUBLICATION','live':False}
    live=lambda s:s['live']
    assert release_ready(chain,guard,ref,True,True,live)
    assert not release_ready(chain,guard,ref,False,True,live)
    assert not release_ready(chain,guard,ref,True,False,live)
    for pos in range(3):
        states=copy.deepcopy([chain,guard,ref]);states[pos]['live']=True
        assert not release_ready(*states,True,True,live)
    bad=copy.deepcopy(ref);bad['status']='WAITING'
    assert not release_ready(chain,guard,bad,True,True,live)
