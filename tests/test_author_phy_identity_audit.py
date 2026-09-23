import pytest
from tools.audit_author_phy_identity import resources,validate_row


def frame():
    return dict(method='swin_ra32',protocol='common_paid_information',rate=32,data_complex_uses=4096,metadata_complex_uses=402,complex_uses=4498,actual_total_energy=8996,checkpoint_sha256='a'*64,NFE=0,batch_size=1,precision='FP32_TF32_disabled',snr_db=1,seed=2001,TX_seconds=.01,RX_seconds=.02,data_transmitted_sha256='b'*64,data_observed_sha256='c'*64,standard_data_noise_sha256='d'*64,source_pixels_sha256='e'*64)


def test_real_metadata_budget_formula():
    assert [resources('swin',r) for r in (32,64,96)]==[(4096,402,4498),(8192,562,8754),(12288,664,12952)]
    assert [resources('adjscc',r) for r in (2,4,6)]==[(4096,0,4096),(8192,0,8192),(12288,0,12288)]
    with pytest.raises(ValueError):resources('adjscc',3)


def test_paid_information_cannot_be_free_or_relabelled():
    row=frame();validate_row(row,'swin','a'*64)
    for field,value in [('protocol','author_assumed_information'),('metadata_complex_uses',0),('complex_uses',4096),('unmetered_sender_metadata','mask'),('checkpoint_sha256','f'*64),('data_complex_uses',3060)]:
        changed={**row,field:value}
        with pytest.raises(ValueError):validate_row(changed,'swin','a'*64)


def test_finite_resource_noise_and_timing_identity():
    row=frame()
    for field,value in [('actual_total_energy',float('nan')),('actual_total_energy',8990),('TX_seconds',float('inf')),('RX_seconds',-1),('standard_data_noise_sha256',''),('seed',4101),('snr_db',2),('batch_size',2),('precision','TF32')]:
        with pytest.raises(ValueError):validate_row({**row,field:value},'swin','a'*64)
