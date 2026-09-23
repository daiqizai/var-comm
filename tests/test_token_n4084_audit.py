import copy
import itertools
import pytest
from tools.audit_token_efficiency_n4084 import DC, METHODS, METRICS, SNRS, SEEDS, validate_table, validate_timing


def frames():
    return [dict(method=m,image_id=s,source_index=i,snr_db=snr,seed=seed,N=4084,E=8168,decoder_sha256=DC,psnr_db=20,lpips_alex=.2,dino_cosine=.7) for m,(s,i),snr,seed in itertools.product(METHODS,{'a':0,'b':1}.items(),SNRS,SEEDS)]


def test_full_source_noise_pairing_rejects_duplicates_and_missing():
    data=frames();validate_table(data,{'a':0,'b':1})
    for bad in [data[:-1],data+[data[0]],data[:-1]+[data[0]]]:
        with pytest.raises(ValueError):validate_table(bad,{'a':0,'b':1})


def test_resource_decoder_and_metric_integrity():
    for field,value in [('N',4084.5),('E',8167),('decoder_sha256','wrong'),('source_index',1),('lpips_alex',float('nan')),('snr_db',2),('seed',4101)]:
        data=frames();data[0][field]=value
        with pytest.raises(ValueError):validate_table(data,{'a':0,'b':1})


def test_timing_coverage_and_correct_behavior_assertions():
    data=[dict(method=m,source_index=i,snr_db=s,repeat=r,seed=2001,tx_ms=2,rx_ms=3,total_ms=5) for m,i,s,r in itertools.product(METHODS[:3],range(0,100,11),SNRS,(0,1))]
    checks=[dict(method=r['method'],source_index=r['source_index'],snr_db=r['snr_db'],repeat=r['repeat'],max_abs_error=0) for r in data]
    validate_timing(data,checks)
    for bad in [checks[:-1],checks[:-1]+[checks[0]]]:
        with pytest.raises(ValueError):validate_timing(data,bad)
    broken=copy.deepcopy(checks);broken[0]['max_abs_error']=1e-3
    with pytest.raises(ValueError):validate_timing(data,broken)
    broken=copy.deepcopy(data);broken[0]['total_ms']=6
    with pytest.raises(ValueError):validate_timing(broken,checks)
