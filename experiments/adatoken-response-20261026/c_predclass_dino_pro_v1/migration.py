"""Device migration contract. Only hardware changes; no foreign feature/scalar reuse."""
import copy
from contracts import check_identity,PRO_HOST

def check_registration(r,host):
    assert host==r['required_host']==PRO_HOST
    assert r['calibration_only']is True and r['source_reference_calls']==1000
    assert r['cross_device_feature_reuse']is False and r['cross_device_scalar_reuse']is False
    assert r['original_registration']['sha256']=='ef5dd3f162bea7fb226a4b0d6b8be465b91d783a5807f7d028252d2198635bf5'
    assert r['caps']==dict(model_load=1,source_reference_dino=1000,reconstruction_dino=5061)
    assert r['max_seconds']==14400 and r['budget']['GPU_closed_seconds']+14400<=96*3600
    assert r['no_formal_data']is True
    return True

def check_migrated_identity(previous,metadata,runtime,r):
    check_registration(r,runtime['host'])
    assert runtime['device']==r['cuda_admission']
    # Validate identical math, weights, preprocessing, packages and precision.
    # Permit only explicitly registered host/device changes. No prior scores or
    # reference features enter this execution; all are computed on one device.
    compatible=copy.deepcopy(runtime)
    compatible['host']=previous['host'];compatible['device']=previous['device']
    return check_identity(previous,metadata,compatible)
