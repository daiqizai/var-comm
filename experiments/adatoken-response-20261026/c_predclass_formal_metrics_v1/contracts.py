"""Formal grid, closure and cache checks; no model, tensor or image execution."""
import math
POPULATIONS={'common500':(500,(6201,6202,6203)),'kodak24':(24,(2001,2002,2003))}
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
POLICY_SHA='68d12e9b19006e8509b3ae5a62132e73044b88ad29ca041f96b436144923ae8c'


def validate_plan(plan,population):
    count,noises=POPULATIONS[population]
    assert plan['population']==population and plan['frozen_policy']['sha256']==POLICY_SHA
    sources=plan['sources'];frames=plan['frames']
    assert len(sources)==count and {s['source_index']for s in sources}==set(range(count))
    assert len({s['source_id']for s in sources})==count
    si={s['source_index']:s for s in sources}
    expected={(i,s,n)for i in range(count)for s in(4,10,19)for n in noises}
    keys=[(j['source_index'],j['snr_db'],j['noise_seed'])for j in frames]
    assert len(keys)==count*9 and len(set(keys))==len(keys)and set(keys)==expected
    for j in frames:
        s=si[j['source_index']]
        assert j['population']==population and j['source_id']==s['source_id']
        assert j['method']=='ADA_PREDCLASS_FIXEDFRAME'
        assert j['reference_sha256']==s['reference_sha256']
        assert type(j['reuse_old'])is bool and type(j['accepted'])is bool
        assert type(j['received_prefix'])is int and 0<=j['received_prefix']<=256
        assert len(j['pixel_sha256'])==64
        assert j['source_status']in('ACCEPTED_SOURCE','FIXED_GRAY_SOURCE_REJECT','FIXED_GRAY_PHY_REJECT')
        assert j['accepted']==(j['source_status']=='ACCEPTED_SOURCE')
        if j['image']is None:
            assert j['source_status']=='FIXED_GRAY_PHY_REJECT'and not j['accepted']
        if j['reuse_old']:
            assert set(j['old_values'])==set(METRICS)
            assert j['old_metric_identity_verified']is True
            assert all(type(v)in(int,float)and math.isfinite(v)for v in j['old_values'].values())
    return count*9


def validate_closure(wait,completion,gone,audit=None):
    assert wait['actual_wait']is True and type(wait['returncode'])is int and wait['returncode']==0
    assert gone(wait['process'])and gone(wait['parent_process'])
    assert not any(wait.get(k,False)for k in('timed_out','terminated','killed'))
    assert wait.get('metadata_error')is None
    assert completion['status']in('COMPLETE','COMPLETE_PHY_AWAITING_SOURCE_RX')
    ledger=completion.get('ledger',completion.get('packet_ledger'))
    if ledger is not None:assert ledger['unresolved']==0
    if audit is not None:
        assert audit['status']=='PASS'and not audit['violations']
        assert gone(audit['process'])and gone(audit['parent_process'])


def validate_metric_identity(identity,resource,frozen_config,loader):
    assert identity['config']==frozen_config and identity['loader']==loader
    assert identity['torch']==resource['torch']and identity['cuda']==resource['cuda']
    assert identity['pci'].lower()==resource['pci'].lower()
    assert identity['fp32']is True and identity['tf32']is False
    assert identity['batch']==1 and identity['threads']==6 and identity['interop']==2
