"""Score only 400 newly received sources; no old baseline calls or bootstrap draws."""
import argparse,time
from stage_support import *
RUN=L/'metrics_v1';OUT=O/'metrics_v1';REG=OUT/'registration.json'
def prepare():
    import numpy as np
    from metric_data import group
    q=qualification();rx=closed('rx');assert not RUN.exists()and not OUT.exists()
    rows=read(O/'rx_v1/logical_receivers.json');sources=read(O/'source_v1/sources.json')['new_sources']
    gray=hashlib.sha256(np.full((3,256,256),.5,dtype=np.float32).tobytes()).hexdigest();plan={'sources':group(rows,sources,gray)}
    # Population is disjoint by source identity. Within each source, exact pixels
    # are scored once and every original logical frame remains represented.
    pair_keys=set()
    for s in plan['sources']:
        verify(s['archive'])
        for pair in s['pairs']:
            if pair['image']:verify(pair['image'])
            key=(s['archive']['sha256'],pair['image_float_sha256']);assert key not in pair_keys;pair_keys.add(key)
    prior=PO/'a2_metrics_v1/registration.json';old=read(prior);a3=read(PO/'a3_metrics_v1/registration.json')
    for d in old['source_files']+a3['source_files']:verify(d)
    original=R/'experiments/entropy-partial-mismatch-20261010/scripts/h800_ep_pilot_metrics_v1.py'
    from metrics_math import SHA
    assert desc(original)['sha256']==SHA
    supplemental=R/'results/quality_metrics_supplement_20261014_v1/common500/registration.json';sr=read(supplemental)
    assert desc(supplemental)['sha256']=='191f10cce3ac20000583f70edfbe74e83ec0d22272e403ea0b683eed484cfb02'
    for k in ('loader','metric_assets','dependency_overlay'):verify(sr[k])
    resource=old['resource'];gpu=next(g for g in gpus()if g['uuid']==resource['gpu']['uuid'])
    assert gpu['free_mib']>=8192 and gpu['utilization']<=25
    assert read(prior)['host']==os.uname().nodename
    RUN.mkdir();OUT.mkdir();put(OUT/'pair_plan.json',plan)
    code=[desc(E/n)for n in ('common.py','stage_support.py','execute_stage_v1.py','metric_data.py','metrics_math.py','metrics_score.py','metrics_control.py')]
    reg={'schema':'ARPC_COMMON500_REMAINING400_SEVEN_METRICS_V1','host':os.uname().nodename,'created_unix':time.time(),'source_files':old['source_files']+a3['source_files']+code,'inputs':q+[rx,desc(O/'rx_v1/completion.json'),desc(O/'rx_v1/logical_receivers.json'),desc(O/'source_v1/sources.json'),desc(prior),desc(original),desc(supplemental),desc(OUT/'pair_plan.json')]+[sr[k]for k in ('loader','metric_assets','dependency_overlay')],'original_four_metrics':desc(original),'metric_config':old['metric_config'],'resource':resource,'startup_free_mib':8192,'actual_metric_resource':{'startup_free_mib':8192,'running_free_mib':2048,'allocator_gib':12,'numeric_flags':'Original FP32 TF32off; threads6/2; sequential models'},'pair_plan':desc(OUT/'pair_plan.json'),'supplemental_registration':desc(supplemental),'supplemental_overlay_root':a3['supplemental_overlay_root'],'absolute_deadline_unix':time.time()+16200,'limits':{'source_references':400,'new_pairs':3600,'logical_frames':3600,'new_baseline_scores':0,'training':0,'bootstrap_draws':0},'stage_seconds':{'score':14400},'phase_wall_seconds':15000,'selection':False}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    r=read(REG);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['resource']['cuda_admission']['visible_index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');env.pop('PYTHONPATH',None)
    run_stages('metrics',expected,[('score',[str(PY),'-B',str(E/'metrics_score.py'),'--expected-registration',expected])],env,r['stage_seconds'],gpu=r['resource']['gpu'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','run']);p.add_argument('--expected-registration');a=p.parse_args();prepare()if a.role=='prepare'else run(a.expected_registration)
