"""Deterministic ARPC source registration; metadata only, no model imports."""
from pathlib import Path
import csv
import hashlib
import io
import json
import subprocess
import time

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
O = R / 'results/arpc_external_baseline_20261019_v1'
EXPECTED_HEAD = '3ac13a02feccbbe5f87ca2bc801c18ae07dd29a0'

def sha(b): return hashlib.sha256(b).hexdigest()
def desc(p):
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
def read_pin(d):
    p=Path(d['path']);b=p.read_bytes()
    assert sha(b)==d['sha256'] and len(b)==d.get('bytes',len(b)),p
    return json.loads(b)
def put(name,obj):
    p=O/name
    b=obj.encode() if isinstance(obj,str) else (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    with p.open('xb') as f:f.write(b)
    return desc(p)

def main():
    head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
    assert head==EXPECTED_HEAD,head
    assert not O.exists(),'Inspect an existing preparation rather than duplicate it'
    cal_reg=R/'results/submission_closure_20261016_v1/source_registration.json'
    eval_reg=R/'results/t1c_m10_test_20261016_v1/source_registration.json'
    cal=json.loads(cal_reg.read_text())['sources']
    reg=json.loads(eval_reg.read_text());full_eval=read_pin(reg['sources'])
    ev=[s for s in full_eval if s['population']=='common500']
    assert len(cal)==1000 and len(ev)==500
    for pool in (cal,ev):assert len({s['source_id'] for s in pool})==len(pool)
    def select(pool,salt):
        ranked=sorted(pool,key=lambda s:sha((salt+s['source_id']).encode()))[:100]
        result=[]
        for rank,s in enumerate(ranked):
            a=s['archive'];p=Path(a['path']);assert not p.is_symlink()
            assert desc(p)==a,(p,desc(p),a)
            result.append({'rank':rank,'source_id':s['source_id'],'source_index':s['source_index'],
                'selection_sha256':sha((salt+s['source_id']).encode()),'archive':a,
                'preprocessing_id':s['preprocessing_id']})
        return result
    cal100=select(cal,'ARPC_EXT_CAL_V1|');eval100=select(ev,'ARPC_EXT_EVAL_V1|')
    O.mkdir()
    rows=[]
    for population,ss in [('cal100',cal100),('external_common100_v1',eval100)]:
        for s in ss:
            rows.append({'population':population,'a1_member':int(population=='cal100' and s['rank']<8),
              **{k:v for k,v in s.items() if k!='archive'},'archive_path':s['archive']['path'],
              'archive_sha256':s['archive']['sha256']})
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest=put('source_id_manifest.csv',out.getvalue())
    sources=put('sources.json',{'cal100':cal100,'a1_first8':cal100[:8],'external_common100_v1':eval100})
    auth=put('execution_scope.json',{
        'schema':'ARPC_EXTERNAL_ROOT_SCOPE_20261019_V1','basis_HEAD':head,
        'user_instruction':'全部托管给你 直接运行到最后 我要看到推进到最后的结果',
        'interpretation':'Latest explicit full execution authorizes A0 through gated A3, overriding the package default first-batch stop only.',
        'upstream_commit':'f7480889175271a58c7e5f8e48e0b58f247ccf63',
        'stages':['A0','A1','A2a','A2b','A3'],'successor_requires_actual_stage_pass':True,
        'A1_caps':{'sources':8,'captions':8,'source_probability_traversals':8,'prefix_roundtrips':24,'reference_plus_rx_generations':48,'wireless_frames':0},
        'A2_caps':{'screen_sources':100,'clean_prefix_states':700,'strategies':84,'candidates_per_snr':2,'calibration_sources':1000,'logical_frames':18000,'engineering_wireless_frames':24},
        'A3_caps':{'sources':100,'SNRs':[4,10,19],'noises_per_source':3,'logical_frames':900,'bootstrap_replicates':10000,'bootstrap_seed':2026100701},
        'training':False,'extra_baseline':False,'full_common500_ARPC':False,'backup_deferred':True,
        'source_manifest':manifest,'source_assets':sources,'registration_inputs':[desc(cal_reg),desc(eval_reg),reg['sources']],
        'pool_sizes':{'cal1000':1000,'common500':500},'source_selection_uses_metrics_or_images':False,
        'metadata_observed_unix':time.time()})
    put('actual_execution_counts.json',{'status':'A0_METADATA_ONLY','caption_calls':0,'tokenizer_calls':0,'source_probability_traversals':0,'reference_image_generations':0,'independent_receiver_generations':0,'wireless_frames':0,'training_updates':0})
    print(json.dumps({'scope':auth,'sources':sources,'manifest':manifest,'a1_source_indices':[s['source_index'] for s in cal100[:8]]}))

if __name__=='__main__':main()
