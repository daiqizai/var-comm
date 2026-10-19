"""Once-only population and reusable-evidence registration; no model/channel calls."""
import time,subprocess
from common import *
def main():
    import numpy as np
    assert not O.exists() and not L.exists(),'Inspect existing extension before any repeat'
    source_reg=R/'results/t1c_m10_test_20261016_v1/source_registration.json'
    sr=read(source_reg);verify(sr['sources'])
    full=[s for s in read(sr['sources']['path']) if s['population']=='common500']
    olds=PO/'a3_source_v1/sources.json';old=read(olds)['new_sources']
    full,remaining=partition(full,old)
    cal=read(PO/'a2b_source_v1/sources.json')['full_sources']
    assert not {s['source_id'] for s in full}&{s['source_id'] for s in cal}
    archive_checks=[]
    for s in full:
        verify(s['archive'])
        with np.load(s['archive']['path'],allow_pickle=False)as z:
            pixels=z['pixels'];assert pixels.shape==(3,256,256)and pixels.dtype==np.uint8
            archive_checks.append({'source_id':s['source_id'],'source_index':s['source_index'],'archive':s['archive'],'preprocessing_id':s['preprocessing_id'],'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()})
    closed=[]
    for stage in ('a3_source_v1','a3_link_v1','a3_rx_v1','a3_metrics_v1'):
        p=PO/stage/'root_closure_review.json';c=read(p)
        assert c['all_processes_gone'] and c['unresolved']==0
        for d in c['controls']+c['outputs']:verify(d)
        closed.append(desc(p))
    frozen=PO/'a2b_metrics_v1/frozen_policies.json'
    assert read(frozen)['policies']==POLICIES
    catalogue=read(PO/'a3_source_v1/stream_catalogue.json')['sources']
    assert {s['source_id'] for s in catalogue}=={s['source_id'] for s in old}
    for s in catalogue:
        assert len(s['streams'])==7
        for e in s['streams']:verify(e['wire'])
    logical=read(PO/'a3_rx_v1/logical_receivers.json')
    scores=csvrows(PO/'a3_metrics_v1/logical_scores.csv')
    def keys(rows):return {(r['source_id'],int(r['source_index']),int(r['snr_db']),int(r['noise_seed']))for r in rows}
    expected={(s['source_id'],s['source_index'],d,n)for s in old for d in (4,10,19)for n in (6201,6202,6203)}
    assert len(logical)==len(scores)==900 and keys(logical)==keys(scores)==expected
    bootstrap_report=R/'results/n1024_consistency_20261019_v1/completion_report_v1/actual_compute_and_reuse.json'
    bank=read(bootstrap_report)['bootstrap'];verify(bank['indices'])
    indices=np.load(bank['indices']['path'],allow_pickle=False)
    assert indices.shape==(10000,500)and indices.dtype==np.int64 and indices.min()>=0 and indices.max()<500
    assert bank['seed']==2026100701 and hashlib.sha256(indices.tobytes()).hexdigest()==bank['array_sha256']
    baseline_paths={'nest':R/'results/n1024_consistency_20261019_v1/repair_v1/delivery/source_means_common500.csv','main':R/'results/entropy_main_common500_20261013/analysis_v1/source_means.csv','supplement':R/'results/quality_metrics_supplement_20261014_v1/common500/source_noise_means.csv'}
    old_inputs=[PO/'a3_source_v1/stream_catalogue.json',PO/'a3_rx_v1/logical_receivers.json',PO/'a3_metrics_v1/logical_scores.csv',PO/'a3_metrics_v1/arpc_source_noise_means.csv',PO/'a3_metrics_v1/matched100_summary.csv',PO/'a3_metrics_v1/nest_minus_arpc_paired.csv',PO/'a3_metrics_v1/four_metric_identity.json',PO/'a3_metrics_v1/supplemental_identity.json']
    pins=[desc(source_reg),sr['sources'],desc(olds),desc(frozen),desc(bootstrap_report),bank['indices']]+closed+[desc(p)for p in old_inputs]+[desc(p)for p in baseline_paths.values()]
    head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
    O.mkdir();L.mkdir()
    population={'common500':full,'original100':old,'remaining400':remaining,'source_index_order':list(range(500)),'renumbered':False,'selection_uses_metrics':False}
    put(O/'population.json',population);put(O/'source_pixels_verified.json',{'sources':archive_checks})
    write_csv(O/'remaining400.csv',[{'source_id':s['source_id'],'source_index':s['source_index'],'archive_path':s['archive']['path'],'archive_sha256':s['archive']['sha256'],'preprocessing_id':s['preprocessing_id']}for s in remaining])
    scope={'schema':'ARPC_COMMON500_EXTENSION_AUTHORIZATION_V1','authorized_by':'User explicitly authorizes ARPC common500 extension and final checks/commit/push','basis_head':head,'created_unix':time.time(),'original_A3_preserved':True,'new_population':400,'reuse_population':100,'full_population':500,'frozen_policies':POLICIES,'N':1024,'head_symbols':68,'body_symbols':956,'Es':2,'noise_seeds':[6201,6202,6203],'counter_namespace':3000000,'source_index_preserved':True,'limits':{'caption':400,'tokenizer':400,'TX_probability':400,'endpoint_streams':2800,'new_logical_events':3600,'header_decode':3600,'body_decode':3600,'new_RX_states':3600,'new_score_pairs':3600,'source_metric_references':400,'new_bootstrap_draws':0,'new_baseline_scores':0,'training':0,'TX_clean_renders':0},'bootstrap':bank,'baseline_inputs':{k:desc(v)for k,v in baseline_paths.items()},'runtime':str(L),'output':str(O),'code':str(E),'scope_excludes':['N2048 expansion','SNR grid changes','calibration','training','other baseline inference','timing','full manuscript rewrite'],'engineering_resume':'Preserve every attempt; resume only uncompleted work after diagnosing failed or unresolved reservations. Never silently repeat successful calls.'}
    put(O/'scope.json',scope)
    reg={'schema':'ARPC_COMMON500_EXTENSION_POPULATION_REGISTRATION_V1','status':'INPUTS_REGISTERED_SCIENCE_NOT_STARTED','created_unix':time.time(),'repository_head':head,'source_files':[desc(E/n)for n in ('common.py','inventory.py','README.md')],'inputs':pins,'population':desc(O/'population.json'),'remaining400':desc(O/'remaining400.csv'),'verified_pixels':desc(O/'source_pixels_verified.json'),'scope':desc(O/'scope.json'),'reuse':{'sources':100,'logical_events':900,'streams':700},'new_scientific_calls':0,'execution_registration_required':True}
    put(O/'registration.json',reg)
    print(json.dumps({'registration':desc(O/'registration.json'),'remaining400':desc(O/'remaining400.csv'),'original_reused_events':900,'new_events_pending':3600}))
if __name__=='__main__':main()
