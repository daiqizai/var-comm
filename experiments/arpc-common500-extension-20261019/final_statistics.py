"""Combine closed 100+400 records and reuse the registered 500-source draw bank."""
import argparse,collections,math,time
import numpy as np
from stage_support import *
from metric_data import means,DIRECTIONS
OUT=O/'delivery_v1'
def summary_function():
    p=PE/'a3_metrics_v1/final_stats.py';old=read(PO/'a3_metrics_v1/registration.json');pin=next(d for d in old['source_files']if d['path']==str(p));verify(pin)
    # Import definitions only; the old main (which draws an N=100 bank) is never called.
    s=p.read_text();s=s.replace('from data import METRICS,BASELINES,DIRECTIONS,means','from common import METRICS,BASELINES\nfrom metric_data import DIRECTIONS,means')
    ns={'__name__':'_original_arpc_summary_definitions'};exec(compile(s,str(p),'exec'),ns);return ns['summarize']
def conclusion(r):
    lo,hi=float(r['ci_low']),float(r['ci_high']);lower=r['direction']=='lower'
    if lo<=0<=hi:return 'INTERVAL_INCLUDES_ZERO'
    return 'NeST_FAVORED'if (hi<0 if lower else lo>0)else'ARPC_FAVORED'
def main():
    assert not OUT.exists()
    closepin=closed('metrics');registered=read(O/'registration.json')
    for p in registered['inputs']+[registered['population'],registered['scope']]:verify(p)
    population=read(registered['population']['path']);sources=sorted(population['common500'],key=lambda s:s['source_index'])
    assert [s['source_index']for s in sources]==list(range(500));ids=[s['source_id']for s in sources]
    byid={s['source_id']:s['source_index']for s in sources};oldids={s['source_id']for s in population['original100']};newids=set(ids)-oldids
    old=csvrows(PO/'a3_metrics_v1/arpc_source_noise_means.csv');newlogical=csvrows(O/'metrics_v1/logical_scores.csv')
    new=means(newlogical,newids);arpc=old+new
    assert len(arpc)==10500 and {r['source_id']for r in old}==oldids
    for r in arpc:assert byid[r['source_id']]==int(r['source_index'])and int(r['noise_count'])==3
    baseline_done=read(O/'baseline_inputs_v1/completion.json')
    for d in baseline_done['inputs']+baseline_done['outputs']:verify(d)
    baseline=csvrows(O/'baseline_inputs_v1/source_means.csv');allrows=baseline+arpc;assert len(allrows)==52500
    done=read(O/'metrics_v1/score_completion.json');assert done['logical_frames']==3600 and done['unresolved']==0
    for d in done['outputs']:verify(d)
    oldreg=read(PO/'a3_metrics_v1/registration.json');newreg=read(O/'metrics_v1/registration.json')
    for k in ('original_four_metrics','metric_config','supplemental_registration'):assert oldreg[k]==newreg[k],k
    newidentity=read(O/'metrics_v1/supplemental_identity.json');oldidentity=read(PO/'a3_metrics_v1/supplemental_identity.json')
    for k in ('original_registration','model_metadata','protocol','FP32','TF32','seed'):assert newidentity[k]==oldidentity[k],k
    scope=read(registered['scope']['path']);bank=scope['bootstrap']['indices'];verify(bank)
    draws=np.load(bank['path'],allow_pickle=False);assert draws.shape==(10000,500)and draws.dtype==np.int64
    h=hashlib.sha256(str(draws.dtype).encode()+str(draws.shape).encode()+draws.tobytes()).hexdigest();assert h==scope['bootstrap']['array_sha256']
    OUT.mkdir();put(OUT/'analysis_registration.json',{'created_unix':time.time(),'population':registered['population'],'bootstrap':scope['bootstrap'],'metrics_closure':closepin,'source_files':[desc(E/'final_statistics.py')],'inputs':[desc(p)for p in (O/'metrics_v1/logical_scores.csv',PO/'a3_metrics_v1/arpc_source_noise_means.csv',O/'baseline_inputs_v1/source_means.csv')],'new_rng_draws':0,'primary_unit':'source','source_index_order':list(range(500))})
    summary,paired=summary_function()(allrows,ids,draws);assert len(summary)==105 and len(paired)==21
    previous=csvrows(PO/'a3_metrics_v1/nest_minus_arpc_paired.csv');prev={(int(r['snr_db']),r['metric']):r for r in previous}
    comparison=[]
    for r in paired:
        prior=prev[r['snr_db'],r['metric']]
        comparison.append({'snr_db':r['snr_db'],'metric':r['metric'],'direction':r['direction'],'old100_mean':prior['mean'],'old100_ci_low':prior['ci_low'],'old100_ci_high':prior['ci_high'],'new500_mean':r['mean'],'new500_ci_low':r['ci_low'],'new500_ci_high':r['ci_high'],'old100_conclusion':conclusion(prior),'new500_conclusion':conclusion(r),'conclusion_changed':conclusion(prior)!=conclusion(r)})
    oldbaselines=[r for r in csvrows(PO/'a3_metrics_v1/all_method_source_means.csv')if r['method']!='ARPC_ADAPTED']
    baseline_lookup={(r['method'],int(r['snr_db']),r['metric'],r['source_id']):float(r['source_mean'])for r in baseline}
    drifts=[]
    for r in oldbaselines:
        key=(r['method'],int(r['snr_db']),r['metric'],r['source_id']);v=baseline_lookup[key];delta=v-float(r['source_mean'])
        if abs(delta)>1e-12:drifts.append({**r,'current_source_mean':v,'difference':delta})
    logical=csvrows(PO/'a3_metrics_v1/logical_scores.csv')+newlogical
    assert len(logical)==4500 and collections.Counter(int(r['snr_db'])for r in logical)=={4:1500,10:1500,19:1500}
    assert len({(r['source_id'],int(r['snr_db']),int(r['noise_seed']))for r in logical})==4500
    for r in logical:assert byid[r['source_id']]==int(r['source_index'])
    receivers=read(PO/'a3_rx_v1/logical_receivers.json')+read(O/'rx_v1/logical_receivers.json');assert len(receivers)==4500
    distribution=collections.Counter((int(r['snr_db']),int(r['endpoint']),r['actual_rx_endpoint'],r['actual_status'],r['receiver_status'])for r in receivers)
    endpoints=[{'snr_db':k[0],'tx_endpoint':k[1],'rx_endpoint':k[2],'actual_link_status':k[3],'receiver_status':k[4],'logical_count':v}for k,v in sorted(distribution.items(),key=lambda x:repr(x[0]))]
    for name,rows in [('arpc_source_means.csv',arpc),('all_method_source_means.csv',allrows),('common500_summary.csv',summary),('nest_minus_arpc_paired.csv',paired),('original100_vs_common500.csv',comparison),('logical_scores.csv',logical),('endpoint_status_distribution.csv',endpoints)]:write_csv(OUT/name,rows)
    put(OUT/'logical_receivers.json',receivers)
    put(OUT/'old100_baseline_version_check.json',{'changed_source_cells':len(drifts),'differences':drifts,'note':'If nonzero, old100-to-new500 conclusions involve baseline version repair as well as population expansion.'})
    newlink=read(O/'link_v1/completion.json');newrx=read(O/'rx_v1/completion.json');tx=read(O/'source_v1/completion.json')
    compute={'new_source_calls':tx['counts'],'new_endpoint_streams':tx['streams'],'new_logical_events':newlink['logical_frames'],'new_encoded_waveforms':newlink['encoded_waveforms'],'new_header_decodes':newlink['header_decodes'],'new_body_decodes':newlink['body_decodes'],'new_unique_received_states':newrx['actual_unique_states'],'new_rx_prefix_calls':newrx['new_prefix_calls'],'new_rx_image_renders':newrx['new_image_renders'],'reused_rx_states':newrx['reused_states'],'new_metric_pairs':done['unique_pairs'],'new_logical_scored_frames':3600,'new_metric_calls':done['counts'],'within_new_population_logical_score_reuse':3600-done['unique_pairs'],'original_sources_reused':100,'original_events_and_scores_reused':900,'original_streams_reused':700,'baseline_source_metric_cells_reused':42000,'new_baseline_scores':0,'bootstrap_bank_reused':bank,'new_rng_draws':0,'full_population':500,'full_events':4500,'frames_per_snr':{'4':1500,'10':1500,'19':1500}}
    put(OUT/'actual_compute_and_reuse.json',compute)
    put(OUT/'statistics_protocol.json',{'source_order':ids,'source_index_order':list(range(500)),'source_aggregation':'Mean of three noise realizations within each source, then equal source weight','bootstrap':scope['bootstrap'],'intervals':'Pointwise 95% percentile, quantile linear; no multiplicity adjustment','paired':'NeST minus ARPC per source before resampling; never marginal CI subtraction','noise':'ARPC/NeST use6201/2/3; historical P/Swin/BPG use2001/2/3. Source-paired, not common-noise across all five.','old100_bank_reused':False,'new_rng_draws':0,'metric_algorithms_unchanged':True,'native_host_change_admission':desc(O/'host3781_admission_v2/admission.json')})
    files=sorted(p for p in OUT.iterdir()if p.is_file())
    put(OUT/'completion.json',{'status':'COMMON500_FIVE_METHOD_STATISTICS_COMPLETE','sources':500,'logical_frames':4500,'summary_cells':105,'paired_cells':21,'new_rng_draws':0,'unresolved':0,'outputs':[desc(p)for p in files]})
if __name__=='__main__':main()
