"""Join matched raw timing repeats; device wall time is accounted separately."""
from opt_common import *
from route_statistics_v2 import csvout
import collections

def main():
 txrole='benchmark_tx_identity_v2'
 for role in ['benchmark_rx',txrole,'benchmark_phy']:
  w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
  assert read(L/role/'completion.json')['ledger']['unresolved']==0
 a=read(L/'benchmark_rx/actual_native_identity.json');b=read(L/txrole/'actual_native_identity.json');assert a==b,'Same actual hardware, precision and CPU affinity required'
 rx=read(L/'benchmark_rx/rows.json');tx=read(L/txrole/'rows.json');phy=read(L/'benchmark_phy/rows.json');variants=read(L/'benchmark_inputs/variants.json');txmap={(v['variant'],v['case_id'],v['phase'],v['repeat']):v for v in tx};phymap={(v['input_grid'],v['case_id'],v['phase'],v['repeat']):v for v in phy};vm={v['name']:v for v in variants};full=[];summary=[]
 basefiles=cr(read(D/'p0_registration.json')['model_routes'])['files'];storage=sum(v['bytes']for v in basefiles)
 models=read(D/'model_storage_and_recovery.json')['models'];state_bytes=sum(v['state_tensor_bytes']for v in models if 'variant'not in v)
 mr=cr(read(D/'protocol_resolved_p0.json')['metrics']);receipt=cr(mr['config']['config']['metric_config']['weight_receipt']);efiles=[v for v in receipt['mapping']if v['sha256']in ['7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02','df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0']];assert len(efiles)==2
 for v in rx:
  variant=vm[v['variant']];grid=variant.get('input_grid','quarter');txname=v['variant']if grid!='quarter'else'original';key=(v['case_id'],v['phase'],v['repeat']);t=txmap[(txname,*key)];p=phymap[(grid,*key)]
  row=dict(v,tx_source_seconds=t['tx_source_seconds'],tx_PHY_seconds=p['tx_PHY_seconds'],rx_PHY_seconds=p['rx_PHY_seconds'],TX_total=t['tx_source_seconds']+p['tx_PHY_seconds'],RX_total=v['rx_source_image_seconds']+p['rx_PHY_seconds'],independent_termination_attempts=t['independent_termination_attempts'],length_checks=t['length_checks'],E_candidate_reconstructions=t['candidate_reconstructions'],E_candidate_LPIPS_calls=t['candidate_LPIPS_calls'],TX_component_reused_for_A_B_C=grid=='quarter'and v['variant']!='original',aggregation='Matched component times summed, not a concurrently observed end-to-end stopwatch')
  full.append(row)
 for name in vm:
  selected=[v for v in full if v['variant']==name and v['phase']=='measured'];assert len(selected)==96
  rxmem=read(L/'benchmark_rx'/(name+'_memory.json'));grid=vm[name].get('input_grid','quarter');txn=name if grid!='quarter'else'original';txmem=read(L/txrole/(txn+'_memory.json'));extra=vm[name].get('checkpoint',{}).get('bytes',0)
  if name=='E':extra=sum(v['bytes']for v in efiles)
  extra_state=258654988 if name=='A'else 2899456*4 if vm[name]['kind']=='C'else ''if name=='E'else 0
  for snr in [7,10,'pooled_core']:
   vs=selected if isinstance(snr,str)else[v for v in selected if v['snr_db']==snr]
   for component in ['TX_total','RX_total','tx_source_seconds','rx_source_image_seconds','tx_PHY_seconds','rx_PHY_seconds']:
    x=np.asarray([v[component]for v in vs]);summary.append(dict(variant=name,N=1024,snr_db=snr,component=component,sources=16,measured_repeats_per_source=3,measurements=len(x),mean_seconds=float(x.mean()),median_seconds=float(np.median(x)),p95_seconds=float(np.quantile(x,.95)),peak_TX_allocated_bytes=txmem['peak_allocated'],peak_RX_allocated_bytes=rxmem['peak_allocated'],peak_TX_reserved_bytes=txmem['peak_reserved'],peak_RX_reserved_bytes=rxmem['peak_reserved'],base_serialized_checkpoint_bytes=storage,extra_serialized_checkpoint_bytes=extra,base_uncompressed_state_tensor_bytes=state_bytes,extra_uncompressed_state_tensor_bytes=extra_state,E_storage_includes_AlexNet_and_LPIPS_calibration=name=='E',serialized_checkpoint_may_include_optimizer=True,mean_independent_terminations=float(np.mean([v['independent_termination_attempts']for v in vs])),mean_length_checks=float(np.mean([v['length_checks']for v in vs])),mean_E_candidate_reconstructions=float(np.mean([v['E_candidate_reconstructions']for v in vs])),B_single_output=vm[name]['kind']=='B',quality_evaluation_generation_repeats=3 if vm[name]['kind']=='B'else 1,shared_hardware=True,component_sum_not_single_stopwatch=component in ['TX_total','RX_total']))
 out=D/'timing_delivery';out.mkdir(exist_ok=False);csvout(out/'timing_raw.csv',full);csvout(out/'timing_and_compute.csv',summary)
 save(out/'identity.json',dict(status='COMPLETE',benchmark_protocol=pin(D/'benchmark_protocol.json'),actual_native_identity=pin(L/'benchmark_rx/actual_native_identity.json'),same_GPU_CPU_and_precision_verified=True,raw_inputs=[pin(L/x/'rows.json')for x in ['benchmark_rx',txrole,'benchmark_phy']],TX_pretrial_identity_failure_preserved=pin(D/'benchmark_tx_identity_correction.json'),model_files=basefiles,E_extra_weight_receipt=efiles,shared_environment_not_exclusive_hardware=True,E_full_simulation_and_LPIPS_in_TX=True,B_single_output_vs_three_generation_quality_repeats_separate=True,A_original_plus_anchor_both_stored=True,C_model_parameter_bytes=2899456*4,serialized_training_checkpoint_contains_more_than_inference_weights=True,initial_model_load_excluded_from_steady_state_latency_but_charged_in_resource_ledger=True,CPU_and_GPU_components_measured_separately_and_summed=True))
 print('TIMING_COMPLETE',len(full),len(summary))
if __name__=='__main__':main()
