"""Evidence-only development overview; never rerun scoring or choose candidates."""
from opt_common import *
from route_statistics_v2 import csvout,CORE,METRICS
import csv,collections

def loadrows(path):
 with Path(path).open()as f:return list(csv.DictReader(f))
def main():
 out=D/'development_summary';out.mkdir(exist_ok=False);routes=[];own=[];paired=[];per_source=[];inputs=[]
 families=[('A','fit_A_delivery'),('B','fit_B_delivery'),('C','fit_C_delivery'),('D','fit_D_delivery'),('E','fit_E_delivery')]
 for route,folder in families:
  root=D/folder;rows=loadrows(root/'summary.csv');inputs.append(pin(root/'summary.csv'));values=collections.defaultdict(dict)
  for v in rows:values[(v['method'],v['variant'],int(v['N']),int(v['snr_db']))][v['metric']]=float(v['mean'])
  gates=read(root/'gate.json')['gates']
  for variant in sorted({v['variant']for v in rows}-{'original'}):
   g=gates.get(variant,{})
   for method in ['NEST','WHOLE']:
    if not all((method,variant,*c)in values for c in CORE):continue
    ratios={m:float(np.mean([values[(method,variant,*c)][m]/values[(method,'original',*c)][m]for c in CORE]))for m in ['lpips_alex','dreamsim']};psnr=float(np.mean([values[(method,variant,*c)]['psnr_db']-values[(method,'original',*c)]['psnr_db']for c in CORE]))
    own.append(dict(route=route,method=method,candidate=variant,core_J=float(np.mean(list(ratios.values()))),normalized_lpips=ratios['lpips_alex'],normalized_dreamsim=ratios['dreamsim'],mean_core_psnr_delta=psnr,selection_population=method=='NEST'))
    if method!='NEST':continue
    status='NO_GO'
    if route=='D':status='CONFIRM_PASS_FINAL_FROZEN'if variant=='D_token'else'NOT_SELECTED_FINER_GRID_WON_FIT'
    elif route=='A'and variant=='A2':status='VERIFICATION_ONLY_NOT_SELECTABLE'
    elif route=='C':status='CONTROL_NOT_SELECTABLE'if 'no_content'in variant else'NO_GO_FROZEN_1000_GUARD_FAILURE'if variant.endswith('_1000')else'NOT_SELECTED_CHECKPOINT'
    routes.append(dict(own[-1],status=status,fit_core_gate=g.get('fit_pass','see gate'),guard_violations=json.dumps(g.get('guard_violations',[])),source_report=str(root.relative_to(R))))
  for v in loadrows(root/'paired_deltas.csv'):paired.append(dict(population='FIT100',route=route,**v))
  for v in loadrows(root/'per_source_metrics.csv'):per_source.append(dict(population='FIT100',route=route,**v))
 # All guards and confirmation remain distinct populations/segments, never silently averaged.
 for route,folder,pop in [('C','fit_C_complete_delivery','FIT100_SELECTED_ALL7'),('D','fit_D_complete_delivery','FIT100_SELECTED_ALL7'),('D','confirm_D_delivery','CONFIRM100')]:
  root=D/folder;inputs.append(pin(root/'gate.json'))
  for v in loadrows(root/'paired_deltas.csv'):paired.append(dict(population=pop,route=route,**v))
  for v in loadrows(root/'per_source_metrics.csv'):per_source.append(dict(population=pop,route=route,**v))
 csvout(out/'route_summary.csv',routes);csvout(out/'own_baseline_normalized.csv',own);csvout(out/'paired_deltas.csv',paired);csvout(out/'per_source_metrics.csv',per_source)
 save(out/'identity.json',dict(inputs=inputs,all_candidates_preserved=True,WHOLE_uses_own_baseline=True,second_C_seed='NOT_RUN_CONDITIONAL_PRIMARY_FAILED_GUARD',development_not_blind=True));print(routes)
if __name__=='__main__':main()
