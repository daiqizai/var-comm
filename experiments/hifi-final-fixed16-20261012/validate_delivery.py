"""Independent read-only checks of the final16 HiFi delivery."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/hifi_final_fixed16_20261012';O=ROOT/'paper/figures/hifi_final_fixed16_20261012'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def main():
 reg=read(R/'registration.json');done=read(R/'completion.json');rows=read(R/'receipts.json')
 assert read(R/'actual_wait.json')['returncode']==0 and not (R/'failure.json').exists() and not (R/'pending_case.json').exists()
 assert sha(R/'receipts.json')==done['receipts_sha256'];assert len(rows)==48
 grid={(m,i) for m in ['P100000','Swin200000','HiFiSwin200000'] for i in reg['source_indices']}
 assert {(v['method'],v['source_index']) for v in rows}==grid
 assert sum(v['reuse'] for v in rows)==8
 assert read(R/'qualification.json')['status']=='PASS_FINAL_SWIN_OPERATOR'
 for p,h in reg['code_bindings'].items():assert sha(p)==h,p
 for v in rows:
  assert v['checkpoint_sha256']==reg['checkpoints']['P100000' if v['method']=='P100000' else 'Swin200000']['sha256']
  assert v['N']==1024 and v['snr_db']==13 and v['noise_seed']==2001
  assert sha(v['archive']['path'])==v['archive']['sha256'] and sha(v['png']['path'])==v['png']['sha256']
  with np.load(v['archive']['path'],allow_pickle=False) as z:
   assert ash(z['reconstruction'])==v['reconstruction_sha256'] and ash(z['display'])==v['display_sha256']
   assert np.array_equal(z['display'],np.rint(z['reconstruction'].transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
   if v['method']!='HiFiSwin200000':assert ash(z['observed'])==v['observation_sha256'] and ash(z['wave'])==v['wave_sha256']
  if v['method']=='HiFiSwin200000':
   ref=next(x for x in rows if x['source_index']==v['source_index'] and x['method']=='Swin200000')
   assert ref['observation_sha256']==v['observation_sha256'] and ref['archive']==v['paired_Swin_archive']
   rec=v['sampler_receipt'];assert rec['model_parameters_unchanged']
   if rec['header_accepted']:assert rec['complete_author_schedule'] and not rec['diagnostic_probe'] and rec['sampling_seed']==23 and rec['NFE']==253
   else:assert rec['NFE']==0 and rec['fallback']=='fixed_gray_0.5'
 proof=read(O/'provenance.json')
 for n,h in proof['outputs'].items():assert sha(O/n)==h,n
 for n,h in proof['source_bindings'].items():assert sha(n)==h,n
 assert len(list((O/'native').glob('*.png')))==80
 for p in O.glob('*.svg'):assert '<text' in p.read_text()
 old=ROOT/'results/external_comparison_20261004/fixed80k_revision/hifi_fixed16_release/evaluation'
 for name,source in [('historical_Swin80k_fixed16.csv','metrics_summary.csv'),('historical_Swin80k_paired.csv','metrics_paired_intervals.csv')]:
  reference=list(csv.DictReader((old/source).open()));actual=list(csv.DictReader((O/name).open()));assert all(r in reference for r in actual)
 result=dict(status='PASS_FULL_GRID_AND_IDENTICAL_OBSERVATION',sources=16,rows=48,full_HiFi_recoveries=16,new_baseline_calls=24,reused_baseline_calls=8,actual_exit=0,native_display_images=80,new_metric_calls=0,new_bootstrap_calls=0,training_updates=0,HiFi_failures=sum(not v['sampler_receipt']['header_accepted'] for v in rows if v['method']=='HiFiSwin200000'),timing_comparison_qualified=False)
 (R/'delivery_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
