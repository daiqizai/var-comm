"""Select immutable existing CSV rows and trace metric/source identities; no statistics."""
from pathlib import Path
import json,csv,hashlib,subprocess,base64
W=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent
O=W/'paper/figures_redesign_v2';O.mkdir(exist_ok=True)
for d in ['plot_inputs','review','panels']:(O/d).mkdir(exist_ok=True)
HEAD='24fa1ecf3a6a959e6023d47ff460df41a711d30f'
R='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM'
B='/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011'
CPY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def rows(p):
 with p.open(encoding='utf8',newline='')as f:return list(csv.DictReader(f))
def csvout(p,x):
 with p.open('w',encoding='utf8',newline='')as f:
  c=csv.DictWriter(f,fieldnames=list(x[0]),lineterminator='\n');c.writeheader();c.writerows(x)
def pin(p):return dict(path=p.relative_to(W).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
style=read(O/'figure_style_v2.json');save(O/'figure_style_v2.json',style)
geometry=read(W/'paper/figures_redesign_v1/template_geometry.json');geometry['revalidated_head']=HEAD;save(O/'template_geometry.json',geometry)
methods=[dict(id=k,label=l,**style['methods'][l],markersize=s,fill='none' if k=='P100000' else 'full',checkpoint=c)for k,l,s,c in [
 ('EC_VAR_PARTIAL','NeST-Com (entropy-coded)',3.6,'Frozen policy; corrected m10 at19dB'),('P100000','Latent-space JSCC',4.2,'Final100k'),('Swin200000','SwinJSCC (adapted)',4.0,'Final200k'),('BPG_ADAPTIVE','Adaptive BPG + LDPC',3.6,'Frozen adaptive resolution')]]
save(O/'methods.json',methods)
resolved=dict(font_priority=style['font_fallback'],fig2_width_mm=geometry['mm']['TEXTWIDTH'],fig2_height_mm=55,fig3_width_mm=geometry['mm']['COLUMNWIDTH'],fig3_height_mm=50,
 metrics=read(W/'paper/figures_redesign_v1/style.json')['metrics'],fig2_xlim=[2.8,20.2],fig3_limits={'psnr_db':[-.08,1.55],'dreamsim':[-.002,.041]},fig3_ticks={'psnr_db':[0,.5,1,1.5],'dreamsim':[0,.01,.02,.03,.04]},dpi=600,
 changes=['Latent-space JSCC uses hollow square4.2pt and NeST-Com circle3.6pt to retain10dB LPIPS outlines; all locations exact','Long panel title wraps to two lines; common top title baseline; no repeated y label','Figure3 intervals are copied direct paired intervals; DreamSim endpoints swap on negation'])
save(O/'resolved_style.json',resolved)
paths={
 'main':'results/entropy_main_common500_20261013/analysis_v1/summary_all_methods.csv',
 'supp':'results/quality_metrics_supplement_20261014_v1/common500/summary.csv',
 't1c':'results/t1c_m10_test_20261016_v1/delivery/summary_19dB.csv',
 'entropy_pair':'results/entropy_main_common500_20261013/analysis_v1/paired.csv',
 'supp_pair':'results/quality_metrics_supplement_20261014_v1/common500/paired_source_intervals.csv',
 't1c_pair':'results/t1c_m10_test_20261016_v1/delivery/paired_19dB.csv',
 'raw_pair':'results/main_raw64_20261007/final_common500_r6/paired.csv',
 'raw_freeze':'results/main_raw64_20261007/final_common500_r6/evidence/unified_holdout_final_freeze.json',
 'raw_amendment':'results/main_raw64_20261007/final_common500_r6/evidence/identity_amendment.json',
 'supp_identity':'results/quality_metrics_supplement_20261014_v1/common500/evidence_v1/common500_metrics_v1/metric_runtime_identity.json',
 'supp_registration':'results/quality_metrics_supplement_20261014_v1/common500/registration.json',
 't1c_identity':'results/t1c_m10_test_20261016_v1/execution_evidence/metrics/supplemental_identity.json',
 'endpoints':'results/t1c_m10_test_20261016_v1/delivery/actual_fallback_counts.csv',
 'status':'RESEARCH_STATUS.md','manuscript':'paper/manuscripts/nestcom_wcl_v5_20261012/nestcom_wcl_draft_v5.tex'}
inputs=[dict(role=k,**pin(W/v))for k,v in paths.items()]
# Independently compare local inputs with current remote HEAD; read only existing metadata.
code=f'''from pathlib import Path
import json,hashlib,subprocess
R=Path({R!r});inputs={inputs!r}
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=={HEAD!r}
for x in inputs:
 b=(R/x['path']).read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],x['path']
p=Path({B!r})/'common500_original_sources/manifest.json';raw=json.loads(p.read_bytes())
reg=json.loads((R/{paths['supp_registration']!r}).read_bytes());assert len(raw['records'])==len(reg['sources'])==500
for a,b in zip(raw['records'],reg['sources']):
 assert all(a[k]==b[k] for k in ['source_index','source_id','preprocessing_id']) and a['archive_sha256']==b['archive']['sha256']
print(json.dumps(dict(status='PASS',head={HEAD!r},input_files=len(inputs),source_population_order_matches=500,source_archive_SHA_matches=500,preprocessing_id_matches=500,source_manifest=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),original_source_freeze=raw['final_freeze'],source_pixels_opened=0)))
'''
p=subprocess.run(['ssh','-T','DevMachineModel-H100-4C-P1660-01',CPY,'-B','-'],input=code,text=True,encoding='utf8',capture_output=True,check=True)
source_proof=json.loads(p.stdout);save(O/'review/input_remote_verification.json',source_proof)
raw=read(W/paths['raw_freeze']);sup=read(W/paths['supp_identity']);new=read(W/paths['t1c_identity'])
a=raw['holdout_metrics']['metadata']['metric_evaluator']['metrics']['dreamsim'];b=sup['model_metadata']['dreamsim'];c=new['model_metadata']['dreamsim']
checks=[]
for k in a['assets']:
 for f in ['files','sha256','revision']:
  if f in a['assets'][k]:
   assert a['assets'][k][f]==b[k][f]==c[k][f],(k,f)
   checks.append(dict(component=k,field=f,raw_supplement_t1c_equal=True))
assert sup['protocol']['dreamsim']==new['protocol']['dreamsim']
assert a['preprocessing']==dict(antialias=True,crop=None,interpolation='bicubic',normalization='official per-backbone internal normalization and normalize_embeds=True',resize=[224,224])
assert raw['population']['source_ids']==[s['source_id']for s in read(W/paths['supp_registration'])['sources']]
assert 'no additional 8-bit quantization' in raw['holdout_metrics']['metadata']['metric_evaluator']['input']
assert read(W/paths['raw_amendment'])['metric_arithmetic_changed'] is False
save(O/'review/dreamsim_identity.json',dict(status='COMPATIBLE_MATHEMATICAL_METRIC_AND_SOURCE_INPUTS',selected_fig3_metric='dreamsim',raw_metric_identity='25aca2656de9c9fdacf7774bc27ca284b51d33cb13c188c3360b8ae873fa491b',asset_checks=checks,raw_model=a['model_id'],raw_preprocessing=a['preprocessing'],raw_input=raw['holdout_metrics']['metadata']['metric_evaluator']['input'],current_protocol=sup['protocol']['dreamsim'],source_proof=source_proof,t1c_original_registration=new['original_registration'],notes=['Raw and newer scoring use the same DreamSim implementation/weights/ensemble, FP32 and no added quantization. GPU hardware differs; this is mathematical/input identity, not bitwise runtime identity.','Common500 source archives and preprocessing IDs match across all500 sources; no images opened or rescored for this drawing task.','Raw WHOLE was absent from newer six-method supplement; its genuine historical paired rows are retained.']))
tab={k:rows(W/v)for k,v in paths.items()if v.endswith('.csv')}
def exactly(role,fn):
 found=[(n,r)for n,r in enumerate(tab[role],2)if fn(r)];assert len(found)==1,(role,len(found));return found[0]
allrows=[]
def emit(fig,panel,method,metric,snr,role,fn,scale=1,flip=False):
 n,r=exactly(role,fn)
 original=[r.get('mean',r.get('mean_difference')),r.get('ci_low',r.get('CI95_low')),r.get('ci_high',r.get('CI95_high'))]
 vals=list(map(float,original));assert vals[1]<=vals[0]<=vals[2]
 vals=[-vals[0],-vals[2],-vals[1]]if flip else vals;vals=[v*scale for v in vals]
 x=dict(figure=fig,panel=panel,method=method,population='common500',N=1024,snr_db=snr,metric=metric,mean=vals[0],ci_low=vals[1],ci_high=vals[2],source_count=500,noise_per_source=3,logical_frames=1500,input_path=paths[role],input_csv_line=n,input_sha256=sha(W/paths[role]),row_identifier=json.dumps({k:r[k]for k in ['population','method','reference','family','comparison','snr_db','metric']if k in r},sort_keys=True),original_mean=original[0],original_ci_low=original[1],original_ci_high=original[2],transform='negate mean and exchange negated bounds'if flip else ('multiply mean and bounds by100'if scale==100 else 'identity'),interval='pointwise95% direct source-paired bootstrap'if fig==3 else 'pointwise95% source-bootstrap',version=role,bootstrap_recomputed=False)
 allrows.append(x)
for j,m in enumerate(resolved['metrics']):
 for spec in methods:
  for snr in [4,10,19]:
   mid=spec['id'];metric=m['id'];role='t1c'if mid=='EC_VAR_PARTIAL'and snr==19 else ('supp'if metric=='dreamsim'else'main')
   def select(r):
    if role=='t1c':return r['population']=='common500'and r['family']=='PARTIAL'and r['metric']==metric
    rid='BPG_ADAPTIVE'if r['method']=='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'else r['method']
    return rid==mid and float(r['snr_db'])==snr and r['metric']==metric
   emit(2,'abcd'[j],spec['label'],metric,snr,role,select,scale=m['scale'])
for j,metric in enumerate(['psnr_db','dreamsim']):
 for family in ['Entropy-coded','Raw']:
  for snr in [4,10,19]:
   role='raw_pair'if family=='Raw'else('t1c_pair'if snr==19 else('supp_pair'if metric=='dreamsim'else'entropy_pair'))
   def select(r):
    if role=='raw_pair':return r['method']==f'RAW64_PARTIAL_VAR_COMPLETION_SNR_{snr}'and r['reference']==f'RAW64_WHOLE_VAR_COMPLETION_SNR_{snr}'and r['metric']==metric
    if role=='t1c_pair':return r['population']=='common500'and r['comparison']=='PARTIAL_minus_WHOLE'and r['metric']==metric
    return r['method']=='EC_VAR_PARTIAL'and r['reference']=='EC_VAR_WHOLE'and float(r['snr_db'])==snr and r['metric']==metric
   emit(3,'ab'[j],family,metric,snr,role,select,flip=metric=='dreamsim')
assert len(allrows)==60
for r in allrows:
 if r['figure']==3 and r['method']=='Entropy-coded'and r['snr_db']==19:
  expected=(.3194168545132511,.28978511265719037,.34936179958492)if r['metric']=='psnr_db'else(.0030405402183532717,.0024240583151578904,.003717284870147705)
  assert tuple(r[k]for k in ['mean','ci_low','ci_high'])==expected
csvout(O/'plot_inputs/point_manifest.csv',allrows)
for f in [2,3]:csvout(O/f'plot_inputs/fig{f}_plot_data.csv',[r for r in allrows if r['figure']==f])
save(O/'input_manifest.json',dict(status='VERIFIED_EXISTING_CSV_ROWS_ONLY',current_head=HEAD,reference_snapshot='9e59f62e6b803b79564d57beb905d59dbe1e1ece',difference='Only previous figure2 previews and release_manifest; no science changes',inputs=inputs,fig2_points=48,fig3_points=12,fig3_second_metric='dreamsim',new_scoring=0,new_bootstrap=0,new_channel=0,new_reconstruction=0,populations_used=['common500'],excluded_populations=['new100','cal1000','Kodak','fixed16'],old_H800_timing_untouched=True))
print(json.dumps(dict(status='PASS',fig2=48,fig3=12,metric='dreamsim',source_proof=source_proof)))
