from pathlib import Path
import json,csv,hashlib
H=Path(__file__).resolve().parent;F=H/'deliverable';S=H/'snapshot';R='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM';O=R+'/results/submission_closure_20261016_v1'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
p=F/'implementation_fact_sheet.md';s=p.read_text(encoding='utf8');s=s.replace('Original ImageNet preprocessing is frozen RGB center256 under the original next_scale_prior/cache lineage;', 'Original ImageNet preprocessing converts to RGB, bicubic-antialiased resize of the shorter side to256 (each scaled dimension rounded), then center-crops256×256, under the original next_scale_prior/cache lineage;');p.write_text(s,encoding='utf8')
assets=list(csv.DictReader((F/'asset_inventory.csv').open(encoding='utf8')))
for a in assets:
 p=Path(a['actual_path'])
 if p.is_relative_to(S):a['actual_path']=R+'/'+p.relative_to(S).as_posix()
 elif p.is_relative_to(F):a['actual_path']=O+'/'+p.relative_to(F).as_posix()
 elif p.is_relative_to(H):a['actual_path']=O+'/evidence_v1/local_metadata/'+p.name
def asset(role,p,status,gap,identity_path=None):
 q=S/p if p.startswith(('results/','experiments/')) else F/p
 assets.append(dict(role=role,actual_path=R+'/'+p if p.startswith(('results/','experiments/')) else O+'/'+p,sha256=sha(q),exists=True,status=status,reuse_condition='Use exact registration and native-pixel identities; no fresh sampling',gap=gap))
asset('source_pixel_token_and_24_old_stream_identities','results/submission_closure_20261016_v1/source_registration.json','1000_NATIVE_SOURCE_TOKEN_ARCHIVES_AND_CURRENT_STREAMS_VERIFIED','m10 full-stream archive created under registered runtime; full CDF arrays are not persisted, ordered CDF hashes are retained')
asset('all64_native_fixed16_pairs_and_RX_lineage','results/submission_closure_20261016_v1/metric_registration.json','ALL64_NATIVE_PAIRS_VERIFIED','raw6201; continuous2001; only Swin/HiFi same actual RX')
asset('frozen_numerics_and_metric_preprocessing','results/submission_closure_20261016_v1/fixed16/runtime_identity.json','ACTUAL_RUNTIME_CLOSED','No metric branch or preprocessing changed after scores')
asset('both_families_workpoints','physical_workpoints.csv','EXISTING_PAID_PROFILES','m10 profile assignment not executed')
asset('all16_display_gallery','results/submission_closure_20261016_v1/figures_v2/manifest.json','ALL9_PNGS_ACTUALLY_OPENED','Display uint8 derived uniformly; scores remain native float32')
asset('cohort_roles_and_selection_exposure','cohort_roles.csv','SEVEN_REGISTERED_ROLES','Already evaluated sets are not fresh blind tests')
with (F/'asset_inventory.csv').open('w',encoding='utf8',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(assets[0]));w.writeheader();w.writerows(assets)
v=read(F/'actual_visual_review.json');v['local_opened_manifest']=dict(v['manifest']);v['manifest']['path']=O+'/figures_v2/manifest.json';v['dedicated_failure_example']='Existing first-two figure, index0, also identified descriptively as minimum HiFi PSNR among the fixed16; see illustrative_failure_note.md. No gallery reselection or new render.';dump(F/'actual_visual_review.json',v)
(F/'illustrative_failure_note.md').write_text('''# Explanatory fidelity example, not a success-rate estimate

The top row (index0) in [the unchanged first-two figure](figures_v2/fixed16_first_two.png)
is also a useful explanatory failure example. The ordinary figure remains the
first two sources in the original registration; it was not chosen to favor a method.
For this additional annotation, the explicit descriptive rule is minimum HiFi PSNR
over all16 (index0,16.9861dB). This is post-score explanatory selection, not a new test.

The source contains a person holding a fish. The HiFi reconstruction has altered
facial and shirt details; its texture reconstruction does not ensure fidelity.
Raw also changes the face/expression and shirt lettering. The visual example does
not establish identity preservation by raw or an overall ordering. All four native
outputs, all16 sources and all numerical scores remain included. Only Swin/HiFi
share an actual received observation; raw uses its preserved6201 noise.
The figure uses uniform display quantization, with no local enhancement. No new
image recovery, metric or population hallucination-rate estimate was made.
''',encoding='utf8')
p=F/'REPORT_zh.md';s=p.read_text(encoding='utf8').replace('[预定前两张](figures_v2/fixed16_first_two.png)；','[预定前两张](figures_v2/fixed16_first_two.png)；[解释性失真例图说明](illustrative_failure_note.md)；');p.write_text(s,encoding='utf8')
print('REPORTS_FINALIZED',len(assets))
