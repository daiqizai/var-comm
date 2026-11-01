"""Complete metadata exports only; no statistical interval or model recomputation."""
from pathlib import Path
import sys,json,csv
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,pin
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');L=B/'nestcom_v9_completion_20261101_v1';D=R/'results/nestcom_v9_completion_20261101_v1'
def csvout(p,rr):
 with p.open('x',newline='')as f:w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 p=D/'reproduction/model_manifest.json';m=read(p);config=read(B/'nestcom_perceptual_followup_20261030_v2/E5_plan_assets_v3/config.json');r=read(R/'results/nestcom_perceptual_followup_20261030_v2/E5_dep5_g00_tx_registration.json')
 m['NeST']['revision_note']='The linked upstream notebook revision identifies the public reference; actual deployed source identity is the per-file author_files SHA list in routes, not an inferred Git revision.'
 m['Ada']['actual_native_config']=config['ada_native'];m['Ada']['classifier_source']='https://download.pytorch.org/models/convnext_tiny-983f1562.pth'
 m['ARPC']=dict(configuration=config['arpc_public'],sources=read(R/'results/arpc_external_baseline_20261019_v1/a0_engineering/asset_sources.json'),runtime='Original ARPC environment_v1, /usr/bin/python3; source-config precision preserved, no conversion to NeST route')
 m['Latent_JSCC']=dict(checkpoint=config['P_checkpoint'],updates=100000,loss='MSE+0.1 LPIPS+0.01 latent consistency',decoder='shared adapted Dc',training_sources=20000,recipe='experiments/var-latent-enhancement-20260917/configs/experiment.json, stage B; continued baseline_strength_20261011/P',availability='project trained checkpoint on PFS; no public URL claimed',runtime='frozen_visual_py310 FP32, same recorded baseline route')
 m['SwinJSCC']=dict(checkpoint=config['Swin_checkpoint'],updates=200000,loss='RGB MSE',training_sources=20000,selection='minimum calibration MSE among scheduled checkpoints',upstream='https://github.com/semcomm/SwinJSCC',availability='project adapted checkpoint on PFS; upstream checkpoint is not interchangeable',runtime='frozen_visual_py310 FP32; exact checkpoint identities below')
 m['checkpoint_file_pins']=[v for v in r['code']if Path(v['path']).suffix in ['.pth','.pt','.ckpt','.safetensors']]
 m['state_hash_semantics']='NeST tensor-state hashes are routes.models; checkpoint_file_pins and upstream.weights are file SHA256. A separate state hash was not recorded for some external checkpoints; file identity and exact architecture/configuration are provided instead. No fabricated state hashes.'
 m['reader_asset_mapping']={Path(v['path']).name:v['path']for v in m['checkpoint_file_pins']};m['reader_asset_mapping'].update({Path(v['path']).name:v['path']for v in config['ada_native']['weights'].values()})
 p.write_text(json.dumps(m,indent=2)+'\n')
 ff=read(L/'token_frame_plan/logical.json');rr=[]
 for snr in [4,10,19]:
  fs=[f for f in ff if f['snr_db']==snr];assert len(fs)==72
  rr.append(dict(method='Token',N=1024,snr_db=snr,sources=24,events=72,mean_source_bits=float(np.mean([f['source_bits']for f in fs])),capacity_bits=fs[0]['capacity_bits'],mean_utilization=float(np.mean([f['utilization']for f in fs])),nofit=sum(f['tx_no_fit']for f in fs),failures=0))
 csvout(D/'kodak/capacity_summary.csv',rr)
 save(D/'kodak/reuse_identity.json',dict(source=pin(L/'metric_plan/sources.json'),native_pairs=pin(L/'metric_plan/dists_pairs.json'),metric_plan=pin(L/'metric_plan/frames.json'),baseline_four_reuse='648 exact historical logical values; no model rerun',token_four_reuse='3 logical values from exact baseline pixel pair; 71 new unique pairs',preprocessing='Kodak original RGB center crop256 without prior resize',source_pixels_sha256_field_note='In inherited grid_source metadata this legacy field stores preprocessing_id; source_float_sha256/reference_sha256 are the actual native pixel hashes. This clarification does not change source inputs.'))
 save(D/'kodak/protocol.json',dict(N=1024,SNR=[4,10,19],sources=24,repeats=[2001,2002,2003],policies=pin(R/'results/t1b_m10_calibration_20261016_v1/delivery/frozen_policies.json'),source_preprocessing='original RGB center crop256; no resize',new_training=0,scope=pin(D/'scope.json')))
 print('Metadata and capacity exports complete; no new scientific calls')
if __name__=='__main__':main()
