"""Bind the new LE2 scope after closed gray metadata, before any packet."""
from common import *
import subprocess
def run():
 assert not (O/'registration.json').exists() and not (T/'packets').exists() and not (T/'packets_control_v1').exists()
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='bbd75f967a612ce968e5f0ea0aa73b88c774ca4c'
 waits=[]
 for role in ('gray_pixel_identity_revision','tests'):
  w=read(T/(role+'_control_v1/actual_wait.json'));assert type(w['returncode'])is int and w['returncode']==0 and w['actual_wait'] and not any(w[k] for k in ('timed_out','terminated','killed')) and gone(w['process']) and gone(w['parent_process']);waits.append(pin(T/(role+'_control_v1/actual_wait.json')))
 assert read(T/'gray_pixel_identity_revision_v1/completion.json')['status']=='COMPLETE_FIXED_GRAY50_REUSE17_NEW33'
 original=read(O1/'registration.json');assert sha(O1/'registration.json')=='a15f32fc31d92acfc33db11cc4a5e3fed56a09143490c6508337ddfc0e3ab6d5'
 for s in original['source_bindings']:checked(s)
 assert read(O1/'final/summary.json')['status']=='COMPLETE_LE0_LE1_ONLY_WITH_DISCLOSED_HISTORICAL_FAILURES'
 own=[pin(q) for q in sorted(E.iterdir()) if q.is_file()]
 deps=[O1/'registration.json',O1/'assets/pilot50.json',O1/'assets/model_routes.json',O1/'source_lengths.csv',O1/'clean_quality.csv',O1/'final/summary.json',O1/'user_package/scope.json',O/'gray_quality.csv',O/'gray_registration.json',T/'gray/completion.json',T1/'feasibility/group_layouts.json',R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py',E1/'group_layout_design.py',E1/'packet_two_group.py',R/'experiments/prior-aware-grouped-mcs-v1/ldpc_backend.py',R/'experiments/prior-aware-grouped-mcs-v1/uep_phy.py',R/'experiments/prior-aware-grouped-mcs-v1/uep_common.py',R/'experiments/main-raw64-20261007/main_raw64_packet_adapter.py',R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json',R/'results/entropy_main_common500_20261013/e0_reuse/frozen_configs.json']
 # Original package lives under the experiment, not under results.
 deps=[E1/'user_package/scope.json' if q==O1/'user_package/scope.json' else T/'gray_pixel_identity_revision_v1/completion.json' if q==T/'gray/completion.json' else q for q in deps]
 deps += [O/'gray_pixel_identity_revision_authorization.json',T/'gray_control_v1/actual_wait.json',T/'gray/failure.json']
 reg=dict(schema='LE2_COMPLETE_STRATEGY_RESOURCE_FIELD_CORRECTION_V1',parent_HEAD='bbd75f967a612ce968e5f0ea0aa73b88c774ca4c',authorized_stages=['LE2'],authorization='User: preserve LE0/LE1; revise allocation29/group; stop capacity-occurrence nominations; complete-strategy7/10 and single956 coverage; actual per-source fallback; execute within21504; deliver LE3 candidates and STOP.',packet_cap=CAP,initial_cell_cap=96,initial_packets_per_cell=128,topup_cell_cap=24,topup_additional_packets_per_cell=384,wireless_image_frames=0,source_probability_traversals=0,clean_reconstruction_calls=0,training_updates=0,LE3_authorized=False,power_expansion_authorized=False,backup_deferred_by_user=True,population='Original hash-selected LE1 Pilot50/cal200',snrs=list(SNRS),hat=10,objective='DINO_L J=.25 true7+.75 true10',group_length_bits=13,group_CRC_bits=16,header_symbols=68,body_symbols=956,N=1024,nominal_Es=2,ratio_rule='round_even 956*(medianB+29)/(medianB+medianE+58)',nomination='Complete strategy, actual source fallback, explicitly screen-only single956 MCS transfer; no capacity counts, no unknown BLER substitution',bindings=own+[pin(q) for q in deps],closed_prerequisite_waits=waits,entropy_nominal_binding=pin(deps[-1]),original_ep_phy_unchanged_sha256=sha(R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py'),mapping_bindings=[pin(R/'experiments/prior-aware-grouped-mcs-v1/uep_phy.py'),pin(R/'experiments/main-raw64-20261007/main_raw64_packet_adapter.py')],no_extra_PHY_qualification_calls=True)
 reg['image_wireless_frames']=0
 assert reg['original_ep_phy_unchanged_sha256']=='12cad3e495fc91d8f6819b895674df08e312da3833291019b8966a2211ec8c75'
 save(O/'registration.json',reg);print(canonical(pin(O/'registration.json')))
if __name__=='__main__':run()
