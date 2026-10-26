"""Metadata-only once registration after verified existing Kodak token restoration."""
from kn_common import *
import argparse,subprocess,numpy as np
def main(tokens_path,tokens_sha):
 assert sha(tokens_path)==tokens_sha;tokens=read(tokens_path);assert set(tokens)==set(map(str,range(24)))
 parent=R/'results/kodak_current_20261026_v1/scope_registration.json';population=R/'results/kodak_current_20261026_v1/population.json'
 assert sha(parent)=='3b5fe30a7d56acc4564beaa6dfaa75745d691def12df5ce63629875989d93b20'
 assert sha(population)=='a1d255e7491a8ae7303ccdf481a4abf48ff2133e9b03c56da8b756af657a941f'
 sources=read(population)['sources'];assert len(sources)==24
 for s in sources:
  d=tokens[str(s['source_index'])];checked(s['png'])
  assert d['source_id']==s['source_id']and d['source_pixels_chw_uint8_sha256']==s['pixels_chw_uint8_sha256']
  with np.load(checked(d['archive']),allow_pickle=False)as z:a=z[d['key']].astype(np.int64)
  assert a.shape==(680,)and asha(a)==d['tokens_sha256']and np.all((a>=0)&(a<4096))
 old=R/'results/t1c_m10_test_20261016_v1/source_registration.json';oldlink=R/'results/t1c_m10_test_20261016_v1/link_registration.json'
 sr=read(old);lr=read(oldlink);code={d['path']:d for d in sr['code']+lr['code']}
 for d in code.values():checked(d)
 extra=[R/'experiments/n1024-consistency-20261019/audit_v2.py',R/'experiments/content-real-64qam-20261006/h_online_components.py',R/'experiments/t1b-m10-calibration-20261016/runtime_boundary.py',R/'experiments/t1c-m10-test-20261016/tc_receiver.py']
 for p in list(E.glob('*.py'))+extra:code[str(p)]=pin(p)
 for k in ('model_routes','catalogue','phy_qualification'):checked(lr[k])
 for k in ('completion','table'):checked(lr['static'][k])
 q=L/'qualification_v1';w=read(q/'actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process'])and gone(w['parent_process'])
 assert 'Ran 9 tests' in(q/'stderr.log').read_text()and 'OK'in(q/'stderr.log').read_text()
 r=dict(schema='KODAK_CURRENT_NEST_WHOLE_EXECUTION_V1',scope='KODAK24_CURRENT_N1024_NEST_WHOLE_ONLY',basis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),parent_scope=pin(parent),population=pin(population),old_source_registration=pin(old),old_link_registration=pin(oldlink),code=list(code.values()),token_reuse=tokens,token_reuse_manifest=pin(tokens_path),model_routes=lr['model_routes'],catalogue=lr['catalogue'],phy_qualification=lr['phy_qualification'],backend_identity=lr['backend_identity'],static=lr['static'],frozen_policies=sr['frozen_policies'],gpu=dict(uuid='GPU-a9818dee-2885-eacd-d344-8b1f35b82798',pci_bus_id='00000000:61:00.0',admission_free_mib=30000,admission_util_percent=80,runtime_free_mib=2048,allocator_cap_bytes=12<<30),python=dict(gpu=str(RT/'envs/frozen_visual_py310/bin/python'),cpu=str(RT/'envs/frozen_ldpc_py311/bin/python')),max_seconds=dict(source=7200,phy=7200,receiver=7200,delivery=1800),source_count=24,logical_frames=432,physical_frame_cap=432,packet_decode_cap=864,source_probability_pass_cap=24,tokenizer_calls_expected=0,new_finished_streams=600,noise_seeds=[2001,2002,2003],snr_db=[4,10,19],noise_rule='Unchanged ep_phy.standard_noise(source_id,noise_seed); WHOLE/PARTIAL same standard array, waveform-specific observation',counter_rule='(SNR_slot*24+source_index)*3+noise_slot; shared across families',quality_scores=0,training_updates=0,calibration_calls=0,qualification=pin(q/'actual_wait.json'),automatic_successor=False)
 O.mkdir(parents=True,exist_ok=False);save(O/'registration.json',r)
 for name in ('launch.json','actual_wait.json','stdout.log','stderr.log'):
  dest=O/'qualification_v1'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes((q/name).read_bytes())
 print(json.dumps(pin(O/'registration.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tokens',required=True);p.add_argument('--tokens-sha256',required=True);a=p.parse_args();main(a.tokens,a.tokens_sha256)
