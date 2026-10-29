"""One-time CPU-only immutable registration; no neural or PHY forward calls."""
import sys,json,subprocess,os
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
from common import pin,cr,checked,save,asha
import numpy as np
O=R/'results/adatoken_response_20261026_v1';D=R/'results/nestcom_focused_review_20261029_v3'
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');L=B/'nestcom_focused_20261029_v3'
RT=R.parent.parent/'var_comm_runtime_20261010'

def main():
    assert not(D/'timing_registration.json').exists()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='e2ed72c95ddeab15cc4ca42b1dd7f2e0e5236c8c'
    first=json.loads((O/'a_first_source_hash_v2_registration.json').read_bytes());remaining=json.loads((O/'a_remaining99_source_registration.json').read_bytes());link=json.loads((O/'a_link_import_v2_registration.json').read_bytes())
    initial=cr(first['all_sources']);sources=cr(link['sources']);assert [s['source_id'] for s in sources]==[s['source_id'] for s in initial]
    logical=json.loads((B/'adatoken_response_20261026_v1/a_link_import_v2/logical_frames.json').read_bytes());assert len(logical)==3600
    jobs={(j['source_id'],j['grid'],j['noise_seed']):j for j in logical if j['snr_db']==10 and j['grid'] in (4,8)};assert len(jobs)==600
    manifest=[]
    for s,orig in zip(sources,initial):
        checked(s['streams']);checked(s['trace'])
        with np.load(checked(orig['archive']),allow_pickle=False) as z:pixels_sha=asha(z['pixels'])
        entry={k:orig[k] for k in ('source_id','source_index','archive','tokens_sha256','preprocessing_id')}
        entry.update(pixels_sha256=pixels_sha,streams=s['streams'],trace=s['trace'],jobs={})
        for grid in (4,8):
            entry['jobs'][str(grid)]={}
            for seed in (6201,6202,6203):
                j=jobs[s['source_id'],grid,seed];raw=cr(j['physical_frame'])
                assert raw['payload_sha256']==j['payload_sha256']
                keys=('m','K','profile_id','public_frame_counter','length_checks','arithmetic_bits','payload_sha256','physical_frame')
                entry['jobs'][str(grid)][str(seed)]={**{k:j[k] for k in keys},'waveform_sha256':raw['waveform_sha256']}
        manifest.append(entry)
    L.mkdir(exist_ok=False)
    r=dict(schema='FOCUSED_V3_QUARTER_EIGHTH_TX_V1',base_commit='e2ed72c95ddeab15cc4ca42b1dd7f2e0e5236c8c',sources=manifest,model_routes=remaining['model_routes'],catalogue=link['catalogue'],original_link_registration=pin(O/'a_link_import_v2_registration.json'),source_lineage=[pin(O/'a_first_source_hash_v2_registration.json'),pin(O/'a_remaining99_source_registration.json')],output=str(L/'timing_v1'),max_seconds=7200,gpu_index=7,gpu_uuid='GPU-cefdc82b-dbab-987f-8b1a-4f98a2840e9f',gpu_pci='00000000:A4:00.0',gpu_affinity=[0,1],phy_affinity=[2,3],cpu_python=str(RT/'envs/frozen_ldpc_py311/bin/python'),visual_python=str(RT/'envs/frozen_visual_py310/bin/python'),native_build=str(B/'baseline_strength_20261011/native_build'),caps=dict(measured=600,qualification=4,warmup=6,complete_tx=610,tokenizer=610,prior_scale=6100,tx_waveform=610,receive=0,metrics=0,training=0),schedule='first two sources x grids qualification, first source x3 xgrids warmup, then repeat-major source-order alternating grid-first by source+repeat parity',timing=dict(input='Original preprocessed uint8 RGB256 pixels resident in host memory; conversion/H2D/tokenizer included',output='Complete 1024x2 transmit waveform resident in host memory',included='Tokenizer, one full prior/CDF traversal, 680 incremental AC tokens, grid-specific nonmutating finishes, lookup, packing/LDPC/QAM, persistent CPU-worker IPC and transfers',excluded='Model load, file/archive I/O, durable ledger reservation/completion, output identity auditing',components='CPU wall timers; GPU sync at tokenizer and source boundaries, not at endpoints; total measured directly',interpretation='Shared-server software timing, not uncontended device latency; no comparison to legacy timing population',statistic='mean/median/linear p95 over300 calls per grid; first average3 repeats/source then direct paired Eighth-Quarter mean; no new CI/bootstrap'),automatic_retry=False)
    code={x['path']:x for reg in (first,remaining,link) for x in reg['code']}
    for p in (R/'experiments/nestcom-focused-v3-20261029').glob('*.py'):code[str(p)]=pin(p)
    for name in ['experiments/kodak-current-20261026/nest_v1/kn_native.py','experiments/kodak-current-20261026/nest_v1/kn_common.py','experiments/content-real-64qam-20261006/h_online_components.py']:code[str(R/name)]=pin(R/name)
    for v in code.values():checked(v)
    r['code']=list(code.values());save(D/'timing_registration.json',r)
    save(D/'timing_source_manifest.json',dict(sources=manifest,source_lineage=r['source_lineage'],registration=pin(D/'timing_registration.json')))
    print(json.dumps(dict(status='REGISTERED_NOT_LAUNCHED',registration=pin(D/'timing_registration.json'),sources=len(manifest))))

if __name__=='__main__':main()
