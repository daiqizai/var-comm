"""CPU audit of real frozen Swin/ADJSCC PHY receipts and exact noise identities."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/external-baseline-positioning-20260916'
BASE=ROOT/'outputs/EXTERNAL-BASELINE-POSITIONING-20260916'


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)

def resources(family,rate):
    if family=='swin' and rate in (32,64,96):
        data=128*rate;metadata=2*(32+(math.comb(320,rate)-1).bit_length()+22)
    elif family=='adjscc' and rate in (2,4,6):data=2048*rate;metadata=0
    else:raise ValueError('registered family/rate required')
    return data,metadata,data+metadata

def key(row):return row['method'],row['image_id'],float(row['snr_db']),int(row['seed'])

def validate_row(row,family,checkpoint_hash):
    data,metadata,total=resources(family,int(row['rate']))
    require(row['protocol']=='common_paid_information','unpaid reference cannot enter paid scope')
    for field,value in [('data_complex_uses',data),('metadata_complex_uses',metadata),('complex_uses',total)]:require(float(row[field])==value,'actual data/metadata resource ledger mismatch')
    energy=float(row['actual_total_energy']);require(math.isfinite(energy) and abs(energy-2*total)<.01,'finite strict energy required')
    require(row['checkpoint_sha256']==checkpoint_hash,'model identity mismatch')
    require(int(row['NFE'])==0 and int(row['batch_size'])==1 and row['precision']=='FP32_TF32_disabled','execution scope mismatch')
    require(not row.get('unmetered_sender_metadata'),'sender metadata must be charged')
    require(float(row['snr_db']) in (1.,4.,7.,13.,19.) and int(row['seed']) in (2001,2002,2003),'noise population mismatch')
    for field in ('TX_seconds','RX_seconds'):require(math.isfinite(float(row[field])) and float(row[field])>=0,'invalid timing receipt')
    for field in ('data_transmitted_sha256','data_observed_sha256','standard_data_noise_sha256','source_pixels_sha256'):
        require(len(row[field])==64 and all(c in '0123456789abcdef' for c in row[field]),'missing actual waveform/noise/source identity')
    return data,metadata,total


def main():
    sys.path.insert(0,str(ROOT/'src'))
    from var_comm.study import seeded_noise
    output=ROOT/'results/token_channel_efficiency_20260923/author_phy_audit_v1'
    require(not output.exists(),'immutable existing audit')
    bindings={}
    def bind(p,expected=None):
        p=Path(p);sha=digest(p)
        if expected:require(sha==expected,'binding changed: '+str(p))
        bindings[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=sha
        return sha
    protocol=read(EXP/'configs/protocol.json');bind(EXP/'configs/protocol.json')
    inputs=read(BASE/'inputs_001/development_inputs.json');sources={r['image_id']:r for r in inputs};require(len(sources)==100,'original 100 development only')
    bind(BASE/'inputs_001/development_inputs.json')
    source_hash={r['image_id']:r['source_pixels_sha256'] for r in inputs}
    summaries=[];checked_frames=0
    for family,rates in [('swin',(32,64,96)),('adjscc',(2,4,6))]:
        folder=BASE/f'development_{family}_001';metadata=read(folder/'metadata.json');done=read(folder/'completion.json')
        bind(folder/'metadata.json');bind(folder/'completion.json');bind(folder/'per_frame.csv',done['per_frame_sha256'])
        for p,sha in metadata['bindings'].items():bind(p,sha)
        with (folder/'per_frame.csv').open() as f:data=[r for r in csv.DictReader(f) if r['protocol']=='common_paid_information']
        prefix='swin_ra' if family=='swin' else 'adjscc_c'
        expected=set(itertools.product([prefix+str(r) for r in rates],sources,(1.,4.,7.,13.,19.),(2001,2002,2003)))
        require(len(data)==4500 and {key(r) for r in data}==expected,'full paid frame coverage')
        by_rate={rate:[] for rate in rates}
        for row in data:by_rate[int(row['rate'])].append(row)
        for rate,group in by_rate.items():
            checkpoint='SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model' if family=='swin' else f'ADJSCC_C={rate}.pth.tar'
            sha=protocol['checkpoints'][checkpoint];bind(EXP/'checkpoints'/checkpoint,sha)
            max_energy_error=0.;fallbacks=0;crc_rejected=0;legal_rejected=0;metadata_not_exact=0;tx=[];rx=[]
            for row in group:
                ndata,nmeta,total=validate_row(row,family,sha)
                require(row['source_pixels_sha256']==source_hash[row['image_id']],'source pixel identity')
                index=int(row['image_index']);require(index==int(sources[row['image_id']]['image_index']),'source index')
                frame_seed=2026091600+index*32+(1.,4.,7.,13.,19.).index(float(row['snr_db']))*3+int(row['seed'])-2001
                require(int(row['frame_counter_seed'])==frame_seed,'pre-shared frame counter changed')
                noise=seeded_noise(row['image_id'],int(row['seed']),(total,2))[:ndata]
                require(hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest()==row['standard_data_noise_sha256'],'actual standardized noise mismatch')
                archive=Path(row['image_archive']);require(archive.is_relative_to(folder),'archive outside registered run')
                record=read(archive.parent/'frame.json');saved=[r for r in record['rows'] if r['protocol']=='common_paid_information'];require(len(saved)==1,'frame paid receipt coverage');saved=saved[0]
                for field in ('method','image_id','data_transmitted_sha256','data_observed_sha256','standard_data_noise_sha256','checkpoint_sha256','image_sha256'):require(saved[field]==row[field],'frame/CSV identity mismatch')
                require(float(saved['actual_total_energy'])==float(row['actual_total_energy']),'frame/CSV energy mismatch')
                # Pixels/archive integrity was independently verified in reference_audit_v1.
                bind(archive.parent/'frame.json')
                max_energy_error=max(max_energy_error,abs(float(row['actual_total_energy'])-2*total));fallbacks+=bool(row['fallback']);tx.append(float(row['TX_seconds']));rx.append(float(row['RX_seconds']))
                if family=='swin':
                    crc_rejected+=row['metadata_crc_accepted']=='False';legal_rejected+=row['metadata_crc_accepted']=='False' and row['metadata_usable']=='True';metadata_not_exact+=row['offline_metadata_exact']=='False'
                checked_frames+=1
            summaries.append({'method':prefix+str(rate),'rows':len(group),'data_N':ndata,'metadata_N':nmeta,'total_N':total,'max_abs_energy_error':max_energy_error,'checkpoint_sha256':sha,'noise_identity_checks':len(group),'fallbacks_retained':fallbacks,'metadata_CRC_rejected':crc_rejected,'legal_rejected_candidates_retained':legal_rejected,'metadata_inexact':metadata_not_exact,'historical_TX_mean_ms':1000*float(np.mean(tx)),'historical_RX_mean_ms':1000*float(np.mean(rx)),'timing_scope':'historical native torch1.12.1; CPU4 threads, one warmup/rate; not uniform current six-thread timing'})
    qualification={}
    for family in ('swin','adjscc'):
        folder=BASE/f'qualify_{family}_001';q=read(folder/'inputs.json');d=read(folder/'completion.json');events=read(folder/'events.json')
        for p in folder.glob('*.json'):bind(p)
        qualification[family]={'events':len(events),'torch':d['torch'],'cuda':d['cuda'],'max_native_path_RGB_difference':max(e['author_path_max_RGB_difference'] for e in events),'recorded_adapter_sha256':q['adapter_sha256'],'matches_development_adapter':q['adapter_sha256']==digest(EXP/'src/external_positioning/author_models.py'),'meaning':'historical native check only; adapter revision differs, fresh real parity required before current acceptance'}
    output.mkdir(parents=True)
    with (output/'resource_and_timing_scope.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    # Keep the large per-frame manifest local; publish its hash and compact provenance.
    local=ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/author_phy_audit_v1';local.mkdir(parents=True,exist_ok=True)
    (local/'full_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    audit={'status':'REAL_AUTHOR_PHY_RECEIPTS_NOISE_AND_RESOURCE_IDENTITY_PASS_FRESH_PARITY_TIMING_PENDING','rows':checked_frames,'synthetic':False,'GPU_used':False,'new_holdout':False,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'tool_sha256':digest(__file__),'full_bindings_sha256':digest(local/'full_bindings.json'),'bound_files':len(bindings),'summaries':summaries,'historical_qualification':qualification,'side_information':'Swin float32 power plus combinatorial mask rank protected and paid; ADJSCC RX uses power1 and pre-shared frame counter, no TX power supplied','CRC_failure_rule':'legal hard-decoded metadata retained under original protocol; never replace it with TX truth','uniform_timing_reuse':False,'pending':['fresh native parity for actual development adapter revision','same scope uniform endpoint timing in pinned author runtime; disclose runtime difference','N3060 DeepJSCC/R3/digital full PHY/timing review','existing common-metric rescoring and combined paired publication'],'bindings':{k:v for k,v in bindings.items() if not k.endswith('/frame.json')}}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:audit[k] for k in ('status','rows','bound_files','historical_qualification')},indent=2))
if __name__=='__main__':main()
