"""CPU-only audit of actual N3060 reference waveforms, lineage and frozen policy.

No model inference, new quality metrics, source selection or old-artifact writes.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
OUT=BASE/'n3060_phy_audit_v1'
PUBLIC=ROOT/'results/token_channel_efficiency_20260923/n3060_phy_audit_v1'
HISTORY=ROOT/'outputs/COMMUNICATION-CONVERGENCE-20260915'
DEEP=ROOT/'outputs/VAR-PREFIX-JSCC-EVAL-001'
R3=ROOT/'outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-EVALUATION/quality_0010000'
SNRS=(1.,4.,7.,13.,19.)
METHODS=('raw_adaptive','arithmetic_adaptive','perceptual_deepjscc','wetok_r3')

def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def key(row):return int(row['image_index']),float(row['snr_db']),int(row['seed'])

def waveform_check(tx,rx,standard_noise,snr,tolerance):
    tx=np.asarray(tx);rx=np.asarray(rx)
    require(tx.shape in ((3060,2),(1,3060,2)) and rx.shape==tx.shape,'actual N3060 waveform dimensions')
    require(np.isfinite(tx).all() and np.isfinite(rx).all(),'finite real TX/RX')
    energy=float(np.square(tx,dtype=np.float64).sum())
    require(abs(energy-6120)<.01,'actual waveform energy, not declared energy')
    expected=tx.astype(np.float64)+standard_noise.reshape(tx.shape)/np.sqrt(10**(snr/10))
    error=float(np.max(np.abs(rx-expected)))
    require(error<=tolerance,'single-noise actual observation mismatch')
    return energy,error

def digital_ledger(row):
    family=row['family'];header=68 if family=='raw' else 94
    require(family in ('raw','arithmetic'),'registered coding family')
    require(int(row['header_uses'])==header and int(row['data_uses'])==3060-header,'paid header/data allocation')
    require(int(row['complex_uses'])==3060 and float(row['total_energy'])==6120,'declared resource identity')
    payload=int(row['actual_payload_bits'])
    require(int(row['header_payload_bits'])==(12 if family=='raw' else 25),'control payload size')
    require(int(row['header_crc_bits'])==16 and int(row['header_tail_bits'])==6 and int(row['header_coded_bits'])==2*header,'protected control ledger')
    require(int(row['data_crc_bits'])==16 and int(row['data_tail_bits'])==6,'body CRC/tail ledger')
    require(int(row['data_mother_coded_bits'])==2*(payload+22) and int(row['data_coded_bits'])==2*(3060-header),'source/FEC/uses must be separate')
    require(payload>0 and int(row['raw_payload_bits'])>0,'positive actual source stream')
    return header,payload

def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','mask CUDA for CPU audits')
    from var_comm.study import seeded_noise
    require(not PUBLIC.exists(),'immutable published audit already exists')
    bindings={};checks=[];changes=[]
    def bind(p,expected=None):
        p=Path(p)
        name=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        if name not in bindings:bindings[name]=digest(p)
        if expected:require(bindings[name]==expected,'file identity: '+str(p))
        return bindings[name]
    input_root=ROOT/'outputs/EXTERNAL-BASELINE-POSITIONING-20260916/inputs_001'
    source_inputs=read(input_root/'development_inputs.json');require(len(source_inputs)==100,'original development only')
    sources={r['image_index']:r for r in source_inputs}
    reference=rows(input_root/'reference_per_frame.csv');require(len(reference)==6000,'complete reference table')
    lookup={(r['method'],*key(r)):r for r in reference}
    expected={(m,i,s,n) for m in METHODS for i in range(100) for s in SNRS for n in (2001,2002,2003)}
    require(set(lookup)==expected and len(lookup)==6000,'full source/SNR/noise/method identity')
    imported=read(input_root/'completion.json');bind(input_root/'completion.json')
    bind(input_root/'reference_per_frame.csv',imported['reference_csv_sha256'])
    for p,h in imported['bindings'].items():bind(Path(p),h)
    policy_path=HISTORY/'POLICIES_001/policies.json';policy=read(policy_path);bind(policy_path)
    require(policy['status']=='CALIBRATION_POLICIES_FROZEN' and not policy['development_used_for_fitting'] and not policy['holdout_accessed'],'calibration-frozen mode policy')
    bind(HISTORY/'CALIBRATION_001/completion.json',policy['calibration_receipt_sha256'])
    bind(HISTORY/'CALIBRATION_001/per_frame.csv',policy['calibration_rows_sha256'])
    cfgpath=ROOT/'configs/learned_prefix_jscc.yaml';cfg=yaml.safe_load(cfgpath.read_text());bind(cfgpath)
    assets={}
    for k in ('checkpoint','initialization_checkpoint'):
        p=Path(cfg['deepjscc'][k]);h=cfg['deepjscc']['checkpoint_sha256' if k=='checkpoint' else 'initialization_sha256']
        bind(p,h);assets['deep_'+k]={'path':str(p),'sha256':h}
    milestone_path=ROOT/'outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-TRAINING/milestones/step_0010000.json'
    milestone=read(milestone_path);choice=milestone['selected'];bind(milestone_path)
    bind(Path(choice['checkpoint']),choice['checkpoint_sha256']);assets['r3_selected']=choice
    require(milestone['completed_updates']==10000 and milestone['global_data_step']==17000,'R3 opportunity lineage')
    completed={p:read(p/'completion.json') for p in (HISTORY/'DEVELOPMENT_001',DEEP,R3)}
    for directory,receipt in completed.items():
        bind(directory/'completion.json')
        before=receipt.get('frozen_before',receipt.get('frozen_models_before'))
        after=receipt.get('frozen_after',receipt.get('frozen_models_after'))
        require(before and before==after,'historical model freeze receipt')
        for name,old in receipt['source_hashes'].items():
            relative=name.removeprefix('VAR_COMM/');p=ROOT/relative
            current=digest(p) if p.is_file() else None
            if current!=old:changes.append({'scope':str(directory.relative_to(ROOT)),'file':relative,'historical_sha256':old,'current_sha256':current,'interpretation':'source identity differs; fresh real compatibility required, not proof of a model bug'})
    require(completed[R3]['r3_milestone_sha256']==digest(milestone_path),'R3 actual evaluation selected lineage')
    deep_rows={key(r):r for r in rows(DEEP/'per_frame.csv') if r['arm']=='perceptual_deepjscc'}
    r3_rows={key(r):r for r in rows(R3/'per_frame.csv') if r['arm']=='r3__full_grid_prediction_features'}
    def archive(directory,relative):
        p=directory/relative;bind(p,completed[directory]['output_hashes'][relative]);return np.load(p,allow_pickle=False)
    def record(method,row,tx,rx,noise,header,payload=None):
        i,s,n=key(row);ref=lookup[method,i,s,n];source=sources[i]
        require(row['image_id']==source['image_id']==ref['image_id'],'source identity')
        require(ref['source_pixels_sha256']==source['source_pixels_sha256'] and ref['standard_noise_sha256']==array_sha(noise),'source/noise identity')
        energy,error=waveform_check(tx,rx,noise,s,1e-12 if method.endswith('adaptive') else 2e-6)
        checks.append({'method':method,'source_index':i,'source_id':ref['image_id'],'preprocessing_id':source['source_pixels_sha256'],'snr_db':s,'noise_seed':n,'N':3060,'E':energy,'header_N':header,'data_N':3060-header,'source_payload_bits':payload,'single_noise_max_abs_error':error,'tx_sha256':array_sha(tx),'rx_sha256':array_sha(rx),'standard_noise_sha256':array_sha(noise),'image_sha256':ref['image_sha256']})
    for i in range(100):
        source=sources[i];bind(Path(source['path']),source['file_sha256'])
        px=np.load(source['path'],allow_pickle=False);require(array_sha(px)==source['source_pixels_sha256'],'real source pixels')
        with archive(HISTORY/'DEVELOPMENT_001',f'images/{i:04d}/transmissions.npz') as dz, archive(DEEP,f'images/{i:03d}/waveforms.npz') as pz, archive(R3,f'images/{i:03d}/waveforms.npz') as rz:
            local=HISTORY/f'DEVELOPMENT_001/images/{i:04d}/per_frame.csv';bind(local,completed[HISTORY/'DEVELOPMENT_001']['output_hashes'][str(local.relative_to(HISTORY/'DEVELOPMENT_001'))])
            dr={(r['family'],*key(r)):r for r in rows(local) if int(r['mode'])==policy['actions'][r['family']]['quality'][r['snr_db']]}
            for s in SNRS:
                for n in (2001,2002,2003):
                    noise=seeded_noise(source['image_id'],n,(3060,2))
                    for family in ('raw','arithmetic'):
                        row=dr[family,i,s,n];header,payload=digital_ledger(row);tx=dz[f'signal_{family}_m{row["mode"]}'].astype(np.float64);rx=tx+noise/np.sqrt(10**(s/10))
                        require(array_sha(tx)==row['transmitted_sha256'] and array_sha(rx)==row['received_sha256'],'actual digital TX/RX hash')
                        require(int(row['image_index_in_archive'])==int(lookup[family+'_adaptive',i,s,n]['image_slot']),'selected digital image slot')
                        record(family+'_adaptive',row,tx,rx,noise,header,payload)
                    row=deep_rows[i,s,n];si=cfg['evaluation']['snrs_db'].index(s);ni=cfg['evaluation']['noise_seeds'].index(n)
                    tx=pz[f'frame_{si}_{ni}_deep_tx'];rx=pz[f'frame_{si}_{ni}_deep_rx']
                    require(int(row['header_uses'])==0 and int(row['data_uses'])==3060 and row['noise_sha256']==array_sha(noise),'Deep PHY identity')
                    require(int(row['image_ref'].split(':')[1])==int(lookup['perceptual_deepjscc',i,s,n]['image_slot']),'Deep image slot')
                    record('perceptual_deepjscc',row,tx,rx,noise,0)
                    row=r3_rows[i,s,n];name=f'r3__full_grid_prediction_features__snr{s}';tx=rz[name];rx=rz[name+f'__seed{n}']
                    require(row['checkpoint_sha256']==choice['checkpoint_sha256'] and int(row['selected_step'])==choice['step'],'selected R3 checkpoint')
                    require(row['decoder_interface']=='continuous_mean' and row['bit_metric_role']=='diagnostic_thresholded_logits_not_delivered_bits','R3 bits are diagnostics, not channel payload')
                    require(array_sha(tx)==row['transmitted_sha256'] and array_sha(rx)==row['received_sha256'] and row['noise_sha256']==array_sha(noise),'actual R3 TX/RX hash')
                    require(row['image_sha256']==lookup['wetok_r3',i,s,n]['image_sha256'],'R3 received image identity')
                    record('wetok_r3',row,tx,rx,noise,0)
        if i%20==0:print('REAL_N3060_WAVEFORMS',i+1,flush=True)
    require(len(checks)==6000,'complete audited waveform matrix')
    bind(Path(__file__));bind(input_root/'development_inputs.json')
    write(OUT/'full_bindings.json',bindings)
    PUBLIC.mkdir(parents=True)
    with (PUBLIC/'per_frame_phy.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
    scope=[]
    for method in METHODS:
        rr=[r for r in checks if r['method']==method]
        scope.append({'method':method,'frames':len(rr),'N':3060,'max_energy_error':max(abs(r['E']-6120) for r in rr),'max_single_noise_error':max(r['single_noise_max_abs_error'] for r in rr),'header_N':sorted({r['header_N'] for r in rr}),'current_uniform_timing':'NOT_RUN_REQUIRES_REAL_ONLINE_REPLAY','common_quality_metrics':'ALREADY_QUEUED_REFERENCE_COMMON_METRICS_DO_NOT_DUPLICATE'})
    write(PUBLIC/'audit.json',{'status':'REAL_N3060_ARCHIVED_WAVEFORM_AND_LINEAGE_AUDIT_PASS_FRESH_RUNTIME_TIMING_PENDING','synthetic':False,'GPU_used':False,'new_holdout':False,'rows':6000,'bound_files':len(bindings),'full_bindings_sha256':digest(OUT/'full_bindings.json'),'per_frame_sha256':digest(PUBLIC/'per_frame_phy.csv'),'summaries':scope,'source_changes':changes,'assets':assets,'calibration_policy_actions':policy['actions'],'policy_sha256':digest(policy_path),'pending':'actual loaded model/native parity plus uniform CPU endpoints TX/RX timing; source changes not evidence of historical model errors; paired publication after existing queues'})
    print('REAL_N3060_CPU_AUDIT_PASS',len(checks),len(bindings),len(changes))

if __name__=='__main__':main()
