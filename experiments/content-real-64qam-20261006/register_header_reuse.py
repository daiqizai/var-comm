"""Register narrowly scoped old-header probability reuse, without new decodes.

Read-only validation precedes creation of one new H/HEADER_PROBABILITY_REUSE.json.
Neither old probabilities/receipts nor the immutable initial H registration change.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

TABLE_SHA = 'e7e316ba722e2315261b4ba3c75aa0b4d33973fab09ad50978fe4b46909d4f52'
CPP_SHA = '035058221b35df245fe1520a0e14f741528b53d84c0a987359d0df0d226a7b52'
OLD_REF = '040e94d'
HEADER_KEY = 'uep-v1-profile12-crc16-tail6-conv-ratematch136-qpsk-Es2'
POINTS = {
    13: ('353617a689653e4439bca3d78677f08645b9c1c18774212de44c161ca27d5c75',
         '14d547ea2859ba99185752ff343ad08d0c490cf99e44064588dc7e62eb22b971'),
    19: ('2fa8eef84b8520b8cfa916a6f21089e0632720eaadaddbe26643cf2a42744307',
         'e2c9c493a6bca205c760c7ed645dd01a3aea22399b183c59f431ec6784db9192')}
SOURCES = ('experiments/prior-aware-grouped-mcs-v1/uep_phy.py',
           'src/var_comm/scale_channel.py','src/var_comm/token_trellis.cpp',
           'experiments/content-real-64qam-20261006/h64_phy.py')


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    # Exact extreme endpoints are algebraic0/1, preserving all old files as-is.
    return [0. if k==0 else max(0.,center-width),1. if k==n else min(1.,center+width)]


def validate_checkpoint(cp,snr,row):
    require(cp.get('synthetic') is False and cp['point']=={'phy_key':HEADER_KEY,'snr_db':snr},
            'Point is synthetic or has a different physical identity')
    require(cp['payload_sha256']==digest({k:v for k,v in cp.items() if k!='payload_sha256'}),
            'Point internal payload checksum differs')
    require(row['point_checkpoint_sha256']==cp['payload_sha256'],'Table row does not bind this actual point')
    cursor=0;totals=dict(n_blocks=0,n_correct=0,n_reject=0,n_undetected=0)
    for batch in cp['batches']:
        require(batch['start']==cursor and type(batch['n_blocks']) is int and 0<batch['n_blocks']<=128,
                'Actual packet ranges missing, overlapping, or outside old registered batches')
        for key in totals:
            require(type(batch[key]) is int and batch[key]>=0,'Invalid actual packet counts')
            totals[key]+=batch[key]
        require(batch['n_correct']+batch['n_reject']+batch['n_undetected']==batch['n_blocks'],
                'Actual C/R/U do not partition a batch')
        require(re.fullmatch('[0-9a-f]{64}',batch['event_bits_sha256']) is not None,'Missing original event-vector hash')
        require(batch['energy_sum']==136.*batch['n_blocks'] and batch['energy_sum_squares']==136.**2*batch['n_blocks']
                and batch['energy_min']==136. and batch['energy_max']==136.,'Original header is not fixed68 QPSK Es2')
        cursor+=batch['n_blocks']
    expected=dict(n_blocks=20000,n_correct=20000,n_reject=0,n_undetected=0)
    require(totals==expected and cp['counts']==expected and cursor==20000,'Expected exact20000 correct and no R/U')
    require(all(row[key]==value for key,value in expected.items()),'Row counts differ from complete point')
    return dict(sha256_of_internal_payload=cp['payload_sha256'],binding=cp['binding'],
                batches=len(cp['batches']),verified_packet_count=cursor)


def validate_header_row(row,snr):
    require(row['kind']=='header' and row['phy_key']==HEADER_KEY and row['snr_db']==snr,
            'Selected row is not the requested header point')
    require(row['source_id']=='*' and row['applicability']=='source_independent_verified','Different old applicability')
    require(row['physical_configuration']==dict(phy_key=HEADER_KEY,kind='header',source_bits=12,
        crc_bits=16,tail_bits=6,symbols=68,transmitted_bits=136,q=2,modulation='QPSK',profile_id_range=[0,4095]),
        'Paid header format changed')
    require(row['p_correct']==1. and row['p_reject']==0. and row['p_undetected']==0.
            and row['block_errors']==0 and row['zero_observed_errors_is_not_true_BLER_zero'] is True,
            'Old header empirical probabilities differ')
    for label,count in (('correct',20000),('reject',0),('undetected',0)):
        expected=wilson(count,20000)
        require(len(row['p_'+label+'_ci'])==2 and
            max(abs(a-b) for a,b in zip(row['p_'+label+'_ci'],expected))<1e-14,
            'Stored Wilson interval differs beyond known extreme endpoint roundoff')
    require(row['actual_energy_mean']==136. and row['actual_energy_sd']==0.
            and row['observed_average_energy_per_complex_symbol']==2.,'Header energy convention differs')


def point_contract(snr,row,proof):
    return dict(snr_db=snr,old_counts={k:row[k] for k in ('n_blocks','n_correct','n_reject','n_undetected')},
        old_raw_uniform_id_empirical_C_R_U=[1.,0.,0.],
        old_stored_individual_wilson95={k:row['p_'+k+'_ci'] for k in ('correct','reject','undetected')},
        p_header_correct_point=1.,p_header_correct_wilson95=wilson(20000,20000),
        zero_observed_errors_is_not_true_zero_failure=True,
        old_unrestricted_U_wilson95=wilson(0,20000),
        new_finite_codebook_U_exact_reuse=False,new_finite_codebook_R_exact_reuse=False,
        new_finite_codebook_U_probability=None,
        new_finite_codebook_U_upper_envelope=wilson(0,20000)[1],
        U_envelope_condition='Under the same message-independent symmetric header channel, legal-ID filtering can only remove unrestricted wrong CRC accepts; this is an uncertainty envelope, not a new measured U rate.',
        checkpoint=proof)


def build(root,registration,git_show=None):
    root=Path(root).resolve();registration=Path(registration).resolve();reg=read(registration)
    require(reg['status']=='H_EXECUTION_REVISION_REGISTERED' and reg['branch']=='H','Initial H execution registration differs')
    source_bindings={}
    for relative in SOURCES:
        path=root/relative;checksum=sha(path)
        require(reg['source_bindings'].get(str(path))==checksum,'Header source not bound by initial registration: '+str(path))
        source_bindings[str(path)]=checksum
    require(sha(root/'src/var_comm/token_trellis.cpp')==CPP_SHA,'Current trellis source does not have the verified exact hash')
    if git_show is None:
        def git_show(ref,relative):
            return subprocess.run(['git','show',ref+':'+relative],cwd=root,check=True,
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
    comparisons=[]
    for relative in SOURCES[:3]:
        old=git_show(OLD_REF,relative);old_hash=hashlib.sha256(old).hexdigest()
        require(old_hash==sha(root/relative),'Historical/current actual header source differs: '+relative)
        comparisons.append(dict(ref=OLD_REF,path=relative,historical_sha256=old_hash,current_sha256=old_hash,
                                exact_bytes_equal=True))
    require(comparisons[2]['historical_sha256']==CPP_SHA,'Historical trellis hash differs')
    base=root/'outputs/PRIOR-AWARE-UEP-20261004-V1';table_path=base/'bler_refined_02.json'
    require(sha(table_path)==TABLE_SHA,'Original selected refined02 table changed')
    table=read(table_path)
    require(table['status']=='REFINEMENT_COMPLETE' and table['synthetic'] is False
            and table['version']=='UEP-ACTUAL-LDPC-HEADER-BLER-V1','Not completed actual old table')
    require(table['registration_sha256']==digest(table['merge_registration']),'Old merge registration identity changed')
    shards=[s for s in table['merge_registration']['shards'] if s['shard_index']==0]
    require(len(shards)==1,'Original paid-header shard identity missing')
    inputs={str(registration):sha(registration),str(table_path):TABLE_SHA};points=[]
    for snr,(directory,checksum) in POINTS.items():
        rows=[r for r in table['rows'] if r['kind']=='header' and r['snr_db']==snr]
        require(len(rows)==1,'Duplicate/missing original header point')
        row=rows[0];validate_header_row(row,snr)
        folder=base/'phy_shards/0/points'/directory
        previous=None
        for name in ('checkpoint.json','refined.json'):
            path=folder/name
            require(sha(path)==checksum and table['checkpoint_bindings'].get(str(path))==checksum,
                    'Exact original point/snapshot file hash changed: '+str(path))
            cp=read(path);proof=validate_checkpoint(cp,snr,row)
            require(cp['binding']==shards[0]['registration_sha256'],'Actual point belongs to another old shard')
            if previous is not None:require(cp==previous,'Completed refined snapshot differs from checkpoint')
            previous=cp;inputs[str(path)]=checksum
        proof.update(checkpoint_path=str(folder/'checkpoint.json'),refined_path=str(folder/'refined.json'),file_sha256=checksum)
        points.append(point_contract(snr,row,proof))
    return dict(status='H_HEADER_CORRECT_PROBABILITY_REUSE_REGISTERED',schema='H_HEADER_REUSE_V1',
        main_snrs_db=[13,19],new_packet_decodes=0,new_ledger_charges=0,old_files_modified=False,
        scope='PSNR_PRESCREEN_ONLY; actual shortlisted/development headers still undergo paid real decoding',
        input_bindings=inputs,source_bindings=source_bindings,
        helper_source_sha256=sha(__file__),historical_source_comparisons=comparisons,points=points,
        correct_probability_compatibility=dict(
            reused='Only the probability of exact recovery of a transmitted legal12-bit profile ID',
            old_population='Uniform raw profile IDs0..4095, receiver codebook=None',
            new_population='Only transmitted IDs in the finite H codebook, unknown decoded IDs rejected',
            basis='Same terminated linear/affine CRC+convolutional waveform, rate matching, symmetric QPSK AWGN and uninformative trellis metric; old source-independent C model is reused. A correctly decoded transmitted legal ID always passes membership.',
            not_same_measured_sample='Old uniform IDs are not replayed through the new finite codebook; no claim of identical old/new C/R/U sample vectors',
            restrictions='Finite membership can move wrong CRC accepts from U to R; old R/U are not exact new-codebook measurements. No free correction from true profile ID.'),
        prescreen=dict(
            estimator='pH_C*pB_C*mean_source_PSNR_of_correct_state + (1-pH_C*pB_C)*mean_source_gray_PSNR',
            header_C_point_source='points[].p_header_correct_point',
            uncertainty='Propagate individual header C Wilson95 bounds together with separately measured body uncertainty; these are not simultaneous95% candidate bounds.',
            unknown_accepted='Gray PSNR is the already registered screening surrogate only; actual wrong accepted/parseable outputs must be decoded and rendered in real-image validation.',
            selection_updates='No new samples, no C-REAL weights updated, no protocol or candidate objective changed'),
        diagnostic_16dB=dict(status='NOT_MEASURED',header_probability=None,
            action='Body-only physical diagnostic; no interpolation of header data and no full-link score at16dB'))


def seal_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(read(path)==value,'Existing header reuse registration differs; do not overwrite')
        return 'VERIFIED_EXISTING'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return 'CREATED'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--registration',required=True)
    parser.add_argument('--out',help='H output directory; defaults to the independent H branch')
    args=parser.parse_args();root=Path(args.root).resolve()
    out=Path(args.out).resolve() if args.out else root/'outputs/CONTENT-REAL-64QAM-20261006/H'
    require(out==root/'outputs/CONTENT-REAL-64QAM-20261006/H','Write only the independent registered H receipt directory')
    value=build(root,args.registration)
    path=out/'HEADER_PROBABILITY_REUSE.json';state=seal_new(path,value)
    print(json.dumps(dict(status=value['status'],action=state,path=str(path),sha256=sha(path),new_packet_decodes=0)))


if __name__=='__main__':main()
