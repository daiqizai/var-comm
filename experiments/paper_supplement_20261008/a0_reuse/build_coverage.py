"""Update A0 from local receipts only; never dispatch or recompute science."""
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/paper_supplement_20261008/a0_reuse'
SUP=ROOT/'results/paper_supplement_20261008'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_read(p):
    with Path(p).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def verify_outputs(receipt,local_root,remote_root):
    done=read(receipt);verified=[];remote_only=[];mismatch=[]
    for source,expected in done.get('outputs',{}).items():
        suffix=source[len(remote_root):].lstrip('/')if source.startswith(remote_root)else source
        path=Path(local_root)/suffix
        if not path.exists():remote_only.append(source)
        elif sha(path)!=expected:mismatch.append(dict(path=rel(path),expected=expected,actual=sha(path)))
        else:verified.append(rel(path))
    return dict(receipt=rel(receipt),receipt_sha256=sha(receipt),listed_outputs=len(done.get('outputs',{})),
        local_hash_verified=len(verified),remote_only_not_downloaded=len(remote_only),mismatches=mismatch)

def build():
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    old=OUT/'reuse_matrix.csv';old_sha=sha(old);old_readme_sha=sha(OUT/'README.md')
    rows=csv_read(old)
    for row in rows:
        row['execution_state']='PRIOR_SNAPSHOT_REUSED';row['snapshot_updated_at']=now
        row['evidence_level']='PRIOR_LOCAL_EVIDENCE';row['first_batch_hard_gap']='false'
    index={r['item_id']:r for r in rows}
    def update(item,status,evidence,scope,missing='',entry='',state='COMPLETE',**extra):
        r=index[item];r.update(status=status,execution_state=state,evidence_file=rel(evidence),
            evidence_sha256=sha(evidence),cache_location=rel(Path(evidence).parent),available_scope=scope,
            missing_scope=missing,minimum_next_entry=entry,evidence_level='ACTUAL_LOCAL_RECEIPT',**extra)
    def append(task,method,status,evidence,scope,dataset='not applicable',snr='not applicable',sources='',repetitions='',
               receiver='not applicable',metrics='implementation/delivery',state='COMPLETE',missing='',entry='',hard=False):
        item=f'A0-{len(rows)+1:03d}';r={k:''for k in rows[0]};r.update(item_id=item,task=task,method=method,dataset=dataset,
            N=1024 if snr!='not applicable'else '',snr_db=snr,source_count=sources,noise_count_or_repetitions=repetitions,
            receiver=receiver,metrics_or_artifact=metrics,snapshot_updated_at=now,first_batch_hard_gap=str(hard).lower())
        rows.append(r);index[item]=r;update(item,status,evidence,scope,missing,entry,state)
    checks=[];audits=[]
    def check(name,status,evidence,details,limitation=''):
        paths=[Path(p)for p in evidence];checks.append(dict(check=name,status=status,
            evidence=[dict(path=rel(p),sha256=sha(p))for p in paths],details=details,limitation=limitation))
    native_pairs=SUP/'native_bpg_paired_v1';native_done=read(native_pairs/'completion.json')
    native_audit=verify_outputs(native_pairs/'completion.json',ROOT,'');audits.append(native_audit)
    require(native_done['status']=='NATIVE_BPG_NEW_SOURCE_PAIRED_STATISTICS_COMPLETE_V1'
        and native_done['new_paired_rows']==72 and native_done['draws_calls']==1
        and native_done['draws_shape']==[10000,500]and native_done['inputs_unchanged']
        and not native_done['old_summary_recomputed']and not native_done['old_bootstrap_recomputed']
        and native_audit['local_hash_verified']==3 and not native_audit['mismatches'],'Actual new native-BPG paired closure')
    update('A0-056','COMPLETE',native_pairs/'completion.json',
        '72 actual NEW source-paired contrasts:native256 BPG minus published Proposed/P/Swin;500 same source IDs;3-noise means first;10k shared resampling indices',
        notes='Post-hoc supplementary comparisons;not shared noisy observations;unchanged original single-method means/CIs;correct DINOv2-L and LPIPS sign.')
    check('Native256 BPG new source-paired reference comparisons','PASS',[native_pairs/'completion.json',native_pairs/'pins.json',native_pairs/'paired.csv'],
        dict(new_paired_rows=72,source_count=500,noise_count=3,input_source_means=48000,draws_calls=1,shared_draws_shape=[10000,500],
            seed=2026100701,new_metric_calls=0,new_PHY=0,old_summary_recomputed=False,old_bootstrap_recomputed=False),
        '72 newly authorized pointwise95% comparisons;no multiplicity-adjusted global winner claim;Swin19dB remains outside training/calibration.')
    # A1: direct, finite metadata from actual completed stages.
    a1=SUP/'a1_bpg_adaptive';probe=a1/'a1a_probe_v1/completion.json';pr=read(probe)
    require(pr['status']=='A1A_CALIBRATION32_BPG_CODEC_PROBE_COMPLETE_V1'and pr['source_count']==32 and pr['tested_case_count']==3076,'Actual A1a completion')
    pp=csv_read(probe.parent/'cases.csv');errors=[abs(float(x['psnr_db'])+10*math.log10(float(x['mse'])))for x in pp]
    require(len(pp)==3076 and max(errors)<1e-10 and all(int(x['source_bits'])==int(x['complete_BPG_bytes'])*8 for x in pp),'A1a MSE/PSNR/file byte domain')
    audits.append(verify_outputs(probe,probe.parent,'/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/a1a_probe_v1'))
    update('A0-057','COMPLETE',probe,'32 original calibration sources;3076 real source-codec cases;256/128/64/32;complete stream lengths and per-image MSE/PSNR/cost',state='COMPLETE')
    check('BPG adaptive source bytes and per-image MSE/PSNR','PASS',[probe,probe.parent/'cases.csv'],
          dict(cases=3076,source_count=32,maximum_PSNR_formula_error=max(errors),bytes_times_8_exact=True),
          'Local probe summaries are verified; large individual stream archives remain on original host, not all downloaded.')
    stage_root=a1/'a1b_adaptive_v1';request=a1/'a1b_materials_v1/request.json';request_sha=sha(request)
    actual={}
    for stage in ['qualification','screen','codec_calibration','calibration','freeze']:
        receipt=stage_root/stage/'completion.json';d=read(receipt);actual[stage]=d
        require(d['status']=='ADAPTIVE_BPG_STAGE_COMPLETE_V1'and d['actual_children_waited']
            and all(code==0 for code in d['worker_exit_codes'])and d['root_budget_before']==d['root_budget_after']
            and d['ledger']['unresolved']==0 and d['request_sha256']==request_sha,'A1 actual stage closure '+stage)
        audits.append(verify_outputs(receipt,stage_root,'/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/a1b_adaptive_v1'))
    qpath=stage_root/'qualification/worker_0.json';q=read(qpath)
    source_checks=[x for x in q['checks']if 'source_index'in x]
    require(len(source_checks)==20 and all(x['roundtrip_exact']and x['received_bytes_only']for x in source_checks),'20 real received-only BPG source roundtrips')
    require(len(q['checks'])==22 and all(x['roundtrip_exact']for x in q['checks'])and len(q['layouts'])==15,'New profile and inherited legal layouts')
    require(actual['qualification']['ledger']['total']==44 and actual['screen']['ledger']['total']==3116
            and actual['calibration']['frame_count']==5400 and actual['freeze']['ledger']['total']==13916,'Actual bounded A1 counts')
    fpath=stage_root/'freeze/worker_0.json';fr=read(fpath)
    require(fr['calibration_source_count']==100 and fr['noise_count']==3 and not fr['holdout_used_for_selection'],'Calibration-only strategy freeze')
    check('BPG adaptive noiseless complete-stream/header/container roundtrip','PASS',[qpath,stage_root/'qualification/completion.json'],
        dict(real_calibration_sources=20,new_profile_patterns=2,new_PHY_decodes=44,legal_layouts=15,old_14_layouts_reused=True,received_bytes_only=True))
    check('BPG calibration policy and protected original ledger','PASS',[stage_root/'calibration/completion.json',fpath,stage_root/'freeze/completion.json'],
        dict(calibration_frames=5400,calibration_sources=100,noise_count=3,qualification_calls=44,prescreen_calls=3072,
            calibration_calls=10800,total_calls=13916,root_ledger_unchanged=True,holdout_used_for_selection=False,selected_profile_ids=fr['selected_profile_ids']))
    for stage,label in [('qualification','actual20 source/container checks'),('screen','actual matched-BLER prescreen'),('codec_calibration','fresh/verified source choices for100cal'),('calibration','actual5400 calibration frames'),('freeze','calibration-only six-SNR frozen policies')]:
        append('A1b','BPG adaptive: '+label,'COMPLETE',stage_root/stage/'completion.json',label,
            dataset='old calibration',snr='1|4|7|10|13|19',sources=100 if stage in ['codec_calibration','calibration','freeze'] else 20 if stage=='qualification' else 32,repetitions=3)
    update('A0-058','MISSING',stage_root/'freeze/completion.json','Source probe, qualification, prescreen,100cal actual links and frozen MCS COMPLETE;13916 actual paid decodes',
        'Adaptive500 holdout normal completion;four-core quality and source-paired statistics;figures/resource/failure export',
        'Reuse existing controller; wait for actual codec_holdout/holdout completion then run prepared metrics stage',
        'PENDING_RUNNING',first_batch_hard_gap='true',notes='Coordinator last observed codec_holdout PID3293663 and controller3253540 running. This local audit does not assert current liveness and never schedules a duplicate.')
    # A2: all44 outputs including original float arrays, and independent formula checks.
    a2=SUP/'a2_swin/native_v1';d=read(a2/'completion.json')
    au=verify_outputs(a2/'completion.json',a2,'/home/liulu/projects/VAR_COMM/outputs/PAPER-SUPPLEMENT-20261008/a2_swin_native_v1');audits.append(au)
    require(au['local_hash_verified']==44 and not au['mismatches'],'All44 Swin outputs')
    mse_errors=[];psnr_errors=[]
    for i in range(20):
        case=read(a2/f'cases/{i:04d}.json')
        with np.load(a2/f'cases/{i:04d}.npz',allow_pickle=False)as z:
            source=z['source_u8'].astype(np.float64)/255;rgb=z['native_rgb'];wrapper=z['wrapper_rgb']
            require(rgb.dtype==wrapper.dtype==np.float32 and rgb.shape==wrapper.shape==(3,256,256)
                and np.isfinite(rgb).all()and rgb.min()>=0 and rgb.max()<=1,'Actual Swin RGB range')
            mse=float(np.mean((source-rgb.astype(np.float64))**2));mse_errors.append(abs(mse-case['native_mse_01']))
            psnr_errors.append(abs(-10*math.log10(mse)-case['native_source_psnr_db']))
    require(max(psnr_errors)<1e-4,'Actual Swin per-image MSE/PSNR relation')
    update('A0-059','COMPLETE',a2/'completion.json','20 actual native/paid-wrapper same-context checks PASS;all44 files locally hash verified;RGB range and per-image PSNR rechecked',
           notes='Free-context implementation check,0paidPHY;not a noisy paid-information comparison.')
    check('Swin native wrapper same-observation/RGB/MSE alignment','PASS',[a2/'completion.json',a2/'native_parity_summary.json'],
        dict(source_count=20,hash_verified_outputs=44,max_independent_MSE_difference=max(mse_errors),max_independent_PSNR_difference=max(psnr_errors),new_paid_PHY=0),
        'Known mask/power were shared only for native diagnostic; fixed256-use header is not proven globally optimal.')
    check('Swin trained support and no unauthorized retraining','PASS',[a2/'support_scope.json'],read(a2/'support_scope.json'))
    # A3 verified complete, including honest SOURCE_UNFIT and required filename alias.
    a3=SUP/'a3_timing';audit=read(a3/'final_v1/audit.json')
    require(audit['status']=='COMPLETE'and len(audit['completed_methods'])==5 and sum(x['verified_output_files']for x in audit['completed_methods'])==1038,'Actual A3 five methods')
    aliases=a3/'final_v1';require(sha(aliases/'timing_table.csv')==sha(aliases/'timing_single_output.csv'),'Required filename content alias')
    for num,method in [(81,'RAW64_PARTIAL'),(82,'RAW64_WHOLE'),(83,'P1024'),(84,'SwinJSCC80k'),(85,'BPG_LDPC_native256')]:
        p=a3/'actual_v1/results'/method/'completion.json';update(f'A0-{num:03d}','COMPLETE',p,
            '16 old development sources;7/13/19;warm1+measure3;actual TX/RX/outer E2E;P95;peak allocated/reserved;model storage',
            notes='One actual outer window;channel separately timed/subtracted;not TX+RX or air latency. BPG source-unfit RX/E2E is NA.')
    check('Five-method timing fresh single-output evidence and naming','PASS',[a3/'final_v1/audit.json',aliases/'timing_single_output.csv'],
        dict(actual_methods=5,source_attempts=960,actual_PHY_decodes=1080,hash_verified_files=1038,independent_E2E_second_pass=False,actual_outer_window=True,
            BPG_source_unfit_attempts=132,BPG_actual_link_frames=60,filename_alias_byte_identical=True),
        'BPG link times are conditional on fit;full loaded VAE scope is disclosed;not minimal deployment or energy evidence.')
    # A4 preserves the original receipt and uses the closed delivery sidecar.
    a4=SUP/'a4_resources/v2';au=verify_outputs(a4/'completion.json',a4,'/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a4_resources/v2');audits.append(au)
    require(all(x['path'].endswith('/console.log')for x in au['mismatches']),'A4 scientific output mismatch')
    closed=verify_outputs(a4/'DELIVERY_MANIFEST.json',a4,'/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a4_resources/v2');audits.append(closed)
    require(closed['local_hash_verified']==44 and not closed['mismatches']and not closed['remote_only_not_downloaded'],'A4 closed delivery all44 hashes')
    analysis=a4.parent/'analysis_v1';binding=read(analysis/'source_bindings.json')
    require(binding['status']=='READ_ONLY_ANALYSIS_COMPLETE'and not binding['scientific_CSV_files_modified']
        and sha(analysis/'MANUSCRIPT_ANALYSIS.md')==binding['analysis_sha256']
        and all(sha(p)==h for p,h in binding['inputs'].items()),'A4 source-bound read-only diagnostic analysis')
    for num in list(range(88,112))+[113,114,115]:
        update(f'A0-{num:03d}','COMPLETE',a4/'completion.json','Actual500-source×3-noise resources/energy/token/failure/state-quality and4 figure groups;published means exact',
               notes='Original receipt preserved;closed DELIVERY_MANIFEST verifies44 files including final log and render revision;scientific hashes unchanged.')
    check('A4 reuse and published-mean identity','PASS',[a4/'completion.json',a4/'published_mean_checks.csv',a4/'DELIVERY_MANIFEST.json',analysis/'source_bindings.json'],
        dict(source_count=500,noise_count=3,frame_memberships=36000,quality_rows=54000,configuration_rows=24,new_PHY=0,new_inference=0,new_bootstrap=0,
            original_hash_verified_outputs=au['local_hash_verified'],original_non_scientific_manifest_mismatches=au['mismatches'],closed_delivery_hash_verified_outputs=44,
            manuscript_source_bindings_verified=len(binding['inputs'])),
        'Original console.log mismatch is retained in the historical receipt and explicitly closed by the44-file sidecar;no science file changed.')
    append('A4','Low-SNR diagnostic case explanation','COMPLETE',analysis/'MANUSCRIPT_ANALYSIS.md',
        'Actual1dB source1/seed6202 diagnostic:CRC-rejected,12/73 wrong tokens,prefix5;paired19dB source diagnosis;retrospective selection disclosed',
        dataset='common500 existing cached frames',snr='1|4',sources=500,repetitions=3,
        metrics='one clearly labelled diagnostic example and low-SNR failure explanation')
    # A5 mechanism and six-column comparison both have closed actual evidence.
    a5=ROOT/'paper/figures/a5_mechanism_N1024_10_19dB_group02';au=verify_outputs(a5/'completion.json',a5,'');audits.append(au)
    require(au['local_hash_verified']==6 and not au['mismatches'],'A5 six figure files')
    metrics=csv_read(a5/'image_metrics.csv');require(len(metrics)==24 and {int(x['snr_db'])for x in metrics}=={10,19},'A5 per-image metric scope')
    for num in [132,133]:update(f'A0-{num:03d}','COMPLETE',a5/'completion.json','Fixed16 entries5–8;10/19dB four-column mechanisms;24 existing per-image metric rows;6 PDF/SVG/PNG files;qualitative notes',
        notes='Development diagnostic;fixed noise6201;no inference/quality/bootstrap. Direct uses zero absent residual contribution,not token index0.')
    for num in range(135,141):update(f'A0-{num:03d}','REUSE',a5/'reference_summary_all_metrics.csv','Existing100-source six-condition126-row reference metrics exported at original precision',
        notes='Reference levels do not map DINO to semantic correctness percentage.')
    six=ROOT/'paper/figures/a5_six_methods_N1024_13dB_group02';six_remote='/home/liulu/projects/VAR_COMM/paper/figures/a5_six_methods_N1024_13dB_group02'
    sd=read(six/'completion.json');bd=read(six/'bpg_metrics_completion.json');cd=read(six/'missing_convnext/completion.json')
    for filename,count in [('completion.json',15),('bpg_metrics_completion.json',33),('missing_convnext/completion.json',9)]:
        checked=verify_outputs(six/filename,six,six_remote);audits.append(checked)
        require(checked['local_hash_verified']==count and not checked['mismatches']and not checked['remote_only_not_downloaded'],'A5 closed child/top-level outputs '+filename)
    require(sd['status']==bd['status']==cd['status']=='COMPLETE'and sd['display_sources']==[4,21,24,29]
        and not sd['holdout_used']and not sd['old_images_changed']and not sd['policy_selection']and sd['remaining_missing_metric_cells']==0,'A5 fixed-sample complete comparison')
    require(bd['source_count']==bd['frame_count']==bd['total_quality_calls']==bd['total_reference_preparations']==16
        and bd['new_PHY_calls']==bd['new_bootstrap']==bd['reconstruction_model_calls']==bd['training_updates']==0,'A5 new cached BPG metric scope')
    require(cd['new_reconstruction_predictions']==cd['resolved_missing_cells']==8 and cd['new_reference_predictions']==0
        and cd['reused_A5_BPG_reference_predictions']==4 and cd['existing_metrics_recomputed']==cd['new_PHY_calls']==cd['new_reconstructions']==0,'A5 missing-only ConvNeXt scope')
    display_rows=csv_read(six/'per_image_metrics.csv');display_values=[x for x in display_rows if x['method']!='source']
    core={'psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction'}
    def metric_number(row):
        value=row['value']
        if row['metric']=='convnext_top1_source_prediction'and value.lower()in ['true','false']:
            return float(value.lower()=='true')
        return float(value)
    require(len(display_rows)==96 and len(display_values)==80 and len({(x['source_index'],x['method'],x['metric'])for x in display_rows})==96
        and {x['metric']for x in display_rows}==core and all(math.isfinite(metric_number(x))for x in display_values),'A5 complete unique four-core metric rows')
    require(all(metric_number(x)in [0,1]for x in display_values if x['metric']=='convnext_top1_source_prediction'),'A5 source prediction agreement is0/1')
    update('A0-134','COMPLETE',six/'completion.json','Actual six-column13dB development figure from fixed4 sources;native256 BPG measured rep1 appended;15 top-level outputs hash verified',
        notes='Original five-column pixels unchanged;BPG SOURCE_UNFIT gray is no received frame;not adaptive BPG and not common500;no shared-observation claim.')
    append('A5','13dB comparison per-image original metrics','COMPLETE',six/'per_image_metrics.csv',
        '96 display rows:16 unscored source references+80 finite metric values;56 reused+16 new cached BPG+8 missing-only ConvNeXt;remaining missing0',
        dataset='fixed development16 entries5–8',snr='13',sources=4,repetitions=1,
        metrics='PSNR,LPIPS,DINOv2-L cosine,ConvNeXt source-prediction agreement')
    check('A5 fixed samples, direct semantics and reference reuse','PASS',[a5/'completion.json',a5/'selection.json',a5/'image_metrics.csv',a5/'QUALITATIVE_OBSERVATIONS.md'],
        dict(SNRs=[10,19],source_count=4,metric_rows=24,figure_files=6,new_inference=0,new_metrics=0,new_bootstrap=0),
        'HiFi is available on these development diagnostics;its common500 absence remains explicit.')
    check('A5 actual13dB six-column delivery and missing-only metrics','PASS',[six/'completion.json',six/'bpg_metrics_completion.json',six/'missing_convnext/completion.json',six/'per_image_metrics.csv'],
        dict(display_sources=[4,21,24,29],SNR=13,N=1024,top_level_hashes=15,BPG_metric_hashes=33,missing_ConvNeXt_hashes=9,
            display_metric_values=80,reference_rows=16,BPG_new_quality_calls=16,BPG_reference_preparations=16,missing_ConvNeXt_predictions=8,
            missing_ConvNeXt_new_reference_predictions=0,remaining_missing=0,new_PHY=0,new_reconstructions=0,new_bootstrap=0),
        'Metrics-only additions;fixed development samples,not holdout selection;BPG SOURCE_UNFIT retained;different methods do not share one observed waveform.')
    # A6 is a completed literature/code comparison, not native inference.
    a6=SUP/'a6_neighbors';require(read(a6/'STATIC_VERIFICATION.json')['verification_status']=='COMPLETE','A6 static verification')
    check('A6 nearest-neighbor identities and reproducibility scope','PASS',[a6/'comparison.csv',a6/'availability.json',a6/'STATIC_VERIFICATION.json'],
        dict(comparison_rows=18,ARPC_native='NOT_RUN',Ada_TokenCom_native='NOT_RUN',ARPC_paid_N1024='UNSUPPORTED',new_weights_downloaded=False),
        'Native real-case/cost still conditional on environment,weights and input binding;not required to label first-round literature table complete.')
    # Static receiver boundary review is scoped honestly and uses actual roundtrips as its execution evidence.
    check('New BPG receiver does not consume source truth','PASS',[ROOT/'experiments/paper_supplement_20261008/a1_bpg_adaptive/adaptive_codec.py',
        ROOT/'experiments/paper_supplement_20261008/a1_bpg_adaptive/adaptive_stages.py',qpath],
        dict(source_truth_use='TX source-quality selection and offline diagnostics only',receiver_input='received IQ/public counter;header-derived profile and received complete BPG bytes',
            real_received_bytes_only_checks=20), 'Static boundary review plus concrete20-case equality;not a formal noninterference proof.')
    check('Original RAW path not requalified','REUSE',[a3/'final_v1/audit.json'],
        dict(raw_backend_qualification_reused=True,new_RAW_qualification_packets=0,raw_scientific_holdout_rerun=False))
    append('A5','Additional adaptive-BPG13dB fixed-development comparison','NOT_RUN',fpath,
        'Frozen A1 profile3009 exists;coordinator authorized only16 fixed development sources at13dB and a separate32-PHY cap;entry being prepared',
        dataset='fixed development16;display entries5–8',snr='13',sources=16,repetitions=1,
        metrics='actual adaptive-BPG reconstruction and new six-column figure;preserve native256 six-column delivery',state='PREPARED',
        missing='Actual16 adaptive development frames,quality metrics and separate adaptive six-column export',
        entry='Use coordinator-owned additional-development entry after actual closure;do not duplicate A1 controller or mutate original ledger',hard=True)
    # Later actual receipts close individual stages without treating reception as metrics/figures.
    final_state=dict(adaptive_holdout=False,adaptive_resources=False,adaptive_metrics=False,adaptive_figures=False,
        adaptive_development_receive=False,adaptive_development_export=False)
    for stage in ['codec_holdout','holdout']:
        receipt=stage_root/stage/'completion.json'
        if not receipt.exists():continue
        d=read(receipt)
        require(d['status']=='ADAPTIVE_BPG_STAGE_COMPLETE_V1'and d['actual_children_waited']and d['worker_exit_codes']==[0]*8
            and d['root_budget_before']==d['root_budget_after']and d['ledger']['unresolved']==0 and d['request_sha256']==request_sha,'Actual final A1 stage '+stage)
        checked=verify_outputs(receipt,stage_root,'/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/a1b_adaptive_v1');audits.append(checked)
        require(not checked['mismatches'],'Downloaded A1 stage output differs')
        if stage=='holdout':
            require(d['frame_count']==9000 and d['ledger']['total']==31916,'Actual9000-frame bounded holdout')
            final_state['adaptive_holdout']=True
            append('A1b','Adaptive BPG actual500-source holdout reception','COMPLETE',receipt,
                '9000 real frames;8 actual child wait-zero;18000holdout/31916total paid decodes;original ledger unchanged',
                dataset='common500 holdout',snr='1|4|7|10|13|19',sources=500,repetitions=3,metrics='actual received frames;quality delivery is separate')
            check('Actual adaptive500 holdout and bounded decoder accounting','PASS',[receipt],
                dict(frame_count=9000,holdout_decodes=18000,total_decodes=31916,total_cap=34604,unresolved=0,children_exit_codes=[0]*8,
                    original_ledger_unchanged=True,local_output_hashes=checked['local_hash_verified'],remote_only_outputs=checked['remote_only_not_downloaded']),
                'Normal completion and any downloaded outputs verified;large remote-only receiver caches are not counted as local hash checks.')
    resource=stage_root/'resources_v1'
    if (resource/'completion.json').exists():
        d=read(resource/'completion.json');checked=verify_outputs(resource/'completion.json',resource,
            '/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/a1b_adaptive_v1/resources_v1');audits.append(checked)
        require(d['status']=='ADAPTIVE_BPG_READ_ONLY_RESOURCE_EXPORT_COMPLETE_V1'and d['frame_count']==9000
            and d['source_configuration_count']==3000 and d['stream_sample_count']==6
            and checked['local_hash_verified']==13 and not checked['mismatches'],'Actual adaptive resource/stream export')
        final_state['adaptive_resources']=True
        append('A1c','Adaptive BPG resources, failures and stream samples','COMPLETE',resource/'completion.json',
            '9000 frame resource rows;3000 source settings;6 fixed source0 complete-stream examples;13 local hashes',
            dataset='common500 holdout',snr='1|4|7|10|13|19',sources=500,repetitions=3,metrics='resources/failures/source lengths/stream samples')
    metric_dir=a1/'metrics_v1'
    if (metric_dir/'completion.json').exists():
        d=read(metric_dir/'completion.json');checked=verify_outputs(metric_dir/'completion.json',metric_dir,
            '/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1');audits.append(checked)
        require(d['status']=='ADAPTIVE_BPG_FOUR_METRICS_NEW_PAIRED_STATISTICS_COMPLETE_V1'and d['frame_count']==9000
            and d['source_count']==500 and d['summary_rows']==24 and d['paired_rows']==96 and d['source_mean_cells']==12000
            and not d['old_summary_recomputed']and not d['old_bootstrap_recomputed']and d['new_packet_decodes']==0
            and d['root_budget_before']==d['root_budget_after']and not checked['mismatches'],'Actual adaptive metrics closure')
        summary=csv_read(metric_dir/'summary.csv');paired=csv_read(metric_dir/'paired.csv');means=csv_read(metric_dir/'source_means.csv')
        comparison=csv_read(metric_dir/'comparison_summary.csv');native_ids=read(SUP/'native_bpg_paired_v1/pins.json')['source_ids']
        require(d['source_ids']==native_ids and len(summary)==24 and len(paired)==96 and len(means)==12000 and len(comparison)==120,'Adaptive complete metric populations')
        require(len({(r['point_id'],r['metric'],r['source_index'])for r in means})==12000
            and all(r['source_id']==native_ids[int(r['source_index'])]for r in means),'Adaptive exact source identities')
        source_grids={}
        for row in means:source_grids.setdefault((row['point_id'],row['metric']),set()).add(int(row['source_index']))
        require(set(source_grids)=={(r['point_id'],r['metric'])for r in summary}
            and all(indices==set(range(500))for indices in source_grids.values()),'Every adaptive metric/SNR contains each500 source index exactly once')
        require(all(int(r['source_count'])==500 and int(r['noise_count'])==3 and float(r['ci_low'])<=float(r['mean'])<=float(r['ci_high'])
            for r in summary+paired+comparison),'Adaptive intervals/denominator')
        require({r['metric']for r in summary}==core and all(r['delta_definition']=='method minus reference'for r in paired),'Adaptive metric identity and pair direction')
        adaptive_prefix='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024_SNR_'
        reference_prefixes=['RAW64_PARTIAL_VAR_COMPLETION_SNR_','P1024_SNR_','SWIN80K_N1024_SNR_','BPG_LDPC_N1024_SNR_']
        snrs=[1,4,7,10,13,19]
        require({(r['point_id'],r['metric'])for r in summary}=={(adaptive_prefix+str(s),m)for s in snrs for m in core}
            and {(r['method'],r['reference'],r['metric'])for r in paired}=={(adaptive_prefix+str(s),ref+str(s),m)for ref in reference_prefixes for s in snrs for m in core},
            'Exact adaptive six-SNR method/reference grid;no duplicate or substituted comparison')
        preserved=read(SUP/'native_bpg_paired_v1/pins.json')['inputs']
        old_summaries=read(ROOT/preserved['published_summary_unchanged']['path'])+csv_read(ROOT/preserved['native_summary_unchanged']['path'])
        old_summary_map={(r['point_id'],r['metric']):r for r in old_summaries}
        reused=[r for r in comparison if r['provenance']!='ADAPTIVE_BPG_NEW_BASELINE']
        require(len(reused)==96 and all(all(float(r[k])==float(old_summary_map[r['point_id'],r['metric']][k])for k in ['mean','ci_low','ci_high'])for r in reused),
            'All96 old comparison estimates copied exactly without recomputation')
        final_state['adaptive_metrics']=True
        append('A1c','Adaptive BPG four-core quality and new paired statistics','COMPLETE',metric_dir/'completion.json',
            '24 new summary rows;96new paired contrasts;12000source means;120comparison rows;old statistics untouched',
            dataset='common500 holdout',snr='1|4|7|10|13|19',sources=500,repetitions=3,metrics='PSNR/LPIPS/DINOv2-L/ConvNeXt source agreement')
        check('Adaptive complete quality and new source-paired statistics','PASS',[metric_dir/'completion.json',metric_dir/'summary.csv',metric_dir/'paired.csv'],
            dict(new_summary_rows=24,new_paired_rows=96,source_means=12000,source_count=500,noise_count=3,
                new_unique_quality_calls=d['new_unique_quality_calls_total'],inherited_native_identical_image_reuses=d['inherited_native256_same_image_reuses'],
                reference_preparations=d['new_reference_preparations_this_run'],new_PHY=0,old_statistics_recomputed=False,
                local_hash_verified=checked['local_hash_verified'],remote_only=checked['remote_only_not_downloaded'],old_comparison_mean_CI_triples_exact=96),
            'New post-hoc baseline and pair intervals only;source pairing does not imply shared waveform/noisy observations.')
    figures=ROOT/'paper/figures/adaptive_bpg_holdout500'
    if (figures/'validation.json').exists():
        vd=read(figures/'validation.json');visual=read(figures/'visual_validation.json')
        require(vd['status']=='COMPLETE'and not vd['issues']and len(vd['output_files'])==30
            and all(sha(figures/p)==h for p,h in vd['output_files'].items())
            and all(sha(metric_dir/p)==h for p,h in vd['input_hashes'].items()),'Adaptive30 figure formats and bound plot inputs')
        require(visual['status']=='PASS'and len(visual['actually_opened_PNGs'])==2
            and all(sha(figures/p)==h for p,h in visual['actually_opened_PNGs'].items()),'Actual combined PNG review matches delivered pixels')
        final_state['adaptive_figures']=True
        check('Adaptive actual five-method curves and source-paired plots','PASS',[figures/'validation.json',figures/'plot_data.csv',figures/'visual_validation.json'],
            dict(vector_PDF=10,editable_SVG=10,PNG600dpi=10,main_rows=120,adaptive_minus_native_rows=24,new_bootstrap=0),
            'Actual PNG visual review is separately recorded in the delivery conversation;no generated raster curves.')
    if final_state['adaptive_holdout']:
        missing=[]
        for label,key in [('resource/failure/stream exports','adaptive_resources'),('four-core quality and96paired contrasts','adaptive_metrics'),('six-SNR curves and paired figures','adaptive_figures')]:
            if not final_state[key]:missing.append(label)
        complete=not missing
        update('A0-058','COMPLETE'if complete else'MISSING',metric_dir/'completion.json'if final_state['adaptive_metrics']else stage_root/'holdout/completion.json',
            'Actual500x6x3 receiver complete;31916/34604paid,original ledger unchanged;completed delivery stages='+','.join(k for k,v in final_state.items()if v),
            ';'.join(missing),''if complete else'Reuse finished reception;finish only existing downstream metric/figure delivery',
            'COMPLETE'if complete else'PENDING_RUNNING',first_batch_hard_gap=str(not complete).lower(),
            notes='Later-added baseline;frozen old100calPSNR policy;all failures retained;no old holdout re-run or winner claim.')
    adaptive_receive=SUP/'a5_examples/adaptive_actual_v1/completion.json'
    if adaptive_receive.exists():
        d=read(adaptive_receive)
        require(d['status']=='A5_ADAPTIVE_FIXED16_ACTUAL_CHILDREN_WAIT_ZERO'and d['actual_children_waited']and d['worker_exit_codes']==[0,0]
            and d['source_count']==d['frame_count']==16 and d['SNRs']==[13]and d['noise_seeds']==[2001]and d['profile_id']==3009
            and d['independent_ledger']['total']==32 and d['root_budget_before']==d['root_budget_after']and not d['holdout_used'],'Actual adaptive16 development reception')
        receive_audit=verify_outputs(adaptive_receive,adaptive_receive.parent,'/home/liulu/projects/VAR_COMM/outputs/PAPER-SUPPLEMENT-20261008/a5_adaptive_fixed16_v1');audits.append(receive_audit)
        require(not receive_audit['mismatches'],'Adaptive16 downloaded receive output mismatch')
        final_state['adaptive_development_receive']=True
        update('A0-158','MISSING',adaptive_receive,'Actual16fixed-development frames,13dB,seed2001,frozen profile3009;32paid;two actual wait-zero;original ledger unchanged',
            'Adaptive cached metric scoring and final six-column export','Reuse actual16 outputs;finish only downstream score/export','PENDING_RUNNING',first_batch_hard_gap='true')
    adaptive_figure=ROOT/'paper/figures/a5_adaptive_six_methods_N1024_13dB_group02'
    if (adaptive_figure/'completion.json').exists():
        d=read(adaptive_figure/'completion.json');md=read(adaptive_figure/'metrics_completion.json')
        waited=read(adaptive_figure/'actual_score_plot_wait.json');visual=read(adaptive_figure/'visual_validation.json')
        require(d['status']==md['status']=='COMPLETE'and d['source_count_scored']==16 and d['display_sources']==[4,21,24,29]
            and d['N']==1024 and d['snr_db']==13 and d['missing_metric_cells']==0 and not d['old_native_figure_modified']
            and not d['old_five_columns_modified'],'Actual adaptive six-column development export')
        require(waited['actual_children_waited']and waited['score_exit_code']==waited['plot_exit_code']==0
            and visual['first_five_columns_pixel_identical']and visual['actual_png_opened']and visual['no_clipped_headers_or_panels'],
            'Adaptive figure actual wait-zero and actual visual/pixel comparison')
        for filename in ['completion.json','metrics_completion.json']:
            checked=verify_outputs(adaptive_figure/filename,adaptive_figure,'/home/liulu/projects/VAR_COMM/paper/figures/a5_adaptive_six_methods_N1024_13dB_group02');audits.append(checked)
            require(not checked['mismatches']and checked['remote_only_not_downloaded']==0,'Complete local adaptive figure/metric outputs')
        values=csv_read(adaptive_figure/'per_image_metrics.csv')
        require(len(values)==96 and len({(r['source_index'],r['method'],r['metric'])for r in values})==96
            and all(math.isfinite(metric_number(r))for r in values if r['method']!='source'),'Complete adaptive figure80 metric values')
        final_state['adaptive_development_export']=True
        update('A0-158','COMPLETE',adaptive_figure/'completion.json',
            'Actual16development outputs at13dB;32PHY;fixed4-source six-column figure and80finite displayed metrics;native256 figure retained',
            first_batch_hard_gap='false',notes='Independent extra-development receipt;no holdout reselection;original five columns/64metric cells unchanged.')
        check('Adaptive fixed-development missing-image completion','PASS',[adaptive_receive,adaptive_figure/'completion.json',adaptive_figure/'metrics_completion.json',adaptive_figure/'actual_score_plot_wait.json',adaptive_figure/'visual_validation.json'],
            dict(receive_sources=16,display_sources=[4,21,24,29],SNR=13,new_PHY=32,profile_id=3009,
                new_quality_calls=md['new_quality_calls_total'],old_metrics_recomputed=0,new_bootstrap=0),
            'Prespecified development demonstration;not a holdout sample or evidence of population success rate.')
    gaps=[dict(item_id=r['item_id'],task=r['task'],method=r['method'],status=r['status'],execution_state=r['execution_state'],
        missing=r['missing_scope'],minimum_next_entry=r['minimum_next_entry'])for r in rows if r['first_batch_hard_gap']=='true']
    require(sha(old)==old_sha and sha(OUT/'README.md')==old_readme_sha,'Historical snapshots must remain unchanged')
    with (OUT/'coverage.csv').open('w',encoding='utf8',newline='')as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    payload=dict(status='CURRENT_LOCAL_COVERAGE_AUDIT_COMPLETE',snapshot_at=now,published_commit='252176e041758ecb2d3e81b6fde5b587e7e17bb7',
        first_round_status='COMPLETE'if not gaps else'PENDING_REQUIRED_ITEMS',
        prior_snapshots_preserved=dict(reuse_matrix_sha256=old_sha,README_sha256=old_readme_sha),row_count=len(rows),statuses=dict(Counter(r['status']for r in rows)),
        allowed_matrix_statuses=['COMPLETE','REUSE','MISSING','UNSUPPORTED','NOT_RUN'],execution_state_note='PENDING_RUNNING is separate from the five allowed task statuses;missing completion never means not running.',
        checks=checks,receipt_audits=audits,hard_remaining_items=gaps,
        actual_final_stage_states=final_state,last_remote_observation=dict(provenance='parent coordinator messages plus downloaded actual receipts;this agent did not connect remotely',
            process_liveness_not_claimed=True,completion_claimed=not gaps),new_training=0,new_inference=0,new_channel=0,new_bootstrap=0,SSH_or_remote_dispatch=False)
    (OUT/'implementation_checks.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    text=['# A0 current coverage and implementation checks','',f'Snapshot: {now}. Historical `reuse_matrix.csv` and `README.md` remain byte-identical. Current files are `coverage.csv` and `implementation_checks.json`.',
        '',f'Matrix rows: {len(rows)}. Statuses: {dict(Counter(r["status"]for r in rows))}. The five document statuses are retained. Actual receiver, metric and figure stages are checked separately; a launch or prepared script is never completion.',
        '',f'First-round remaining required items: {len(gaps)}. Final actual-stage states: {final_state}.',
        '', 'Verified completed deliveries:',
        '', '- Native256 BPG:72 new source-paired contrasts against the published three methods;one shared10000x500 resampling array;all original means/CIs unchanged.',
        '- A1:3076 real source-codec probe cases;20 received-only source roundtrips;44qualification+3072prescreen+10800calibration+18000holdout=31916/34604 paid decodes. All9000 adaptive holdout frames closed with8 actual child exits0 and unchanged original ledger. Resources/failures include3000 source settings and6 complete BPG stream examples. Four-core metrics give24 new mean/CI rows,96 new paired contrasts and12000 source means;2635 new unique image scores,379 inherited native-image reuses and500 reference preparations. New figures contain10 vectorPDF,10editableSVG and10PNG600dpi files;both combined PNGs actually opened and passed visual review.',
        '- A2:20native/wrapper checks and all44 outputs pass;RGB/MSE/PSNR independently checked. FixedC6 adaptation reused;C7untrained/C13overbudget boundaries explicit.',
        '- A3:five methods,960source attempts,1080actualPHY decodes,1038verified output hashes;timing_single_output.csv and model storage tables delivered. NativeBPG source-unfit attempts have RX/E2E NA, not zero.',
        '- A4:scientific tables/published means identical;44-file closed delivery manifest and source-bound1dB/19dB diagnosis pass. Original console.log receipt is preserved;the sidecar closes only its final-log packaging discrepancy.',
        '- A5:10/19dB mechanisms,original126reference rows,and native256 six-column13dB figure complete. Additional adaptive development16source/13dB run has32paidPHY,16decoded images,two actual exits0;separate six-column figure retains the original five columns and native256 figure. New quality calls15 plus1 identical-native-image reuse;the80 displayed reconstruction-metric values have no missing cells. Root visual review and first-five pixel identity are recorded.',
        '- A6:first-round paper/code comparison and reproducibility scope complete. ARPC and Ada native inference remain NOT_RUN, not manufactured performance results.',
        '', 'Scope limits retained: HiFi has no common500 overlap and remains absent from the common500 benchmark;its valid13dB development images are used only as diagnostic examples. Swin19dB remains outside training/calibration,its fixed256-use header is not proven optimal,and no unsupported body channels are substituted. A6 native/cost and B/C expansion are conditional and were not launched. These limitations are not hidden by the first-round completion status.',
        '', 'This read-only audit invokes noSSH,GPU,model,channel,bootstrap or scientific-result changes. For large A1/source-feature/receiver caches not downloaded, only local files count as locally hash verified and remote-only counts remain explicit. Original results,old confidence intervals and the historical A0 snapshots are unchanged. See implementation_checks.json for receipt/output counts and exact evidence hashes.']
    if gaps:text+=['','Still required:']+['- '+g['item_id']+': '+g['missing']for g in gaps]
    (OUT/'CURRENT_README.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps(dict(rows=len(rows),checks=len(checks),gaps=len(gaps),statuses=payload['statuses'],old_snapshot_unchanged=True)))

if __name__=='__main__':build()
