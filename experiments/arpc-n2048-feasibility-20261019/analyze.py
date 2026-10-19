"""Existing-data descriptive analysis and bit accounting; no model/PHY calls."""
from pathlib import Path
from fractions import Fraction
import argparse,csv,hashlib,json,math,statistics,shutil

FIELDS=[('version',4),('codec',2),('endpoint',4),('fragment_index',1),
        ('fragment_count_minus_one',1),('pair_id',32),('original_bit_length',14),
        ('reserved_zero',6),('whole_object_crc32',32)]
assert sum(n for _,n in FIELDS)==96
def loadcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def savecsv(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def dump(p,x):p.write_bytes((json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8'))
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def analyze(inp,out):
    out.mkdir(exist_ok=False)
    audit=json.loads((inp/'asset_audit.json').read_text(encoding='utf8'))
    lens=loadcsv(inp/'arpc_cal100_lengths.csv'); q=loadcsv(inp/'arpc_cal100_quality.csv')
    idx={(x['source_id'],int(x['endpoint'])):x for x in q if x['kind']=='clean_prefix'}
    assert len(lens)==len(idx)==700 and len({x['source_id'] for x in lens})==100
    assert len({x['metric_identity'] for x in q})==1
    calids={x['source_id'] for x in audit['assets'] if x['cohort']=='cal100'}
    testids={x['source_id'] for x in audit['assets'] if x['cohort']=='external_common100'}
    assert not calids & testids and {x['source_id'] for x in lens}==calids
    old_m10={x['source_id']:int(x['arithmetic_bits']) for x in loadcsv(inp/'nest_cal1000_m10_lengths.csv')}
    e7={x['source_id']:int(x['payload_bits']) for x in lens if x['endpoint']=='7'}
    for x in audit['assets']:
        if x['cohort']=='cal100':
            assert x['payload_bits']==(old_m10 if x['method']=='NeST-Com' else e7)[x['source_id']]
    metrics=['psnr_db','lpips_alex','dinov2_vitl14_cosine']; joined=[]
    for x in lens:
        j=idx[x['source_id'],int(x['endpoint'])]
        assert x['source_index']==j['source_index'] and x['rank']==j['rank']
        joined.append(dict(rank=int(x['rank']),source_id=x['source_id'],source_index=int(x['source_index']),endpoint=int(x['endpoint']),payload_bits=int(x['payload_bits']),**{m:float(j[m]) for m in metrics}))
    savecsv(out/'arpc_cal100_payload_quality.csv',joined)
    summary=[]
    for e in range(1,8):
        r=[x for x in joined if x['endpoint']==e]; b=[x['payload_bits'] for x in r]
        summary.append(dict(endpoint=e,sources=len(r),payload_min_bits=min(b),payload_mean_bits=statistics.mean(b),payload_median_bits=statistics.median(b),payload_max_bits=max(b),**{m+'_mean':statistics.mean(x[m] for x in r) for m in metrics}))
    savecsv(out/'arpc_endpoint_summary.csv',summary)
    layouts=json.loads((inp/'existing_arpc_catalogue.json').read_text())['profiles']
    unique={}
    for p in layouts:
        k=(p['q'],p['nominal_rate']); row=p['layout']
        assert row['actual_constructor_used'] and row['n']==956*p['q']
        if k in unique:assert unique[k]['layout_id']==row['layout_id']
        unique[k]=row
    assert len(unique)==12
    fit=[]; fullrows=[]
    cal=[x for x in audit['assets'] if x['cohort']=='cal100']
    for (qm,rate),layout in sorted(unique.items(),key=lambda x:(x[0][0],Fraction(x[0][1]))):
        k=layout['k']; n=layout['n']; fragbytes=(k-13-16-96)//8; cap=16*fragbytes
        assert k==n*Fraction(rate).numerator//Fraction(rate).denominator
        row=dict(q=qm,nominal_rate=rate,k_per_subframe=k,n_per_subframe=n,
          outer_length_bits_per_subframe=13,outer_crc_bits_per_subframe=16,
          reassembly_bits_per_subframe=96,source_bytes_per_subframe=fragbytes,
          source_capacity_pair_bits=cap,pair_header_symbols=136,pair_body_symbols=1912,
          existing_layout_id=layout['layout_id'])
        for method,label in [('ARPC','arpc_e7'),('NeST-Com','nest_m10')]:
            aa=[x for x in cal if x['method']==method]
            assert len(aa)==100
            row[label+'_fit_count']=sum(math.ceil(x['payload_bits']/8)<=2*fragbytes for x in aa)
            row[label+'_fit_rate']=row[label+'_fit_count']/100
        fit.append(row)
    savecsv(out/'cal100_fit_by_legal_mcs.csv',fit)
    for x in audit['assets']:
        # Full test lengths are inventoried after the cal-only shared-MCS proposal.
        bits=x['payload_bits']; nbytes=math.ceil(bits/8);a=(nbytes+1)//2;b=nbytes//2
        fullrows.append(dict(cohort=x['cohort'],method=x['method'],source_id=x['source_id'],endpoint=x['endpoint'],source_bits=bits,source_alignment_bits=nbytes*8-bits,fragment0_bytes=a,fragment1_bytes=b,
          fragment0_information_bits=125+8*a,fragment1_information_bits=125+8*b,
          fits_shared_q6_r5_6=125+8*max(a,b)<=4780,
          total_information_padding_bits=9560-250-8*nbytes,
          full_image_candidate_exists=x['full_image'] is not None))
    savecsv(out/'full_representation_fit_and_reuse.csv',fullrows)
    assert all(x['fits_shared_q6_r5_6'] for x in fullrows)
    groups=[]
    for cohort in ('cal100','external_common100'):
        for method in ('ARPC','NeST-Com'):
            r=[x for x in fullrows if x['cohort']==cohort and x['method']==method]
            groups.append(dict(cohort=cohort,method=method,sources=len(r),min_bits=min(x['source_bits'] for x in r),mean_bits=statistics.mean(x['source_bits'] for x in r),max_bits=max(x['source_bits'] for x in r),fit_count=sum(x['fits_shared_q6_r5_6'] for x in r),existing_full_streams=100,new_TX_passes=0,full_image_candidates=sum(x['full_image_candidate_exists'] for x in r),full_image_gaps_in_audited_scope=sum(not x['full_image_candidate_exists'] for x in r)))
    savecsv(out/'asset_and_compute_summary.csv',groups)
    hist=loadcsv(inp/'historical_reliability_cells.csv')
    savecsv(out/'historical_q6_r5_6_diagnostic.csv',[x for x in hist if x['q']=='6' and x['nominal_rate']=='5/6'])
    plan=dict(schema='ARPC_N2048_PLANNING_ONLY_V1',basis_HEAD=audit['basis_HEAD'],execution_authorized=False,new_scientific_calls_this_delivery=0,
      unchanged_N1024=True,primary_question='Complete-representation quality and delivery under equal total channel resources; not best per-method MCS performance.',
      source_cohorts=dict(calibration='existing ARPC cal100 for both methods, same source archive SHA',validation='existing external_common100_v1 for both methods; already-used subset, not a new blind test'),
      methods={'NeST-Com':'entropy-coded m10 K=0, all 680 tokens','ARPC':'native256 e7, all 7 scales, paid caption and native finished source object'},
      channel=dict(N_total=2048,subframes=2,N_per_subframe=1024,header_symbols_per_subframe=68,body_symbols_per_subframe=956,nominal_Es=2,nominal_total_energy=4096,frame_normalization=False,power_allocation=False,feedback=False,fading=False),
      proposed_primary_MCS=dict(q=6,rate='5/6',k=4780,n=5736,same_both_methods=True,same_both_fragments=True,selection='Only shared existing MCS fitting all 100 full calibration objects in both methods; no test quality used.'),
      reassembly_fields=[dict(field=f,bits=n) for f,n in FIELDS],
      overhead=dict(two_length_fields_bits=26,two_outer_crc_bits=32,two_reassembly_records_bits=192,total_information_overhead_bits=250,source_padding_bits='0..7 per whole NeST object; ARPC already byte aligned',additional_arithmetic_finish_bits=0,source_finish_already_in_measured_payload=True,max_complete_object_bits=9296),
      receive_rule='Both paid headers, fragment CRC16s, consistent paid reassembly fields, object CRC32 and canonical source parser must pass; otherwise fixed gray for the entire pair. Never use TX truth to decide success.',
      future_validation=dict(sources=100,methods=2,snr_db=[4,10,19],noise_seeds=[6201,6202,6203],logical_method_frames=1800,physical_N1024_subframes=3600,maximum_header_and_body_decode_attempts=7200,old_physical_frames_reused=0),
      future_qualification_cap=dict(paired_test_cases=24,physical_subframes=48,maximum_header_and_body_decode_attempts=96,independent_full_source_RX_calls=12),
      future_total_cap=dict(logical_pairs=1824,physical_subframes=3648,header_and_body_decode_attempts=7296,source_RX_and_render_upper_bound=1812),
      expected_clean_input_reuse=dict(validation_existing_complete_streams=200,new_TX_probability_passes=0,validation_known_correct_full_images_available=1,validation_full_image_gaps_in_audited_registries=199,condition='Reuse only after received-object equality and matching model/numerical/runtime identity; these are cache candidates, not granted future successes.'),
      scoring_plan='Reuse final seven-metric versions by exact pixel pair and identity only. Future worst-case budget: <=1800 new validation image pairs + <=100 gray pairs = <=13300 scalar scores; new paired bootstrap only once if future execution is authorized. No metric/scientific execution now.',
      no_calibration_channel_sweep=True,no_extra_methods=True,no_training=True,no_automatic_next_stage=True)
    dump(out/'proposed_scope.json',plan)
    dump(out/'input_manifest.json',dict(inputs=[pin(p) for p in sorted(inp.iterdir())],remote_evidence=audit['input_pins'],skill_paths=['C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md','C:/Users/11946/.codex/skills/scientific-figure-making/references/design-theory.md','C:/Users/11946/.codex/skills/scientific-figure-making/references/common-patterns.md','C:/Users/11946/Documents/ChatGPT/comm/.research/figures_redesign_v1/references/figure_ophthal_review__plot_trend.py']))
    print(json.dumps(dict(summary=summary,fit=fit,assets=groups),indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--inputs',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();analyze(v.inputs,v.output)
