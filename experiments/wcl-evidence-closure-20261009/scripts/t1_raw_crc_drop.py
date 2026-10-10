"""Same-actual-RX raw CRC rejection diagnostic, with no truth-based decisions.

Only body CRC rejection requests a constant0.5 reconstruction; otherwise original
metrics and image identity are retained exactly. This helper does not run PHY,
render a VAR image, or score the requested gray images.
"""
import gzip
import hashlib
import json
from pathlib import Path
from t1_entropy_core import read,require,sha,verify,write
from t1_holdout_metadata import SNRS,SEEDS,RAW_REFS,RAW_DROP

METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def receive_evidence(fact):
    events=fact['packet_events']
    require(fact['status']=='MAIN_RAW64_KEEP_RECEPTION_COMPLETE' and fact['rule']=='KEEP'
        and fact['logical_packet_calls']==1+int(fact['header_ok'])
        and fact['packet_event_ids']==[e['event_id'] for e in events],'Original actual paid receiver facts required')
    frame={k:fact[k] for k in ('source_index','source_id','snr_db','noise_seed','candidate_id','holdout_slot',
        'public_frame_counter','received_sha256','codebook_sha256','aliases_sha256')}
    return dict(paid_receive_sha256=digest(dict(domain='COMMON500_ORIGINAL_PAID_EVENTS_V1',packet_events=events)),
        actual_state_sha256=fact['actual_state_sha256'],
        frame_identity_sha256=digest(dict(domain='COMMON500_ORIGINAL_RECEIVE_FRAME_V1',frame=frame)))

def read_output(done,name):
    pairs=[(p,h) for p,h in done['outputs'].items() if Path(p).name==name]
    require(len(pairs)==1,'Ambiguous original output '+name);path,digest_=pairs[0];verify(path,digest_)
    with gzip.open(path,'rt',encoding='utf-8') as f:value=json.load(f)
    return value,{path:digest_}

def prepare(original_statistics_completion,out):
    out=Path(out);require(not out.exists(),'Fresh diagnostic metadata output required')
    done=read(original_statistics_completion)
    require(done['scientific_statistics_completed'] and done['source_count']==500,'Actual original common500 statistics required')
    facts,b1=read_output(done,'raw_receive_facts.json.gz');normalized,b2=read_output(done,'normalized_rows.json.gz')
    ids=done['source_ids'];bykey={};gray_candidates={}
    for fact in facts:
        if fact['snr_db'] not in SNRS:continue
        for branch in fact['families']:
            if branch not in ('WHOLE','PARTIAL'):continue
            key=(branch,fact['snr_db'],fact['source_index'],fact['noise_seed'])
            require(key not in bykey,'Duplicate same-source raw receive mapping');bykey[key]=fact
    expected={(b,s,i,n) for b in ('WHOLE','PARTIAL') for s in SNRS for i in range(500) for n in SEEDS}
    require(set(bykey)==expected,'Original raw receive grid is incomplete')
    rows=[];needs=set();counts={b:0 for b in ('WHOLE','PARTIAL')}
    for old in normalized:
        branch=old['branch'];snr=old['snr_db']
        if branch not in ('WHOLE','PARTIAL') or old['arm']!='VAR_COMPLETION' or snr not in SNRS:continue
        key=(branch,snr,old['source_index'],old['noise_seed']);fact=bykey[key];i=old['source_index']
        require(old['source_id']==fact['source_id']==ids[i] and old['receive_evidence']==receive_evidence(fact),
            'Original metric and actual receiver evidence differ')
        require(fact['header_ok'] is fact['header']['header_ok'] and fact['body_crc_accept'] in (None,True,False),
            'Actual header/CRC state differs')
        if fact['header_ok']:
            require(fact['body'] is not None and fact['body']['crc_accept'] is fact['body_crc_accept'],
                'Actual decoded body CRC disagrees')
        else:require(fact['gray'] and fact['receiver_state']['kind']=='gray','Header rejection must retain original gray output')
        image=fact['same_RX_images']['VAR_completion']
        require(image['image_sha256']==old['original_image_sha256'],'Original reconstruction identity differs')
        # No transmitted token, source pixel, or true class enters this decision.
        dropped=fact['body_crc_accept'] is False
        if dropped:needs.add(i);counts[branch]+=1
        if fact['gray']:
            gray_candidates.setdefault(i,[]).append(dict(image=image,metrics={m:old['metrics'][m] for m in METRICS},
                metric_evaluator_identity=old['original_metric_evaluator_identity'],metric_identity={m:old['metric_identity'][m] for m in METRICS},
                receive_evidence=old['receive_evidence'],verification_required='Open sealed same-RX float image and require exact all0.5 before reuse'))
        prefix=RAW_DROP[('WHOLE','PARTIAL').index(branch)]
        rows.append(dict(family=prefix,point_id=prefix+'_SNR_'+str(snr),snr_db=snr,source_index=i,source_id=ids[i],noise_seed=old['noise_seed'],
            original_point_id=old['point_id'],original_header_ok=fact['header_ok'],original_body_crc_accept=fact['body_crc_accept'],
            same_actual_receive_evidence=old['receive_evidence'],gray_requested=dropped,
            output_rule='CONSTANT_0P5_RGB' if dropped else 'ORIGINAL_KEEP_OUTPUT_UNCHANGED',
            reused_metrics=None if dropped else {m:old['metrics'][m] for m in METRICS},
            metric_evaluator_identity=old['original_metric_evaluator_identity'],metric_identity={m:old['metric_identity'][m] for m in METRICS},
            original_image=image,TX_truth_used_for_decision=False,new_packet_decodes=0,new_VAR_renders=0))
    require(len(rows)==9000 and len({(x['family'],x['snr_db'],x['source_index'],x['noise_seed']) for x in rows})==9000,
        'Complete9000 diagnostic frames required')
    rows.sort(key=lambda x:(x['source_index'],x['snr_db'],x['noise_seed'],x['family']))
    out.mkdir(parents=True);write(out/'frame_plan.json',rows)
    write(out/'gray_requirements.json',dict(source_indices=sorted(needs),source_count=len(needs),
        old_gray_candidates={str(i):gray_candidates.get(i,[]) for i in sorted(needs)},
        new_gray_score_upper_bound=len(needs),constant_rgb_value=.5,new_VAR_renders=0))
    result=dict(status='T1_RAW_CRC_DROP_METADATA_COMPLETE_NOT_SCORED',diagnostic_frame_count=9000,
        body_CRC_rejected_frames=counts,gray_sources=len(needs),new_VAR_renders=0,new_packet_decodes=0,
        new_metric_calls=0,old_KEEP_outputs_modified=False,selection_by_TX_truth=False,
        input_bindings={str(Path(original_statistics_completion).resolve()):sha(original_statistics_completion),**b1,**b2},
        outputs={str(out/name):sha(out/name) for name in ('frame_plan.json','gray_requirements.json')})
    write(out/'completion.json',result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--original-statistics-completion',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(prepare(a.original_statistics_completion,a.out)['status'])
