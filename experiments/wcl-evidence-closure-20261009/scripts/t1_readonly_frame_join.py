"""Read-only join of completed quality, actual reception and raw-DROP evidence.

Truth is used only for retrospective token-error counts. No receiver decision,
image, scientific metric or confidence interval is recomputed.
"""
import csv,gzip,json
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,read,require,sha,verify
from t1_holdout_metadata import FAMILIES,RAW_DROP,SNRS,SEEDS,pinned
from t1_holdout_statistics import METRICS,bound_output
from t1_raw_crc_drop import receive_evidence
from t1_source_asset_schema import token_sha

def csv_read(path):
    with Path(path).open(newline='',encoding='utf-8-sig')as f:return list(csv.DictReader(f))

def key(row):return row['family'],int(row['snr_db']),int(row['source_index']),int(row['noise_seed'])

def raw_diagnostic_row(item,fact,tokens,score):
    require(item['source_id']==fact['source_id']==score['source_id']and item['same_actual_receive_evidence']==receive_evidence(fact),
        'Same-source raw receiver and frozen diagnostic proof required')
    dropped=fact['body_crc_accept']is False
    require(item['gray_requested']is dropped and item['TX_truth_used_for_decision']is False,'Actual body CRC alone controls diagnostic DROP')
    state=fact['receiver_state'];errors=None;count=0
    if state['kind']!='gray':
        require(state['order']=='raster','Offline raw diagnostic requires registered raster token order')
        received=np.asarray([v for scale in state['prefix']for v in scale]+list(state['partial_values']),np.int64)
        count=OFFSETS[state['m']]+state['K'];require(len(received)==count,'Actual raw token count differs')
        errors=int(np.count_nonzero(received!=tokens[:count]))
    header=bool(fact['header_ok']);crc=fact['body_crc_accept'];gray=bool(dropped or fact['gray'])
    if not header:category='HEADER_REJECT_ORIGINAL_GRAY'
    elif dropped:category='BODY_CRC_REJECT_DIAGNOSTIC_GRAY'
    elif errors:category='CRC_ACCEPTED_WITH_TOKEN_ERRORS_KEEP_UNCHANGED'
    else:category='CRC_ACCEPTED_NO_TOKEN_ERRORS_KEEP_UNCHANGED'
    tx=fact['transmission'];groups=tx['groups'];require(len(groups)==1,'Original single-body raw contract required')
    g=groups[0]
    # Exact transmitter fields differ from entropy names; aliases are explicit.
    get=lambda *names:next((g[n]for n in names if n in g),None)
    k=get('information_bits','k');n=get('transmitted_bits','n')
    q={'QPSK':2,'16QAM':4,'64QAM':6}.get(fact['modulation'])
    row=dict(score,N=fact['N'],method=item['family'],failure_state=category,
        original_header_ok=header,original_body_crc_accept=crc,header_ok=header,body_crc_accept=crc,
        gray=gray,gray_requested=dropped,original_KEEP_gray=bool(fact['gray']),output_rule=item['output_rule'],
        original_KEEP_token_error_count=errors,original_KEEP_token_error_denominator=count,
        CRC_undetected_token_error=bool(header and crc is True and errors),diagnostic_truth_used_for_output=False,
        target_m=fact['m'],actual_m=fact['m'],K=fact['K'],transmitted_token_count=OFFSETS[fact['m']]+fact['K'],q=q,
        actual_k=k,actual_n=n,actual_code_rate=k/n if k is not None and n else None,
        header_symbols=tx['header_uses'],body_symbols=tx['body_uses'],frame_padding_symbols=tx['idle_uses'],
        body_source_bits=12*(OFFSETS[fact['m']]+fact['K']),
        known_information_padding_bits=k-g['source_bits']-g['crc_bits']-g.get('tail_bits',0)if k is not None else None,
        E_frame=tx['E'],rho=tx['E']/(2*tx['N']),physical_status=fact['source_status'],
        source_status=category,original_image_sha256=item['original_image']['image_sha256'],
        output_image_identity='CONSTANT_RGB_0P5'if dropped else item['original_image']['image_sha256'],
        same_actual_receive_evidence=json.dumps(item['same_actual_receive_evidence'],sort_keys=True),population='holdout')
    require(all(score[m] is not None for m in METRICS),'Complete scored diagnostic required')
    return row

def joined_rows(score_completion,score_csv=None):
    done=read(score_completion)if not isinstance(score_completion,dict)else score_completion
    require(done['status']=='T1_POSTHOC_COMMON500_FOUR_METRICS_COMPLETE'and done['source_count']==500,'Completed four metrics required')
    actual=bound_output(done,'per_frame.csv')
    if score_csv is not None:require(sha(score_csv)==sha(actual),'Supplied score rows differ from actual completed scores')
    scores=csv_read(actual);by={key(x):x for x in scores}
    expected={(f,s,i,n)for f in FAMILIES+RAW_DROP for s in SNRS for i in range(500)for n in SEEDS}
    require(len(scores)==len(by)==18000 and set(by)==expected,'Complete unique500x3 source/noise grid')
    request=pinned(dict(path=done['request_path'],sha256=done['request_sha256']));render=pinned(request['render_completion'])
    require(render['source_ids']==done['source_ids']and render['freeze']==done['freeze'],'Same scored render population and freeze')
    frames=csv_read(bound_output(render,'per_frame.csv'));result=[]
    for frame in frames:
        score=by[key(frame)];require(frame['source_id']==score['source_id'],'Rendered source identity mismatch')
        result.append(dict(frame,**{m:score[m]for m in METRICS}))
    require(len(result)==9000,'All entropy frames required')
    diagnostic=pinned(request['raw_diagnostic_completion']);plan=read(bound_output(diagnostic,'frame_plan.json'))
    original=pinned(request['original_statistics_completion']);source=pinned(request['source_request']);mf=pinned(source['original_source_manifest'])
    require(original['source_ids']==mf['source_ids']==done['source_ids'],'Original raw and entropy source identities differ')
    with gzip.open(bound_output(original,'raw_receive_facts.json.gz'),'rt',encoding='utf-8')as z:facts=json.load(z)
    byfact={}
    for fact in facts:
        if fact['snr_db']not in SNRS:continue
        for branch in fact['families']:
            if branch not in ('WHOLE','PARTIAL'):continue
            family=RAW_DROP[('WHOLE','PARTIAL').index(branch)];k=(family,fact['snr_db'],fact['source_index'],fact['noise_seed'])
            require(k not in byfact,'Duplicate actual raw receive facts');byfact[k]=fact
    plans={i:[]for i in range(500)}
    for item in plan:plans[item['source_index']].append(item)
    for entry in mf['records']:
        i=entry['source_index'];verify(entry['archive'],entry['archive_sha256'])
        with np.load(entry['archive'],allow_pickle=False)as z:tokens=z['tokens'].copy()
        require(token_sha(tokens)==entry['tokens_sha256'],'Original source tokens changed')
        for item in plans[i]:result.append(raw_diagnostic_row(item,byfact[key(item)],tokens,by[key(item)]))
    require(len(result)==18000 and {key(x)for x in result}==expected,'Final read-only join omitted or duplicated frames')
    return result
