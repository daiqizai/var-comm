"""Explicit owned m10 extension of the completed original T1 calibration assets.

Only new ten-scale TX/RX work is performed. Completed first100 can be reused
verbatim when extending to1000. No pixels, metrics, PHY or policy search here.
"""
import argparse,json,os,signal,time,traceback
from fractions import Fraction
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,read,require,sha,verify,choose_arithmetic
from t1_holdout_metadata import pin,pinned
from t1_source_asset_schema import token_sha,payload_sha
from t1_holdout_statistics import bound_output
from t1_holdout_render import build_native
from t6_source_codec import SourceCodec10,build_record
import t2_pilot as shared
save=shared.save

def candidates2048():
    return [dict(target_m=m,q=q,nominal_rate=str(rate),source_capacity=min((1980*q*Fraction(rate).numerator)//Fraction(rate).denominator-29,8191))
        for m in (7,8,9,10) for q in (2,4,6) for rate in ('1/2','2/3','3/4','5/6')]

def prepare(a):
    family=pinned(pin(a.family_freeze));policy=pinned(family['original_policy_freeze']);done=read(a.source_completion)
    require(family['status']=='T1_SINGLE_ENTROPY_FAMILY_FROZEN_CALIBRATION_ONLY' and family['family']=='EC_VAR_WHOLE'
        and not family['holdout_used_for_selection'],'Registered single VAR entropy family required')
    require(done['status']=='T1_CALIBRATION_COMPLETE_SOURCE_ASSETS_SEALED' and done['source_count']==1000,'Completed original T1cal1000 required')
    mp=bound_output(done,'manifest.json');manifest=read(mp)
    require(manifest['source_ids']==policy['calibration_source_ids'] and len(manifest['records'])==1000,'Exact original calibration order required')
    out=Path(a.out).resolve();root=Path(a.root).resolve();env=read(a.environment_request)
    require(env['root']==str(root) and out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009') and not out.exists(),'Fresh owned WCL output required')
    prior=None
    if a.prior_completion:
        p=read(a.prior_completion)
        require(p['status']=='T6_CALIBRATION_ENTROPY_SOURCE_ASSETS_COMPLETE' and p['family']==family['family']
            and p['entropy_family_selection']==pin(a.family_freeze) and p['source_count']<a.count
            and p['original_calibration_source_ids']==manifest['source_ids'],'Exact smaller completed source stage required')
        prior=pin(a.prior_completion)
    out.mkdir(parents=True)
    r=dict(schema='T6_CALIBRATION_M10_SOURCE_REQUEST_V1',root=str(root),out=str(out),family=family['family'],
        entropy_family_selection=pin(a.family_freeze),source_completion=pin(a.source_completion),source_manifest=pin(mp),
        original_calibration_source_ids=manifest['source_ids'],records=manifest['records'][:a.count],source_count=a.count,
        static_completion=policy['protocol']['static_completion'],environment_request=pin(a.environment_request),prior_completion=prior,
        scope='first_count_of_original_calibration1000',holdout_used=False,training_updates=0,
        VAR_TX_cap=a.count,VAR_RX_cap=a.count,deadline_unix=a.deadline_unix,max_seconds=86400,
        stop_files=[str(root/'STOP'),str(out/'STOP')],new_packet_decodes=0,new_metric_calls=0,new_image_renders=0)
    names=['t6_calibration_sources.py','t6_source_codec.py','t1_codec_runtime.py','t1_source_asset_schema.py','t1_entropy_core.py',
        't1_holdout_render.py','t1_holdout_metadata.py','t1_holdout_statistics.py','t1_calibrate.py','t2_pilot.py']
    r['source_bindings']={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in names}
    save(out/'request.json',r);return dict(status='T6_CALIBRATION_M10_REGISTERED',count=a.count,prior_reuse=bool(prior))

def run(path):
    r=read(path);rh=sha(path);out=Path(r['out']);require(r['schema']=='T6_CALIBRATION_M10_SOURCE_REQUEST_V1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit T6source GPU0 owner required')
    for p,h in r['source_bindings'].items():verify(p,h)
    require(not(out/'owner_started.json').exists() and not(out/'completion.json').exists(),'No repeated source owner attempt')
    env=pinned(r['environment_request']);base=pinned(r['source_completion']);pinned(r['source_manifest']);pinned(r['entropy_family_selection']);pinned(r['static_completion'])
    prior={};prior_done=None
    if r['prior_completion']:
        prior_done=pinned(r['prior_completion']);prior_manifest=pinned(prior_done['source_manifest'])
        prior={e['source_index']:e for e in prior_manifest['records']}
    started=time.monotonic();native=None;counts={'TX':0,'RX':0};records=[];outputs={};qualifications=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,shared.stop)
    def guard():
        shared.guard(r,started)
        if native is not None:native.health_polling.next_check=0.;native.common.check()
    def call(kind,index,fn):
        guard();require(counts[kind]<r['VAR_'+kind+'_cap'],'Source call cap exhausted')
        p=out/'calls'/f'{index:04d}_{kind}.reserved.json';require(not p.exists(),'Unresolved source probability call, no automatic repeat')
        save(p,dict(request_sha256=rh,source_index=index,kind=kind,m=10));value=fn();counts[kind]+=1
        save(p.with_name(f'{index:04d}_{kind}.complete.json'),dict(request_sha256=rh,source_index=index,kind=kind,m=10));return value
    try:
        with shared.lock(env['visual_config']['visual_lock']),shared.lock(out/'gpu.lock'):
            save(out/'owner_started.json',dict(request_sha256=rh,pid=os.getpid(),started_unix=time.time()))
            native,_=build_native(env,out);codec=SourceCodec10(r['root'],static_completion=r['static_completion']['path'],native=native)
            for entry in r['records']:
                guard();i=entry['source_index'];verify(entry['checkpoint'],entry['checkpoint_sha256']);old=read(entry['checkpoint'])
                require(base['outputs'][entry['checkpoint']]==entry['checkpoint_sha256'] and old['source_role']=='calibration'
                    and old['source_id']==r['original_calibration_source_ids'][i] and old['source_index']==i,'Original calibration checkpoint differs')
                if i in prior:
                    e=prior[i];verify(e['checkpoint'],e['checkpoint_sha256']);cp=read(e['checkpoint'])
                    require(cp['original_source_record']==entry and cp['m10_independent_roundtrip'] and cp['entropy_family_selection']==r['entropy_family_selection'],
                        'Previously extended source does not match original source/family')
                    verify(cp['archive']['path'],cp['archive']['sha256']);require(prior_done['outputs'][e['checkpoint']]==e['checkpoint_sha256']
                        and prior_done['outputs'][cp['archive']['path']]==cp['archive']['sha256'],'Prior source outputs are not sealed')
                    records.append(e);outputs[e['checkpoint']]=e['checkpoint_sha256'];outputs[cp['archive']['path']]=cp['archive']['sha256'];continue
                verify(old['archive']['path'],old['archive']['sha256'])
                require(base['outputs'][old['archive']['path']]==old['archive']['sha256'],'Original calibration array unsealed')
                with np.load(old['archive']['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files if k=='tokens' or k.startswith('var_')}
                tokens=arrays['tokens'];require(token_sha(tokens)==old['source_tokens_sha256'],'Original680 token hash differs')
                existing={int(m):arrays[e['bits_key']] for m,e in old['streams'][r['family']].items()}
                require(set(range(6,10)).issubset(existing),'Original m6..9 streams required')
                for m,bits in existing.items():
                    e=old['streams'][r['family']][str(m)];require(payload_sha(bits)==e['payload_sha256'],'Original source bits changed')
                encoded=call('TX',i,lambda:codec.encode(r['family'],tokens,tuple(sorted(set(existing)|{10}))))
                require(all(np.array_equal(encoded[m]['bits'],bits) for m,bits in existing.items()),'Ten-scale TX changed a frozen old prefix')
                decoded=call('RX',i,lambda:codec.decode(r['family'],encoded[10]['bits'],10))
                require(np.array_equal(decoded['received_tokens'],tokens),'Independent m10 actual roundtrip differs')
                arrays['var_m10_bits']=encoded[10]['bits'];arrays['var_m10_received_tokens']=decoded['received_tokens']
                lengths={m:len(b) for m,b in existing.items()};lengths[10]=len(encoded[10]['bits'])
                require(all(choose_arithmetic(lengths,c,minimum_m=4)['status']=='SOURCE_LENGTH_FITS_LAYOUT_PENDING' for c in candidates2048()),
                    'Any missing reachable fallback prefix must be explicitly prepared')
                ap=out/'sources'/f'{i:04d}.npz';ap.parent.mkdir(exist_ok=True)
                with ap.open('xb') as f:np.savez(f,**arrays)
                cp=build_record(source_index=i,source_id=old['source_id'],tokens=tokens,preprocessing_id=old['preprocessing_id'],
                    source_assets_checkpoint=old['source_assets_checkpoint'],archive=ap,arrays=arrays,static_completion_sha=r['static_completion']['sha256'],
                    origins={'EC_STATIC_WHOLE':'NOT_PREPARED_SINGLE_VAR_FAMILY','EC_VAR_WHOLE':'OLD_M4_M9_EXACT_REUSE_PLUS_ACTUAL_M10_TX_RX'},
                    upstream_evidence={f:dict(original_record=entry,m10_TX=pin(out/'calls'/f'{i:04d}_TX.complete.json'),m10_RX=pin(out/'calls'/f'{i:04d}_RX.complete.json')) for f in ('EC_STATIC_WHOLE','EC_VAR_WHOLE')})
                cp.update(N=2048,source_role='calibration',family=r['family'],m10_independent_roundtrip=True,original_source_record=entry,
                    entropy_family_selection=r['entropy_family_selection'],all_48_resource_queries_have_complete_reachable_fallback=True,
                    physical_qualification_required_separately=True)
                target=out/'source_checkpoints'/f'{i:04d}.json';save(target,cp);outputs[str(ap)]=sha(ap);outputs[str(target)]=sha(target)
                records.append(dict(source_index=i,source_id=old['source_id'],checkpoint=str(target),checkpoint_sha256=sha(target)))
                qualifications.append(dict(source_index=i,original_prefixes_byte_exact=sorted(existing),m10_arithmetic_bits=lengths[10],independent_roundtrip=True,zero_extension_reads=30))
                print(f'T6 calibration source {i+1}/{r["source_count"]}; actual m10 length {lengths[10]}',flush=True)
            native.frozen();guard()
        mp=out/'manifest.json';save(mp,dict(status='T6_CALIBRATION_ENTROPY_SOURCE_MANIFEST_COMPLETE',source_count=len(records),family=r['family'],
            original_calibration_source_ids=r['original_calibration_source_ids'],source_ids=[e['source_id'] for e in records],records=records,
            entropy_family_selection=r['entropy_family_selection'],minimum_fallback_m=4,length_field_bits=13,max_source_length8191=True))
        outputs[str(mp)]=sha(mp);save(out/'qualification.json',dict(status='T6_M10_ACTUAL_SOURCE_ROUNDTRIPS_COMPLETE',records=qualifications,
            new_source_count=len(qualifications),prior_source_reuses=len(prior),new_packet_decodes=0,new_metric_calls=0))
        outputs[str(out/'qualification.json')]=sha(out/'qualification.json')
        for p in (out/'calls').glob('*.json'):outputs[str(p)]=sha(p)
        done=dict(status='T6_CALIBRATION_ENTROPY_SOURCE_ASSETS_COMPLETE',request_sha256=rh,source_count=len(records),family=r['family'],
            source_manifest=pin(mp),entropy_family_selection=r['entropy_family_selection'],original_calibration_source_ids=r['original_calibration_source_ids'],
            outputs=outputs,counts=counts,prior_source_reuses=len(prior),new_packet_decodes=0,new_metric_calls=0,new_image_renders=0,holdout_used=False,
            source_population='original_calibration',source_mean_or_policy_evaluation=False,seconds=time.monotonic()-started)
        save(out/'completion.json',done);return dict(status=done['status'],source_count=len(records),counts=counts)
    except BaseException:
        save(out/'owner_failed.json',dict(request_sha256=rh,counts=counts,traceback=traceback.format_exc(),no_auto_retry=True));raise
    finally:save(out/'owner_exited.json',dict(pid=os.getpid(),counts=counts,completed=(out/'completion.json').exists()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare')
    for n in ('root','family-freeze','source-completion','environment-request','out'):q.add_argument('--'+n,required=True)
    q.add_argument('--count',type=int,choices=(100,1000),required=True);q.add_argument('--prior-completion');q.add_argument('--deadline-unix',type=float,required=True)
    q=s.add_parser('run');q.add_argument('--request',required=True);a=p.parse_args()
    print(json.dumps(prepare(a) if a.command=='prepare' else run(a.request)))
