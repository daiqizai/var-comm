"""Export closed scientific evidence and executable configuration, without inference."""
from pathlib import Path
import sys,json,csv,re,shutil
import numpy as np
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,pin,checked,asha,digest,OFFSETS
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');L=B/'nestcom_v9_completion_20261101_v1';D=R/'results/nestcom_v9_completion_20261101_v1'
def rows(p):return list(csv.DictReader(p.open()))
def csvout(p,rr):
    fields=list(dict.fromkeys(k for r in rr for k in r))
    with p.open('x',newline='')as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rr)
def main():
    assert read(L/'token_rx_control/actual_wait.json')['returncode']==0
    jobs={v['key']:v['projection']for v in read(L/'token_rx_plan/jobs.json')};rx={v['key']:v for v in read(L/'token_rx/outputs.json')};ss=read(L/'sources.json')
    ref=read(L/'token_rx_plan/reference_request.json');physical={v['physical_key']:v for v in read(L/'token_phy/frames.json')}
    partial=next(v for v in read(L/'token_frame_plan/logical.json')if v['endpoint'][1]>0 and physical[v['physical_key']]['received_key']not in read(L/'token_rx_plan/reuse.json'))
    choices=[('whole',ref['key'],ref['source_index']),('partial',physical[partial['physical_key']]['received_key'],partial['source_index'])];vectors=[]
    for label,key,index in choices:
        j=jobs[key];bits=np.array(j['bits'],np.uint8);s=ss[index];v=rx[key]
        trace=read(L/'token_rx/cdf_traces'/(digest(dict(family=j['family'],m=j['m'],K=j['K'],payload_sha256=asha(bits)))+'.json'))
        tx=read(Path(s['archive']['path']).parent/'source_output_fixed.json')['cdf_trace'][:j['m']+bool(j['K'])]
        assert trace['cdf_trace']==tx and trace['canonical']and trace['zero_extension_reads']==30
        assert not trace['receiver_truth_used']and not trace['transmitter_probability_table_used']
        with np.load(checked(v['image']),allow_pickle=False)as z:received=z['received_tokens'].copy()
        with np.load(checked(s['archive']),allow_pickle=False)as z:truth=z['tokens'].reshape(-1)[:OFFSETS[j['m']]+j['K']].copy()
        assert np.array_equal(received,truth)and asha(received)==trace['received_tokens_sha256']
        vectors.append((label,bits,received,dict(endpoint=[j['m'],j['K']],family=j['family'],source_id=s['source_id'],payload_bits=len(bits),payload_sha256=asha(bits),received_tokens_sha256=asha(received),TX_CDF=tx,RX_CDF=trace['cdf_trace'],canonical=True,zero_extension_reads=30,exact_tokens=True,receiver_TX_truth_used=False,receiver_TX_CDF_used=False,source=pin(Path(s['archive']['path']).parent/'source_output_fixed.json'),RX_trace=pin(L/'token_rx/cdf_traces'/(digest(dict(family=j['family'],m=j['m'],K=j['K'],payload_sha256=asha(bits)))+'.json')),numeric_identity=read(L/'token_source/actual_native_identity.json'))))
    pool=[];layoutrows=[]
    cats={n:read(R/f'results/nestcom_opt_20261030_v1/grid_catalogues/N{n}.json')for n in [512,1024]}
    layouts={}
    for n,cat in cats.items():
        for p in cat['profiles']:
            k=(n,p['q'],p['nominal_rate'])
            if k in layouts:continue
            layouts[k]=p
            row=dict(N=n,q=p['q'],rate=p['nominal_rate'],source_cap=p['source_capacity_bits'],admitted=p['layout_status'])
            row.update({k:json.dumps(v,sort_keys=True)if isinstance(v,(dict,list))else v for k,v in p['layout'].items()});layoutrows.append(row)
    protpath=R/'results/nestcom_n512_20261028_v1/protocol.json';p=read(protpath)
    finalists={(v['method'],int(v['snr_db']),v['candidate_id'])for v in rows(R/'results/nestcom_n512_20261028_v1/full_calibration/candidate_summary.csv')}
    winners={(v['method'],v['snr_db'],v['selected']['candidate_id'])for v in read(R/'results/nestcom_n512_20261028_v1/frozen_policies.json')['selection']}
    for snr in [4,10,19]:
        for c in p['candidates']:
            key=(c['method'],snr,c['candidate_id']);lp=layouts[512,c['q'],c['rate']]
            pool.append(dict(budget=512,snr_db=snr,family=c['method'],candidate_id=c['candidate_id'],target_m=c['target'][0],target_K=c['target'][1],q=c['q'],rate=c['rate'],source_cap=lp['source_capacity_bits'],admitted=True,screen=True,final=key in finalists,winner=key in winners,objective='equal-source DINOv2-L cosine; gray retained',tie_rule='candidate ID lexical',screen_sources=100,screen_repeats=1,final_sources=1000,final_repeats=3,registration=str(protpath.relative_to(R)),origin='registered 396 candidates; three SNR evaluations'))
    pp=R/'results/t1b_m10_calibration_20261016_v1/delivery/frozen_policies.json';p=read(pp)
    for snr,families in p['policies'].items():
        for family,v in families.items():
            for cid in v['eligible']:
                m=int(re.search(r'm(\d+)',cid)[1]);km=re.search(r'_K(\d+)',cid);q=int(re.search(r'_q(\d+)',cid)[1]);a,b=re.search(r'_r(\d+)-(\d+)',cid).groups();rate=a+'/'+b
                pool.append(dict(budget=1024,snr_db=int(snr),family=family,candidate_id=cid,target_m=m,target_K=int(km[1])if km else 0,q=q,rate=rate,source_cap=layouts[1024,q,rate]['source_capacity_bits'],admitted=True,screen='legacy pilot'if not cid.startswith('m10_')else 'prespecified m10 extension',final=True,winner=cid==v['winner'],objective='equal-source DINOv2-L cosine; gray retained',tie_rule='legacy first; candidate ID lexical',screen_sources=100,screen_repeats=1,final_sources=1000,final_repeats=3,registration=str(pp.relative_to(R)),origin='actual inherited finalists plus m10; not full Token Cartesian recalibration'))
    p7=R/'results/nestcom_7db_20261030_v1/protocol.json';v7=read(p7);w7=read(R/'results/nestcom_7db_20261030_v1/frozen_policies.json')['methods']
    for family in ['NEST','WHOLE','ADA','ARPC']:
        for q,rate in v7['MCS']:
            pool.append(dict(budget=1024,snr_db=7,family=family,candidate_id=f'q{q}_r{rate.replace("/","-")}',target_m={'ADA':256,'ARPC':7}.get(family,10),target_K=0,q=q,rate=rate,source_cap=layouts[1024,q,rate]['source_capacity_bits'],admitted=True,screen=True,final=True,winner=(q,rate)==(w7[family]['q'],w7[family]['rate']),objective='equal-source DINOv2-L cosine; gray retained',tie_rule='MCS declaration order',screen_sources=100,screen_repeats=1,final_sources=100,final_repeats=1,registration=str(p7.relative_to(R)),origin='fixed max endpoint; five MCS'))
    out=D/'reproduction';out.mkdir(exist_ok=False);(out/'reference_vectors').mkdir()
    for label,bits,tokens,meta in vectors:
        path=out/'reference_vectors'/(label+'.npz')
        with path.open('xb')as f:np.savez_compressed(f,bits=bits,received_tokens=tokens)
        meta['vector']=dict(file=path.name,**pin(path));save(path.with_suffix('.json'),meta)
    csvout(out/'candidate_pool.csv',pool);csvout(out/'phy_layouts.csv',layoutrows)
    config=dict(schema='NESTCOM_V9_REPRODUCTION',repository_commit='dde679637b30f1166bc540d4c7b6fa91fe457e8c',asset_root_environment='NESTCOM_ASSET_ROOT',asset_override_json='NESTCOM_ASSETS_JSON',N=[512,1024],header_symbols=68,nominal_Es=2,header=dict(profile_bits=12,crc_poly='0x1021',crc_init='0xffff',crc_order='MSB first',convolutional_octal=['171','133'],constraint_length=7,tail_bits=6,terminated_viterbi=True,rate_matching='68 mother bits repeated to 136 bits',modulation='68 complex rotated BPSK symbols +/- (1+j)'),body=dict(length_bits=13,crc_bits=16,crc_covers='length+source+known zero pad',rate_matching='RV0 one block; filler shortening; first2Z and tail puncturing; no repetition',scramble='PCG64 seeded from SHA256 canonical [protocol,scramble,session,counter], first16 bytes little endian; RX inverse LLR signs',decoder=layoutrows[0]['decoder_config'],mapping=layoutrows[0]['implementation']),cdf=dict(V=4096,total=2**24,logp='FP32 torch.log_softmax to NumPy float64 stable exp normalization',frequency='floor(p*(M-V))+1; remainder to first argmax',dtype='int64',same_scale_truth=False),arithmetic=dict(interval_bits=32,initial_low=0,initial_high=2**32-1,termination='b=(low<2^30?0:1) followed by pending+1 inverse bits; non-mutating finish',zero_extension=30,canonical_bit_equality=True),policies='candidate_pool.csv lists actual family pools; Token inherits selected target/MCS and refines fallback only',controls=dict(E2='WHOLE/Token max(10,0), six total bit budgets; no PHY',E3='WHOLE/Quarter/Eighth/Token max(8,126), q4 r2/3 shared noise'),paid_null=dict(protocol=read(L/'paid_null_regression/catalogue.json')['protocol'],validation=read(L/'paid_null_regression/completion.json'),extension=read(L/'paid_null_regression/catalogue.json')['null_extension']),native_identity=read(L/'token_source/actual_native_identity.json'),metric_identities={r:read(L/r/'identity.json')for r in ['token_three','token_clip','kodak_dists']})
    save(out/'reproduction_config.json',config)
    routes=read(R/'results/checkpoint_relocation_20261028_v1/model_routes.json');ada=read(R/'results/adatoken_response_20261026_v1/upstream_assets.json')
    manifest=dict(schema='MODEL_FILE_AND_STATE_IDENTITIES_DISTINCT_V1',NeST=dict(routes=routes,upstream='https://github.com/FoundationVision/VAR',revision='8d0f5fc80cf03491a7156a84043c4f6cac560339',weights_url='https://huggingface.co/FoundationVision/var/tree/main',Dc=dict(update=38000,loss='MSE+0.1 LPIPS',mixture=[.5,.25,.25],mixture_inputs=['continuous','quantized','interpolated prefix-completed'],selection='minimum full-calibration mixture image loss',recipe=read(R/'experiments/var-latent-enhancement-20260917/configs/experiment.json'),availability='Research checkpoint retained in project PFS; not an upstream public download. Supply matching file via asset root or train using the published recipe. No public weight URL claimed.'),numerics=config['native_identity']),Ada=dict(upstream=ada,recipe='frozen LlamaGen-L/VQ16, paid ConvNeXt predicted class; CFG4/T1/topk2000/topp1; FP32 TF32off; entropy uses conditional branch, not CFG'),additional_identity_sources=[])
    # Preserve exact baseline identities without guessing absent state hashes.
    for rel in ['paper/manuscripts/nestcom_wcl_focused_20261029_v6/inputs/training/swin_selected_checkpoint.json','paper/manuscripts/nestcom_wcl_focused_20261029_v6/inputs/training/swin_training_audit.json','paper/manuscripts/nestcom_wcl_focused_20261029_v6/inputs/training/swin_continue_protocol.json','results/unified_metrics_20261002/modelmanifest.json']:
        if (R/rel).exists():manifest['additional_identity_sources'].append(dict(source=pin(R/rel),content=read(R/rel)))
    save(out/'model_manifest.json',manifest)
    for role in ['paid_null','token_source','token_phy','token_rx','token_three','token_clip','kodak_dists']:
        cp=L/(role+'_control');dest=D/'execution_receipts'/role;dest.mkdir(parents=True,exist_ok=True)
        for name in ['actual_wait.json','launch.json','failure.json']:
            p=cp/name
            if p.exists():shutil.copyfile(p,dest/name)
        work=L/('paid_null_regression'if role=='paid_null'else role)
        for name in ['completion.json','failure.json','identity.json','actual_native_identity.json']:
            p=work/name
            if p.exists():shutil.copyfile(p,dest/('worker_'+name))
    save(out/'completion.json',dict(status='EXPORTED',candidate_rows=len(pool),unique_N512_candidates=396,layouts=len(layoutrows),reference_vectors=2,TX_RX_CDF_exact=True,new_scientific_calls=0))
    print(read(out/'completion.json'))
if __name__=='__main__':main()
