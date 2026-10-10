"""Synthetic-only final export tests: no models, channels, or bootstrap."""
import copy,json,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import t6_final_report as f

FAMILY='EC_VAR_WHOLE'

def fixture(root):
    root=Path(root);ids=[f'synthetic-confirmation-{i}'for i in range(100)];calids=[f'synthetic-calibration-{i}'for i in range(1000)]
    def js(name,value):
        p=root/name;f.s.write(p,value);return f.pin(p)
    def table(name,rows):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);f.csv_write(p,rows);return f.pin(p)
    def wait(name,request,exits):
        return js(name,dict(actual_children_waited=True,request_sha256=request['sha256'],worker_exit_codes=exits))
    selected=js('family.json',dict(family=FAMILY,holdout_used_for_selection=False,holdout_files_read=False))
    calibration={}
    for branch in('raw','entropy'):
        prior=None
        for phase,count,noises in[('pilot',100,[4101]),('full',1000,[4101,4102,4103])]:
            request=dict(packet_cap=200,count=count,schedule=[dict(snr_db=x)for x in f.SNRS],confirmation_images_opened=False,holdout_used_for_selection=False)
            if prior:request['pilot_completion']=prior
            rd=js(f'{branch}_{phase}/request.json',request);wd=wait(f'{branch}_{phase}/owner/completion.json',rd,[0,0])
            status='T6_'+branch.upper()+('_PILOT_CALIBRATION_COMPLETE'if phase=='pilot'else'_FULL1000_CALIBRATION_COMPLETE')
            done=dict(status=status,source_count=count,source_ids=calids[:count],noise_count=len(noises),noise_seeds=noises,
                request=rd,frame_count=100,packet_ledger=dict(total=100,cap=200,unresolved=0),actual_children_waited=True,
                worker_exit_codes=[0,0],outputs={wd['path']:wd['sha256']})
            prior=js(f'{branch}_{phase}/completion.json',done)
        calibration[branch+'_calibration_completion']=prior
    rawcat=js('raw_catalogue.json',dict(profiles=[dict(candidate_id='synthetic-raw',m=10,K=0)]))
    rawreq=js('raw_qualification/request.json',dict(candidate_catalogue=rawcat,packet_cap=2))
    rawq=js('raw_qualification/completion.json',dict(status='PASS',request=rawreq,packet_decode_count=2,ledger=dict(total=2,cap=2,unresolved=0)))
    eq=js('entropy_qualification/request.json',dict(catalogue=dict(profiles=[dict(profile_id=0,m=10)]),
        admitted_candidates=[dict(candidate_id=f'c{i}',admitted=True)for i in range(40)],
        unsupported_candidate_queries=[dict(candidate_id=f'u{i}',admitted=False)for i in range(8)],
        packet_cap=2,family=FAMILY,entropy_family_selection=selected))
    ecq=js('entropy_qualification/completion.json',dict(status='PASS',request=eq,packet_decode_count=2,ledger=dict(total=2,cap=2,unresolved=0)))
    policies={branch:{str(snr):dict(candidate_id=('whole'if branch!='ENTROPY_WHOLE'else'entropy')+str(snr),
        m=10 if snr==19 else 7,target_m=10 if snr==19 else 7,K=0,snr_db=snr)for snr in f.SNRS}
        for branch in('RAW_WHOLE','RAW_PARTIAL','ENTROPY_WHOLE')}
    freeze=dict(status='T6_N2048_POLICIES_FROZEN_CALIBRATION_ONLY_V1',N=2048,source_count=1000,noise_count=3,
        selection_used_confirmation=False,holdout_used_for_selection=False,entropy_family_selection=selected,
        policies=policies,calibration_source_ids=calids,calibration_noise_seeds=[4101,4102,4103],
        raw_phy_qualification=rawq,entropy_phy_qualification=ecq,**calibration)
    fd=js('freeze.json',freeze)
    registration=js('registration.json',dict(schema='WCL_N2048_CONFIRMATION100_METADATA_V1',
        status='SOURCE_IDS_FIXED_BEFORE_ANY_NEW_SOURCE_IMAGE_OR_SCORE_ACCESS',source_count=100,records=[dict(source_id=x)for x in ids]))
    dup=js('content_duplicate_check.json',dict(status='T6_CONFIRMATION100_CONTENT_DUPLICATES_CHECKED_PASS',source_count=100,
        duplicate_source_id_count=0,duplicate_preprocessing_count=0,duplicate_original_JPEG_count=0,confirmation_registration=registration,
        scope='all_available_audited_source_content_hashes',available_reference_hash_count=200,unavailable_reference_hash_count=7,
        Encoder_calls_before_check=0,source_reselection=False))
    source_outputs={};records=[]
    for i,sid in enumerate(ids):
        ap=root/'source'/f'asset{i}.bin';ap.parent.mkdir(parents=True,exist_ok=True);ap.write_bytes(b'SYNTHETIC NO IMAGE OR TOKENS')
        rec=dict(source_id=sid,source_index=i,preprocessing_id=f'pre{i}',evaluation_class_index=i,archive=str(ap),archive_sha256=f.sha(ap))
        cp=js(f'source/cp{i}.json',dict(rec,outputs={str(ap):f.sha(ap)}));records.append(dict(rec,checkpoint=cp['path'],checkpoint_sha256=cp['sha256']))
        source_outputs.update({str(ap):f.sha(ap),cp['path']:cp['sha256']})
    manifest=js('source/manifest.json',dict(schema='T6_CONFIRMATION100_SOURCE_MANIFEST_V1',source_count=100,source_ids=ids,records=records,
        confirmation_registration=registration,calibration_freeze=fd,entropy_family_selection=selected,content_duplicate_check_completion=dup))
    sr=js('source/request.json',dict(calibration_freeze=fd));js('source/owner_started.json',dict(pid=111,request_sha256=sr['sha256']))
    js('source_wait/launch.json',dict(owner_script_sha256=f.s.OBSERVER_SHA,child_pid=111,argv=['python','t6_confirmation_sources.py','run','--request',sr['path']]))
    js('source_wait/exit.json',dict(actual_child_waited=True,exit_code=0))
    source_outputs[manifest['path']]=manifest['sha256']
    sd=js('source/completion.json',dict(status='T6_CONFIRMATION100_SOURCE_ASSETS_COMPLETE',source_count=100,source_ids=ids,
        source_manifest=manifest,request_sha256=sr['sha256'],counts=dict(Encoder_VQ=100,VAR_TX=100,VAR_RX=200),outputs=source_outputs))
    rr=js('render/request.json',dict(source_completion=sd));rw=wait('render/owner/completion.json',rr,[0,0])
    common=dict(N=2048,source_count=100,noise_count=3,frame_count=2700,source_ids=ids,noise_seeds=f.SEEDS,snrs=f.SNRS,
        calibration_freeze=fd,entropy_family_selection=selected,source_manifest=manifest,content_duplicate_check_completion=dup,
        family=FAMILY,methods=f.s.methods(FAMILY),comparisons=f.s.comparisons(FAMILY),selection_used_confirmation=False,holdout_used_for_selection=False)
    render=js('render/completion.json',dict(common,status='T6_CONFIRMATION100_RENDER_COMPLETE',request=rr,actual_children_waited=True,
        worker_exit_codes=[0,0],actual_packet_ledger=dict(total=3500,cap=5400,unresolved=0),new_VAR_render_calls=100,
        qualification_VAR_calls=2,new_VAR_source_decode_calls=0,outputs={rw['path']:rw['sha256']}))
    rows=[];means=[];values={}
    for method in f.s.methods(FAMILY):
        entropy='EC_'in method
        for snr in f.SNRS:
            for i,sid in enumerate(ids):
                quality=[20+i/100+(.1 if entropy else 0),.4-i/1000+(.01 if entropy else 0),.7+i/1000-(.02 if entropy else 0),float(i%2)]
                for seed in f.SEEDS:
                    row=dict(N=2048,method_id=method,point_id=f.s.point(method,snr),snr_db=snr,source_index=i,source_id=sid,noise_seed=seed,
                        status='SYNTHETIC_FAILED_OUTPUT'if i==0 else'SYNTHETIC_ACCEPTED',actual_m=10 if snr==19 else 6 if entropy else 7,
                        target_m=10 if snr==19 else 7,K=0,header_ok=True,body_crc_accept=i!=0,parser_accepted=True,canonical_source_accepted=True,
                        gray=i==0 and entropy,undetected_token_error_diagnostic=i==2,header_symbols=68,body_symbols=1980,padding_symbols=0,E_frame=4096,rho=1,
                        received_state_sha256=f'state-{entropy}-{i}-{snr}-{seed}',image_sha256=f'image-{entropy}-{i}-{snr}-{seed}')
                    row.update(zip(f.METRICS,quality));rows.append(row)
                for metric,value in zip(f.METRICS,quality):
                    means.append(dict(N=2048,method_id=method,point_id=f.s.point(method,snr),snr_db=snr,metric=metric,
                        source_index=i,source_id=sid,source_count=100,noise_count=3,mean=value));values[method,snr,metric,i]=value
    per=table('score/per_frame.csv',rows);scr=js('score/request.json',dict(render_completion=render))
    js('score/owner_started.json',dict(pid=222,request_sha256=scr['sha256']))
    sl=js('score_wait/launch.json',dict(owner_script_sha256=f.s.OBSERVER_SHA,child_pid=222,argv=['python','t6_score.py','run','--request',scr['path']]))
    se=js('score_wait/exit.json',dict(actual_child_waited=True,exit_code=0))
    score=js('score/completion.json',dict(common,status='T6_CONFIRMATION100_FOUR_METRICS_COMPLETE',actual_children_waited=True,worker_exit_codes=[0],
        request_path=scr['path'],request_sha256=scr['sha256'],parent_launch=sl,parent_exit=se,counts=dict(new_quality_calls=200),outputs={per['path']:per['sha256']}))
    def interval(method,snr,metric,reference=None):
        vals=[values[method,snr,metric,i]-(values[reference,snr,metric,i]if reference else 0)for i in range(100)]
        mu=math.fsum(vals)/100;delta=0 if all(x==0 for x in vals)else .001;lo=mu-delta;hi=mu+delta;factor=100 if metric==f.METRICS[-1]else 1
        row=dict(N=2048,method_id=method,snr_db=snr,metric=metric,mean=mu,ci_low=lo,ci_high=hi,source_count=100,noise_count=3,frame_count=300,
            display_mean=mu*factor,display_ci_low=lo*factor,display_ci_high=hi*factor,display_scale=factor,
            display_unit=('percentage_points'if reference else'percent')if factor==100 else'metric',improvement_direction=('negative'if metric=='lpips_alex'else'positive')if reference else'higher')
        if reference:row.update(reference_method_id=reference,delta_definition='method minus reference',frame_level_noise_pairing_claimed=False,
            interval_direction='exact_zero_difference'if lo==hi==0 else'interval_includes_zero'if lo<=0<=hi else'method_better'if(hi<0 if metric=='lpips_alex'else lo>0)else'reference_better')
        return row
    summary=[interval(m,n,k)for m in f.s.methods(FAMILY)for n in f.SNRS for k in f.METRICS]
    paired=[interval(m,n,k,r)for m,r in f.s.comparisons(FAMILY)for n in f.SNRS for k in f.METRICS]
    outputs={}
    for name,values0 in [('summary.csv',summary),('paired.csv',paired),('source_means.csv',means),('failure_breakdown.csv',[dict(status='synthetic',frames=2700)])]:
        d=table('stats/'+name,values0);outputs[d['path']]=d['sha256']
    for name in('points.json','pairs.json'):
        d=js('stats/'+name,dict(synthetic=True));outputs[d['path']]=d['sha256']
    strq=js('stats/request.json',dict(score_completion=score))
    stats=js('stats/completion.json',dict(common,status='T6_N2048_CONFIRMATION100_SOURCE_PAIRED_STATISTICS_COMPLETE',request=strq,
        source_mean_first=True,summary_rows=36,paired_rows=36,source_mean_rows=3600,bootstrap_seed=2026100701,bootstrap_replicates=10000,
        multiple_comparison_adjustment=False,outputs=outputs))
    protocol=root/'synthetic_protocol.md';protocol.write_text('Synthetic fixture. No scientific execution.',encoding='utf-8')
    return SimpleNamespace(calibration_freeze=fd['path'],source_completion=sd['path'],source_wait_record=str(root/'source_wait'),
        render_completion=render['path'],score_completion=score['path'],statistics_completion=stats['path'],protocol=[str(protocol)],
        out=str(root/'export'),render_owner_receipt=None),rows,means,summary,paired,ids,freeze

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.args,self.rows,self.means,self.summary,self.paired,self.ids,self.freeze=fixture(self.tmp.name)
    def test_full_export_retains_2700_failures_and_zero_deltas(self):
        done=f.run(self.args);self.assertEqual(done['frame_count'],2700);self.assertEqual(done['new_bootstrap_calls'],0)
        out=Path(self.args.out);self.assertEqual(f.sha(out/'paired.csv'),f.sha(Path(self.tmp.name)/'stats/paired.csv'))
        self.assertEqual(len(f.csv_read(out/'per_frame.csv')),2700)
        mech=f.csv_read(out/'mechanism_and_failure_summary.csv');self.assertTrue(any(x['m10_K0_frames']=='300'for x in mech))
        self.assertEqual(sum(x['partial_same_policy_as_whole']=='True'for x in mech),3)
        self.assertTrue(any(x['interval_direction']=='exact_zero_difference'for x in f.csv_read(out/'paired.csv')))
        text=(out/'REPORT_T6.md').read_text();self.assertIn('not that these images were never encountered anywhere',text)
        self.assertIn('Negative LPIPS is better',text);self.assertIn('percentage points',text)
    def test_missing_frame_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'All2700'):f.s.frame_grid(self.rows[:-1],self.ids,FAMILY,scored=True)
    def test_missing_source_mean_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'All3600'):f.source_mean_check(self.means[:-1],self.rows,self.summary,self.paired,self.ids,FAMILY)
    def test_lpips_sign_and_agreement_unit_corruption_are_rejected(self):
        bad=copy.deepcopy(self.paired);next(x for x in bad if x['metric']=='lpips_alex')['improvement_direction']='positive'
        with self.assertRaisesRegex(RuntimeError,'direction'):f.intervals(self.summary,bad,FAMILY)
        bad=copy.deepcopy(self.paired);next(x for x in bad if x['metric']==f.METRICS[-1])['display_unit']='relative_percent'
        with self.assertRaisesRegex(RuntimeError,'Agreement units'):f.intervals(self.summary,bad,FAMILY)
    def test_source_mean_or_zero_interval_rewrite_is_rejected(self):
        bad=copy.deepcopy(self.means);bad[0]['mean']+=.1
        with self.assertRaisesRegex(RuntimeError,'source mean differs'):f.source_mean_check(bad,self.rows,self.summary,self.paired,self.ids,FAMILY)
        bad=copy.deepcopy(self.paired);bad[0]['ci_high']=.001
        with self.assertRaisesRegex(RuntimeError,'zero intervals'):f.source_mean_check(self.means,self.rows,self.summary,bad,self.ids,FAMILY)
    def test_sealed_file_tamper_stops_before_output(self):
        p=Path(self.tmp.name)/'score/per_frame.csv';p.write_text(p.read_text()+'tampered\n')
        with self.assertRaisesRegex(RuntimeError,'Changed bound input'):f.run(self.args)
        self.assertFalse(Path(self.args.out).exists())
    def test_parent_wait_and_observer_must_be_real(self):
        p=Path(self.tmp.name)/'source_wait/launch.json';v=f.read(p);v['owner_script_sha256']='fake';p.write_text(json.dumps(v))
        with self.assertRaisesRegex(RuntimeError,'actually exited0'):f.run(self.args)
    def test_identical_whole_partial_cannot_have_different_reconstruction(self):
        bad=copy.deepcopy(self.rows);next(r for r in bad if r['method_id']==f.s.RAW_PARTIAL)['image_sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'different reported results'):f.mechanism_rows(bad,self.freeze,FAMILY)

if __name__=='__main__':unittest.main()
