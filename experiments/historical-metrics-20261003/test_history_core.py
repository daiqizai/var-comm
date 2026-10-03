"""CPU engineering evidence only; fixtures are never real image results."""
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from history_common import (NEW_METRICS,row_ids,identity,checkpoint_valid,normalize_metadata,write,read,sha,write_csv,
                            native_interop_setup,metric_numerical_context)
from history_analysis import context,group_means,Bootstrap,metrics_for,reference_map,direction,analyse,collect,coverage_rows
from history_score import ordered_payload,first_source_receipt


class FakeTorch:
    """Only process flag controls; this fixture has no tensor/metric operations."""
    def __init__(self):
        self.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=True,benchmark=False,deterministic=False))
        self.precision='highest';self.deterministic=False;self.warn=True
        self.threads=4;self.interop=2;self.interop_calls=0
    def set_float32_matmul_precision(self,value):self.precision=value
    def get_num_threads(self):return self.threads
    def set_num_threads(self,value):self.threads=value
    def get_num_interop_threads(self):return self.interop
    def set_num_interop_threads(self,value):
        if self.interop_calls:raise RuntimeError('Cannot set a second time')
        self.interop_calls+=1;self.interop=value
    def use_deterministic_algorithms(self,value,*,warn_only):self.deterministic=value;self.warn=warn_only
    def is_deterministic_algorithms_warn_only_enabled(self):return self.warn
    @staticmethod
    def flags(t):
        return dict(matmul_tf32=t.backends.cuda.matmul.allow_tf32,cudnn_tf32=t.backends.cudnn.allow_tf32,
            precision=t.precision,cudnn_benchmark=t.backends.cudnn.benchmark,
            cudnn_deterministic=t.backends.cudnn.deterministic,deterministic=t.deterministic,
            threads=t.threads,interop_threads=t.interop)


class HistoricalCoreTests(unittest.TestCase):
    def setUp(self):
        self.original=[dict(method='P',value='1.0')]
        self.ids=row_ids('STUDY',0,self.original)
        self.evaluation=dict(modelmanifest_sha256='model',evaluator_identity='evaluator',
            metric_batch_size=1,batch_qualification_sha256='batch',numerical_runtime={'tf32':False})
        self.check=dict(expected_rows=self.original,study='STUDY',source_index=0,
            evaluation_identity=self.evaluation,qualified_batch_sizes=[1,2])

    def seal(self,p):
        p['payload_sha256']=identity({k:v for k,v in p.items() if k!='payload_sha256'})
        return p

    def payload(self):
        row=dict(self.original[0],history_row_id=self.ids[0],history_replay_parity_passed=True,
            history_original_row_sha256=identity(self.original[0]),history_study='STUDY',
            history_source_index=0,history_modelmanifest_sha256='model',**{'new_'+k:0 for k in NEW_METRICS})
        return self.seal(dict(binding='abc',source_index=0,rows=[row],
            parity=[dict(history_row_id=self.ids[0],replay_parity_passed=True,synthetic=False)],
            evaluation_identity=copy.deepcopy(self.evaluation),metric_batch_sizes_used=[1],unique_images=1))

    def validate(self,p,ids=None,binding='abc'):
        return checkpoint_valid(p,binding,self.ids if ids is None else ids,**self.check)

    def test_duplicate_rows_remain_distinct_and_exact(self):
        ids=row_ids('STUDY',0,self.original*2);self.assertNotEqual(*ids)
        self.assertNotEqual(ids[0],row_ids('OTHER',0,self.original)[0])
        self.assertNotEqual(ids[0],row_ids('STUDY',1,self.original)[0])
        self.assertNotEqual(ids[0],row_ids('STUDY',0,[dict(method='P',value='1')])[0])

    def test_fixed_metric_flags_restore_native_on_success_and_error(self):
        t=FakeTorch();native=t.flags(t)
        fixed=dict(native,cudnn_tf32=False,cudnn_deterministic=True,deterministic=True,threads=6)
        with metric_numerical_context(t,fixed,t.flags):
            self.assertEqual(t.flags(t),fixed);self.assertFalse(t.warn)
        self.assertEqual(t.flags(t),native);self.assertTrue(t.warn)
        with self.assertRaisesRegex(ValueError,'metric failure'):
            with metric_numerical_context(t,fixed,t.flags):raise ValueError('metric failure')
        self.assertEqual(t.flags(t),native);self.assertTrue(t.warn)

    def test_metric_flag_mutation_rejected_and_native_restored(self):
        t=FakeTorch();native=t.flags(t);fixed=dict(native,cudnn_tf32=False)
        with self.assertRaisesRegex(RuntimeError,'changed its registered'):
            with metric_numerical_context(t,fixed,t.flags):t.backends.cudnn.allow_tf32=True
        self.assertEqual(t.flags(t),native)

    def test_interop_initialized_once_allows_only_same_native_request(self):
        t=FakeTorch();t.interop=8;setter=t.set_num_interop_threads
        with native_interop_setup(t,2):
            t.set_num_interop_threads(2);t.set_num_interop_threads(2)
            self.assertEqual(t.interop_calls,1)
            with self.assertRaisesRegex(RuntimeError,'different interop'):t.set_num_interop_threads(4)
        self.assertEqual(t.set_num_interop_threads,setter)

    def test_interop_wrapper_restored_after_native_exception(self):
        t=FakeTorch();setter=t.set_num_interop_threads
        with self.assertRaises(ValueError):
            with native_interop_setup(t,2):raise ValueError('native failure')
        self.assertEqual(t.set_num_interop_threads,setter)

    def test_valid_receipt_and_coverage_checks(self):
        p=self.payload();self.validate(p)
        for ids in [[],self.ids*2,['wrong']]:
            with self.assertRaises(RuntimeError):self.validate(p,ids)
        with self.assertRaises(RuntimeError):self.validate(p,binding='wrong')
        p['rows'][0]['value']='changed'
        with self.assertRaises(RuntimeError):self.validate(p)

    def test_original_substitution_rejected_even_after_resealing(self):
        p=self.payload();p['rows'][0]['value']='2.0'
        p['rows'][0]['history_original_row_sha256']=identity(dict(method='P',value='2.0'))
        with self.assertRaisesRegex(RuntimeError,'exact original'):self.validate(self.seal(p))

    def test_original_numeric_type_preserved(self):
        self.original[0]['value']=1;self.ids=row_ids('STUDY',0,self.original)
        p=self.payload();p['rows'][0]['value']=1.0
        p['rows'][0]['history_original_row_sha256']=identity(dict(method='P',value=1.0))
        with self.assertRaisesRegex(RuntimeError,'exact original'):self.validate(self.seal(p))

    def test_actual_parity_required_and_order_bound(self):
        for field,value in [('synthetic',True),('replay_parity_passed',False),('history_row_id','other')]:
            p=self.payload();p['parity'][0][field]=value
            with self.assertRaises(RuntimeError):self.validate(self.seal(p))

    def test_evaluator_and_batch_identity_checked(self):
        for field,value in [('evaluator_identity','wrong'),('modelmanifest_sha256','wrong'),
                            ('metric_batch_size',2),('batch_qualification_sha256','changed')]:
            p=self.payload();p['evaluation_identity'][field]=value
            with self.assertRaisesRegex(RuntimeError,'evaluator/batch'):self.validate(self.seal(p))
        for batches,unique in [([3],1),([2],1),([],0),([True],1)]:
            p=self.payload();p.update(metric_batch_sizes_used=batches,unique_images=unique)
            with self.assertRaisesRegex(RuntimeError,'metric batch'):self.validate(self.seal(p))

    def test_additional_metric_coverage_required(self):
        p=self.payload();del p['rows'][0]['new_dists']
        with self.assertRaisesRegex(RuntimeError,'additional metric'):self.validate(self.seal(p))

    def test_ordered_payload_cannot_drop_or_duplicate(self):
        rows=[{'history_row_id':'b'},{'history_row_id':'a'}]
        actual,proof=ordered_payload(rows,copy.deepcopy(rows),['a','b'])
        self.assertEqual([r['history_row_id'] for r in actual],['a','b'])
        with self.assertRaises(RuntimeError):ordered_payload(rows,proof,['a'])

    def metadata(self,**changes):
        return dict(label_conditioned=False,reference_only=False,oracle=False,**changes)

    def row(self,source='source',value='0.2',**changes):
        row=dict(history_study='OLD',history_source_id=source,history_source_index=0,
            history_reference_sha256='target-'+source,history_row_id='row-'+source,
            history_replay_parity_passed=True,method='candidate',snr_equiv_db='-5',lpips=value,
            history_metadata_json=json.dumps(self.metadata(snr_field='snr_equiv_db',
                original_metric_columns={'lpips_alex':'lpips'})))
        row.update(changes);return row

    def test_method_id_priority_and_canonical_model_identity(self):
        metadata=self.metadata(method_id='registered',method='metadata_method',model_identity={'var':'a','vae':'b'})
        c=context(self.row(history_metadata_json=json.dumps(metadata)))
        self.assertEqual(c['method'],'registered')
        self.assertEqual(c['model_id'],identity({'vae':'b','var':'a'}))
        self.assertEqual(c['model_id'],c['model_identity_sha256'])

    def test_receiver_reference_derivation_explicit(self):
        m=dict(label_conditioned=False,oracle=False,output_role='main',scope='historical_development')
        value=normalize_metadata({'method':'P'},m)
        self.assertFalse(value['reference_only']);self.assertIn('metadata_derivations',value)
        m['scope']='source_only_reference'
        self.assertTrue(normalize_metadata({'method':'P'},m)['reference_only'])
        m['output_role']='unknown'
        with self.assertRaises(RuntimeError):normalize_metadata({'method':'P'},m)

    def test_alias_and_equivalent_noise_explicit(self):
        row=self.row();c=context(row)
        self.assertEqual(c['snr_db'],'-5');self.assertEqual(c['snr_definition'],'snr_equiv_db')
        physical=self.row(snr_db='-5.0',history_metadata_json=json.dumps(self.metadata()))
        self.assertEqual(context(physical)['snr_db'],c['snr_db'])
        self.assertNotEqual(context(physical)['snr_definition'],c['snr_definition'])
        self.assertEqual(group_means([row],'lpips_alex'),{'source':.2})

    def test_no_channel_legacy_resource_sentinels_keep_raw_rows(self):
        metadata=dict(label_conditioned=False,reference_only=True,oracle=False,
            channel_uses_not_applicable=True,N='',E='',snr_db='',
            snr_definition='not_applicable_no_channel')
        original=self.row(N='not_applicable_source_reference',E='not_applicable_source_reference',
            snr_db='not_applicable_source_reference',history_metadata_json=json.dumps(metadata))
        before=copy.deepcopy(original)
        normalized=normalize_metadata(original,metadata)
        self.assertEqual((normalized['N'],normalized['E'],normalized['snr_db']),('','',''))
        self.assertEqual((context(original)['N'],context(original)['snr_db']),('',''))
        covered=coverage_rows({'old':[original]},{'old':context(original)})[0]
        self.assertEqual((covered['E_mean'],covered['E_min'],covered['E_max'],covered['energy_frames']),('','','',0))
        self.assertEqual(original,before)
        channel=self.row(N='1024',E='2048')
        covered=coverage_rows({'channel':[channel]},{'channel':context(channel)})[0]
        self.assertEqual((covered['N'],covered['E_mean'],covered['energy_frames']),('1024',2048.,1))
        with self.assertRaises(ValueError):
            coverage_rows({'bad':[self.row(E='not_applicable_source_reference')]},{'bad':context(self.row())})

    def test_noise_averaged_within_source(self):
        values=group_means([self.row('a','0.1'),self.row('a','0.3'),self.row('b','0.9')],'lpips_alex')
        self.assertAlmostEqual(values['a'],.2)
        self.assertAlmostEqual(Bootstrap(replicates=200).interval(list(values.values()))['mean'],.55)

    def test_missing_nonfinite_not_silently_dropped(self):
        for value in ['','nan']:
            with self.assertRaises(RuntimeError):group_means([self.row('a'),self.row('b',value)],'lpips_alex')
        self.assertIsNone(group_means([self.row()],'absent'))

    def test_original_f_and_mismatch_available(self):
        row=self.row(F_mse='0.4',dino_mismatched='.1',dino_specificity='.3')
        for name in ('F_mse','dino_mismatched','dino_specificity'):self.assertIn(name,metrics_for([row]))
        self.assertEqual(direction('F_mse'),'lower');self.assertEqual(direction('dino_mismatched'),'diagnostic')
        self.assertEqual(direction('dino_specificity'),'higher')

    def test_inconsistent_reference_pixels_rejected(self):
        with self.assertRaises(RuntimeError):reference_map([self.row(),self.row(history_reference_sha256='other')])

    def test_annotations_required(self):
        with self.assertRaises(RuntimeError):context(self.row(history_metadata_json='{}'))

    def fixture(self,root):
        queue=root/'queue.json';coverage=root/'coverage.json';write(coverage,{'pending':['OLD']})
        out=root/'results/historical_metrics_20261003/OLD';out.mkdir(parents=True)
        rows=[self.row('a','0.1',F_mse='.4'),self.row('b','0.3',F_mse='.6')]
        for row in rows:row.update({'new_'+k:0 for k in NEW_METRICS})
        table=out/'metrics_per_frame.csv';write_csv(table,rows)
        done=root/'outputs/HISTORICAL-METRICS-20261003/OLD/completion.json'
        write(done,dict(status='HISTORICAL_STUDY_METRICS_COMPLETE',parity_passed=True,synthetic=False,
            training_updates=0,policy_selection_updates=0,inputs={},outputs={str(table):sha(table)},frames=2,sources=2))
        prior=root/'results/unified_metrics_20261002';prior.mkdir(parents=True)
        for name in ('metrics_summary.csv','metrics_paired_intervals.csv','METRICS_REPORT.md'):
            (prior/name).write_text('unchanged original report',encoding='utf-8')
        write(queue,dict(source_bindings={},jobs=[dict(study='OLD',adapter='fixture')],contrasts=[],
            coverage_manifest={'path':str(coverage),'sha256':sha(coverage)}))
        return queue,table,done

    def test_analysis_prior_index_separate_full_scope_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);queue,table,done=self.fixture(root);analyse(root,queue)
            out=root/'results/historical_metrics_20261003';receipt=read(out/'analysis_completion.json')
            self.assertTrue(receipt['registered_queue_complete']);self.assertFalse(receipt['full_history_coverage_complete'])
            self.assertFalse(read(out/'prior_six_study_index.json')['recomputed'])
            with (out/'summary.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            self.assertEqual({r['study'] for r in rows},{'OLD'})
            self.assertAlmostEqual(float(next(r for r in rows if r['metric']=='F_mse')['mean']),.5)

    def test_duplicate_export_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);queue,table,done=self.fixture(root)
            with table.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            rows[1]['history_row_id']=rows[0]['history_row_id'];write_csv(table,rows)
            proof=read(done);proof['outputs'][str(table)]=sha(table);write(done,proof)
            with self.assertRaisesRegex(RuntimeError,'Duplicate'):collect(root,read(queue)['jobs'])

    def add_contrast(self,root,queue,table,done,*,physical=False,bad_reference=False):
        with table.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        additional=[]
        for row in rows:
            value=dict(row,method='baseline',history_row_id=row['history_row_id']+'-base',lpips='0.0')
            if physical:
                value['snr_db']='-5'
                value['history_metadata_json']=json.dumps(self.metadata(original_metric_columns={'lpips_alex':'lpips'}))
            if bad_reference:value['history_reference_sha256']='different-'+row['history_source_id']
            additional.append(value)
        write_csv(table,rows+additional)
        proof=read(done);proof['frames']=4;proof['outputs'][str(table)]=sha(table);write(done,proof)
        reg=read(queue);reg['contrasts']=[dict(name='registered_minus_base',
            A={'study':'OLD','method':'candidate'},B={'study':'OLD','method':'baseline'},match_fields=['snr_db'])]
        write(queue,reg)

    def test_registered_pair_uses_source_means(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);queue,table,done=self.fixture(root)
            self.add_contrast(root,queue,table,done);analyse(root,queue)
            with (root/'results/historical_metrics_20261003/paired.csv').open(newline='',encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            pair=next(r for r in rows if r['metric']=='lpips_alex')
            self.assertAlmostEqual(float(pair['delta_mean']),.2);self.assertEqual(pair['sources'],'2')

    def test_new_no_channel_m8_matches_legacy_prefix_resources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);queue,table,done=self.fixture(root)
            self.add_contrast(root,queue,table,done)
            with table.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            for row in rows:
                fresh=row['method']=='candidate'
                row.update(N='' if fresh else 'not_applicable_source_reference',
                    E='' if fresh else 'not_applicable_source_reference',
                    snr_db='' if fresh else 'not_applicable_source_reference')
                row['history_metadata_json']=json.dumps(dict(label_conditioned=False,reference_only=True,oracle=False,
                    channel_uses_not_applicable=True,N='',E='',snr_db='',decoder='Dc',
                    snr_definition='not_applicable_no_channel',original_metric_columns={'lpips_alex':'lpips'}))
            write_csv(table,rows)
            proof=read(done);proof['outputs'][str(table)]=sha(table);write(done,proof)
            reg=read(queue);reg['contrasts'][0]['match_fields']=['N','snr_db','decoder'];write(queue,reg)
            analyse(root,queue)
            with (root/'results/historical_metrics_20261003/paired.csv').open(newline='',encoding='utf-8') as stream:
                paired=list(csv.DictReader(stream))
            self.assertAlmostEqual(float(next(r for r in paired if r['metric']=='lpips_alex')['delta_mean']),.2)
            with (root/'results/historical_metrics_20261003/coverage.csv').open(newline='',encoding='utf-8') as stream:
                covered=list(csv.DictReader(stream))
            self.assertTrue(all(r['N']==r['snr_db']==r['E_mean']=='' and r['energy_frames']=='0' for r in covered))

    def test_pair_never_mix_physical_equivalent_snr(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);queue,table,done=self.fixture(root)
            self.add_contrast(root,queue,table,done,physical=True)
            with self.assertRaisesRegex(RuntimeError,'no compatible'):analyse(root,queue)

    def test_pair_requires_identical_reference_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);queue,table,done=self.fixture(root)
            self.add_contrast(root,queue,table,done,bad_reference=True)
            with self.assertRaisesRegex(RuntimeError,'different source images'):analyse(root,queue)

    def test_first_source_requires_actual_full_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);checkpoint=out/'source.json';regpath=out/'registration.json'
            p=self.payload();write(checkpoint,p);write(regpath,{'fixture':True})
            first_source_receipt(out,checkpoint,regpath,p)
            self.assertEqual(read(out/'first_source_qualification.json')['evaluation_identity'],self.evaluation)
            p['parity'][0]['synthetic']=True
            with self.assertRaises(RuntimeError):first_source_receipt(out,checkpoint,regpath,p)


if __name__=='__main__':unittest.main()
