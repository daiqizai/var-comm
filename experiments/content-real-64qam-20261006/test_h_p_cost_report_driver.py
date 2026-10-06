import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import h_p_cost_report as projection
import h_p_cost_report_driver as d
from test_h_p_cost_report import fixture


def write(p,value):
    d.save(p,value);return str(p)


def fixture_batches(root):
    hp,h,t=fixture();ids=[f'source{i}' for i in range(100)]
    batches={}
    for key,value in [('H',h),('HP',hp),('timing',t)]:
        p=root/key;p.mkdir();done=copy.deepcopy(value['completion']);done['outputs']={};done['input_bindings']={}
        batches[key]=dict(science=done,data_dir=p,cfg={})
    def add(key,name,value):
        b=batches[key];p=b['data_dir']/name
        if name.endswith('.csv'):d.csv_write(p,value)
        else:d.save(p,value)
        b['science']['outputs'][str(p)]=d.sha(p);return p
    hplan=dict(catalog=h['point_catalog'],comparisons=[dict(pair=i) for i in range(26)],selection=False)
    plan=dict(hp['comparison_plan'],selection=False)
    add('H','comparison_plan.json',hplan)
    for name in ('point_catalog','tx_source_scale','rx_received_scale','receiver_status'):add('H',name+'.csv',h[name])
    for name in ('summary','paired','classifier_transition','wire_diagnostics'):add('H',name+'.csv',[dict(original='0.12345678901234567890')])
    for name in ('metric_provenance','cost_scope'):add('H',name+'.json',{'original_scope':'fixture'})
    for name in ('summary','paired'):add('HP',name+'.csv',hp[name])
    add('HP','point_identity.json',hp['point_identity']);add('HP','metric_availability.json',[])
    pp=add('HP','comparison_plan.json',plan);ap=add('HP','common_metric_admission.json',hp['common_metric_admission'])
    metric_batch=dict(config=str(root/'original_metric_config.json'),owner_config='original owner',registration='original reg',
                      launch='original launch',completion='original completion')
    req=write(root/'hp_request.json',{'H_request':{'metric_batch':metric_batch}})
    batches['H']['cfg']['metric_config']=metric_batch['config']
    batches['HP']['cfg'].update(request=req,comparison_plan=str(pp),admission=str(ap))
    batches['timing']['cfg']['metric_batch']=metric_batch
    hpd=batches['HP']['science'];hpd.update(source_ids=ids,admitted_metric_count=18,missing_P_metric_count=3,
        comparison_plan_sha256=d.sha(pp),common_metric_admission_sha256=d.sha(ap))
    add('HP','normalization_proof.json',dict(source_ids=ids,metrics_recomputed=False,comparisons_selected=False,
        frame_level_noise_pairing_claimed=False))
    pixel=root/'original_pixel_evidence.bin';pixel.write_bytes(b'no tensor; only SHA evidence fixture')
    hpd['input_bindings'].update(d.bind([pixel]));bridge=[]
    for i,sid in enumerate(ids):bridge.append(dict(source_index=i,source_id=sid,pixel_equality='EXACT_UINT8_AND_FLOAT32_CONVERSION',
        shape=[3,256,256],channel_order='RGB_CHW',new_metric_calls=0,reference_sha256='a'*64,
        original_reference_sha256={'H':'b'*64,'P':'c'*64},branch_input_bindings={'H':d.bind([pixel]),'P':d.bind([pixel])}))
    add('HP','target_pixel_bridge.json',bridge)
    td=batches['timing']['science'];td.update(source_ids=[ids[i] for i in projection.FIXED16],registration_sha256='e'*64)
    for i in projection.FIXED16:
        rp=add('timing',f'sources/{i:04d}.json',[dict(component_case=j) for j in range(18)])
        add('timing',f'source_checkpoints/{i:04d}.json',dict(status='H_ONLINE_COMPONENT_SOURCE_COMPLETE',source_index=i,
            source_id=ids[i],registration_sha256='e'*64,component_case_count=18,noise_seed=6201,
            timing_repetitions_are_quality_samples=False,outputs=d.bind([rp]),input_bindings={}))
    add('timing','component_summary.json',t['component_summary'])
    add('timing','historical_phy_event_windows.json',t['historical_phy_event_windows'])
    return batches


class Driver(unittest.TestCase):
    def test_metadata_gate_never_reads_quality_or_cost_values(self):
        with tempfile.TemporaryDirectory() as td:
            b=fixture_batches(Path(td))
            with patch.object(d,'csv_read',side_effect=AssertionError('quality read forbidden')):
                value=d.metadata_admission(b)
            self.assertFalse(value['quality_rows_read']);self.assertFalse(value['component_values_read'])

    def test_mismatched_pixel_bridge_or_sources_refused(self):
        for kind in ('pixel','timing_source'):
            with tempfile.TemporaryDirectory() as td:
                b=fixture_batches(Path(td))
                if kind=='timing_source':b['timing']['science']['source_ids'][0]='different'
                else:
                    p=b['HP']['data_dir']/'target_pixel_bridge.json';v=d.read(p);v[0]['pixel_equality']='same label only'
                    p.write_text(__import__('json').dumps(v),encoding='utf-8');b['HP']['science']['outputs'][str(p)]=d.sha(p)
                with self.assertRaises(RuntimeError):d.metadata_admission(b)

    def test_missing_timing_source_checkpoint_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            b=fixture_batches(Path(td));p=b['timing']['data_dir']/'source_checkpoints/0000.json'
            del b['timing']['science']['outputs'][str(p)]
            with self.assertRaisesRegex(RuntimeError,'not sealed'):d.metadata_admission(b)

    def test_normal_batch_does_not_silently_recover_failed_owner(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);mods={k:write(root/(k+'_module.py'),{}) for k in ('owner','cpu','wait')}
            cfgp=root/'config.json';regp=root/'registration.json';op=root/'owner.json';lp=root/'launch.json';cp=root/'completion.json'
            for p in (op,lp,cp):write(p,{})
            write(cfgp,dict(registration=str(regp),owner_config=str(op)))
            write(regp,dict(source_bindings=d.bind(mods.values()),input_bindings=d.bind([cfgp])))
            spec=dict(config=str(cfgp),owner_config=str(op),registration=str(regp),launch=str(lp),completion=str(cp))
            req=dict(batches={'HP':spec},modules=mods,source_bindings=d.bind(mods.values()),input_bindings=d.bind(spec.values()))
            api=SimpleNamespace(raw_process_state=lambda p:None)
            cpu=SimpleNamespace(closed_batch=Mock(side_effect=RuntimeError('original owner failed')))
            with self.assertRaisesRegex(RuntimeError,'original owner failed'):d.normal_batch(req,'HP',api,cpu,object())
            self.assertEqual(cpu.closed_batch.call_count,1)

    def test_original_hash_change_refused_before_loading_modules(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bound.py';p.write_text('# original');pins=d.bind([p]);p.write_text('# changed')
            with patch.object(d,'module',side_effect=AssertionError('import too early')):
                with self.assertRaisesRegex(RuntimeError,'Changed binding'):
                    d.preflight(dict(schema='H_P_COST_REPORT_REQUEST_V1',batches=dict(H={},HP={},timing={}),
                        modules=dict(owner='',cpu='',wait='',projection=''),source_bindings=pins,input_bindings={}))
            original_open=Path.open;opens=[]
            def counted(path,*args,**kwargs):
                if path==p and args and args[0]=='rb':opens.append(str(path))
                return original_open(path,*args,**kwargs)
            with patch.object(Path,'open',counted),d.hashing_session():
                first=d.sha(p);self.assertEqual(first,d.sha(p));self.assertEqual(len(opens),1)
                p.write_text('# another value changes stat and content');self.assertNotEqual(first,d.sha(p))
                self.assertEqual(len(opens),2)
            self.assertIsNone(d._HASH_CACHE)

    def test_owner_precreated_empty_output_allowed_but_no_retry(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'out';p.mkdir();d.claim(p,'a'*64)
            with self.assertRaisesRegex(RuntimeError,'previous attempts'):d.claim(p,'a'*64)

    def test_export_preserves_original_csv_bytes_and_source_timing_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);b=fixture_batches(root);out=root/'export';out.mkdir()
            originals={p:d.sha(p) for x in b.values() for p in x['science']['outputs']}
            d.export(dict(batches=b,projection=projection),out)
            for p,s in originals.items():self.assertEqual(d.sha(p),s)
            for item in d.read(out/'copy_inventory.json')['files'].values():
                self.assertEqual(item['original_sha256'],item['export_sha256'])
            self.assertIn('不含 Encoder/VQ', (out/'REPORT.md').read_text(encoding='utf-8'))
            self.assertIn('excludes Encoder/VQ',d.read(out/'component_definitions.json')['TX_source_total'])
            self.assertFalse((out/'completion.json').exists())

    def test_no_extra_H_or_MAIN_budget_consumption(self):
        b=dict(charged=164760,phase_charged={'development':10800},development_remaining=2400,failed=0,unresolved=0)
        d.budget_ok(b)
        for k,v in [('charged',164761),('development_remaining',2399),('failed',1),('unresolved',1)]:
            with self.assertRaises(RuntimeError):d.budget_ok(dict(b,**{k:v}))


if __name__=='__main__':unittest.main()
