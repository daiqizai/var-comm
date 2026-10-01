"""Pure CPU, temporary-fixture tests of publication/qualification boundaries."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location('_delivery_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    try:import fcntl
    except ModuleNotFoundError:fcntl=types.ModuleType('fcntl')
    with patch.dict(sys.modules,fcntl=fcntl):spec.loader.exec_module(module)
    return module


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.pub=load('publish');self.sup=load('supervisor')
        for module in [self.pub,self.sup]:
            module.ROOT=self.root;module.OUT=self.root/'outputs';module.OUT.mkdir(exist_ok=True)
        self.pub.RESULT=self.root/'results';self.pub.RESULT.mkdir()
        (self.root/'.gitignore').write_text('')
    def tearDown(self):self.temp.cleanup()

    def test_analysis_requires_real_complete_and_unchanged_all_artifacts(self):
        inp=self.root/'input.csv';out=self.root/'summary.csv';inp.write_text('a\n1\n');out.write_text('b\n2\n')
        receipt=dict(status='COMPLETE',synthetic=False,training_updates=0,
            inputs={str(inp):self.pub.sha(inp)},outputs={str(out):self.pub.sha(out)})
        self.pub.verify_analysis(receipt)
        for field,value in [('status','RUNNING'),('synthetic',True),('training_updates',1)]:
            bad=copy.deepcopy(receipt);bad[field]=value
            with self.assertRaises(RuntimeError):self.pub.verify_analysis(bad)
        out.write_text('b\n3\n')
        with self.assertRaises(RuntimeError):self.pub.verify_analysis(receipt)

    def test_full_screen_export_source_grid_and_registration_checks(self):
        stage='m1_screen';folder=self.pub.OUT/stage/'cells';folder.mkdir(parents=True)
        reg=dict(source_ids=[f's{i}' for i in range(1000)],preprocessing_ids=[f'p{i}' for i in range(1000)],
            calibration_or_development=stage,training_updates=0,input_artifacts={})
        regpath=self.pub.OUT/(stage+'_registration.json');self.pub.write(regpath,reg);identity=self.pub.sha(regpath)
        dest=self.pub.RESULT/'m1_shortlist.json';self.pub.write(dest,dict(registration_sha256=identity))
        done=dict(status='COMPLETE',sources=200,training_updates=0,development_read=False,rows=185000,policy_sha256=self.pub.sha(dest))
        self.pub.write(self.pub.OUT/(stage+'_complete.json'),done)
        template=[dict(N=N,phy_family=fam,snr_db=snr,noise_seed=4101,action_id=f'mock{N}/{fam}/{j}')
            for snr in [1,4,7,13,19] for N,fam,count in [(512,'QPSK',22),(512,'16QAM',47),(1024,'QPSK',47),(1024,'16QAM',69)]
            for j in range(count)]
        for i in range(200):
            rows=[dict(source_index=i,source_id=f's{i}',preprocessing_id=f'p{i}',**r) for r in template]
            self.pub.write(folder/f'{i:04d}.json',dict(identity=identity,rows=rows))
        self.pub.export_cells(stage)
        archive=self.pub.read(self.pub.RESULT/'provenance'/f'{stage}_archive.json')
        self.assertEqual(archive['rows'],185000);self.assertEqual(archive['cells'],200)
        target=folder/'0000.json';cell=self.pub.read(target);cell['rows'][0]['source_index']=1;self.pub.write(target,cell)
        with self.assertRaises(RuntimeError):self.pub.export_cells(stage)
        self.assertFalse((self.pub.RESULT/'m1_screen_per_frame.tmp').exists())
        cell['rows'][0]['source_index']=0;cell['identity']='different';self.pub.write(target,cell)
        with self.assertRaises(RuntimeError):self.pub.export_cells(stage)
        target.unlink()
        with self.assertRaises(RuntimeError):self.pub.export_cells(stage)

    def test_large_csv_exact_restoration_and_published_link_indexes(self):
        csvpath=self.pub.RESULT/'m1_large.csv';header=b'index,text\r\n'
        with csvpath.open('wb') as f:
            f.write(header)
            for i in range(180000):f.write((str(i)+','+'x'*50+'\r\n').encode())
        source_hash=self.pub.sha(csvpath)
        manifest=self.pub.archive_tables('m1');entry=manifest['m1_large.csv']
        self.assertEqual(entry['rows'],180000);self.assertEqual(entry['sha256'],source_hash)
        restored=bytearray(header)
        for part in entry['parts']:
            with (self.pub.RESULT/part['path']).open('rb') as f:
                self.assertEqual(f.readline(),header);restored.extend(f.read())
        self.assertEqual(bytes(restored),csvpath.read_bytes())
        self.assertTrue((self.pub.RESULT/entry['index']).exists())
        self.assertIn('/results/m1_large.csv',(self.root/'.gitignore').read_text())
        original=self.pub.RESULT/'m1_report.md'
        original.write_text('[raw](m1_large.csv)\n[small](m1_summary.csv)\n![image](figures/a.png)\n[remote](https://example.com/a)\n')
        original_hash=self.pub.sha(original);target=self.root/'reports'/'report.md'
        content=self.pub.published_report(original,target,manifest)
        self.assertIn('../results/m1_table_shards/m1_large.csv.index.md',content)
        self.assertIn('../results/m1_summary.csv',content);self.assertIn('../results/figures/a.png',content)
        self.assertIn('https://example.com/a',content);self.assertEqual(self.pub.sha(original),original_hash)

    def test_qualification_exact_final_source_dependencies_and_report_sha(self):
        source=self.root/'runner.py';vendor=self.root/'vendor.py';source.write_text('frozen');vendor.write_text('frozen vendor')
        qualified={str(p):self.sup.sha(p) for p in [source,vendor]}
        report=self.root/'qualification_report.json'
        self.pub.write(report,dict(status='REAL_WEIGHT_QUALIFICATION_PASS',training_updates=0,development_read=False,
            calibration_sources=4,qualification_source_bindings=qualified))
        receipt=dict(status='REAL_WEIGHT_QUALIFICATION_PASS',training_updates=0,development_read=False,
            qualification_path=str(report),qualification_sha256=self.sup.sha(report),source_bindings=qualified)
        self.pub.write(self.sup.OUT/'qualification.json',receipt)
        publication=dict(source_bindings={str(source):qualified[str(source)]})
        self.assertEqual(len(self.sup.verify_qualification(publication,{str(vendor):qualified[str(vendor)]})),2)
        with self.assertRaises(RuntimeError):self.sup.verify_qualification(publication,{})
        vendor.write_text('changed')
        with self.assertRaises(RuntimeError):self.sup.verify_qualification(publication,{str(vendor):qualified[str(vendor)]})
        vendor.write_text('frozen vendor');report.write_text(report.read_text()+' ')
        with self.assertRaises(RuntimeError):self.sup.verify_qualification(publication,{str(vendor):qualified[str(vendor)]})

    def test_publication_receipts_are_explicitly_real_and_recovery_allows_descendants(self):
        record=dict(checks='PASS',commit='abc',remote_commit='abc')
        self.pub.mark_complete('m1',record);self.pub.mark_complete('m2',record)
        self.assertIs(self.pub.read(self.pub.OUT/'m1_complete.json')['synthetic'],False)
        self.assertIs(self.pub.read(self.pub.OUT/'completion.json')['synthetic'],False)
        with patch.object(self.pub.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)) as run:
            self.pub.verify_previous_publication(record)
        self.assertEqual(run.call_args_list[-1].args[0],['git','merge-base','--is-ancestor','abc','origin/main'])
        with patch.object(self.pub.subprocess,'run',return_value=types.SimpleNamespace(returncode=1)):
            with self.assertRaises(RuntimeError):self.pub.verify_previous_publication(record)


if __name__=='__main__':unittest.main()
